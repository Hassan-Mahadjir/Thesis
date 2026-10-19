from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def protocol_ids(protocols_dir: Path, protocol: str, image_id_column: str) -> set[str]:
    path = protocols_dir / f"{protocol}.csv"
    frame = pd.read_csv(path, dtype={image_id_column: "string"}, low_memory=False)
    return set(normalize_identifier(frame[image_id_column]).dropna().tolist())


def sample_pairs(pairs: list[tuple[str, str]], maximum: int, rng: random.Random):
    if len(pairs) <= maximum:
        return pairs
    result = rng.sample(pairs, maximum)
    result.sort()
    return result


def make_pairs(
    frame: pd.DataFrame,
    image_col: str,
    subject_col: str,
    mz_col: str,
    max_genuine: int,
    max_cotwin: int,
    seed: int,
):
    rng = random.Random(seed)
    genuine = []
    cotwin = []

    for subject, group in frame.groupby(subject_col):
        ids = sorted(group[image_col].dropna().astype(str).tolist())
        if len(ids) >= 2:
            genuine.extend((a, b, str(subject), str(subject)) for a, b in sample_pairs(
                list(itertools.combinations(ids, 2)), max_genuine, rng
            ))

    valid = frame.loc[
        frame[mz_col].notna()
        & (~frame[mz_col].astype("string").isin(["", "<NA>", "nan", "None"]))
    ].copy()
    for _, mz_group in valid.groupby(mz_col):
        subjects = sorted(mz_group[subject_col].dropna().astype(str).unique().tolist())
        for s1, s2 in itertools.combinations(subjects, 2):
            ids1 = sorted(mz_group.loc[mz_group[subject_col].astype(str) == s1, image_col].astype(str).tolist())
            ids2 = sorted(mz_group.loc[mz_group[subject_col].astype(str) == s2, image_col].astype(str).tolist())
            candidates = [(a, b) for a in ids1 for b in ids2]
            cotwin.extend((a, b, s1, s2) for a, b in sample_pairs(candidates, max_cotwin, rng))
    return genuine, cotwin


def eer_and_threshold(labels: np.ndarray, scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with resolve_path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    cfg = config["twin_aware_facenet"]
    columns = config["columns"]
    protocols = [
        "p0_controlled_neutral_frontal",
        "p1_controlled_expression",
        "p2_frontal_illumination",
        "p3_frontal_all",
        "p4_moderate_pose",
        "p5_profile",
        "p6_all_usable",
    ]

    features_root = resolve_path(config["output"]["features_dir"]) / cfg["oof"]["features_name"]
    protocols_dir = resolve_path(config["input"]["protocols_dir"])
    calibration_dir = resolve_path(config["output"]["calibration_dir"])
    scores_dir = calibration_dir / "oof_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {}

    for protocol_index, protocol in enumerate(protocols):
        allowed_ids = protocol_ids(protocols_dir, protocol, columns["image_id"])
        rows = []

        for fold in [int(x) for x in cfg["oof"]["folds"]]:
            fold_dir = features_root / f"fold_{fold}"
            frame = pd.read_csv(fold_dir / "index.csv", low_memory=False)
            frame[columns["image_id"]] = normalize_identifier(frame[columns["image_id"]])
            frame[columns["subject_id"]] = normalize_identifier(frame[columns["subject_id"]])
            frame[columns["monozygotic_group_id"]] = normalize_identifier(frame[columns["monozygotic_group_id"]])
            frame = frame.loc[frame[columns["image_id"]].isin(allowed_ids)].copy().reset_index(drop=True)
            features = np.load(fold_dir / "features.npy")

            # index.csv was saved in the same row order as features.npy before protocol filtering.
            full_index = pd.read_csv(fold_dir / "index.csv", low_memory=False)
            full_index[columns["image_id"]] = normalize_identifier(full_index[columns["image_id"]])
            image_to_row = {str(image_id): i for i, image_id in enumerate(full_index[columns["image_id"]].tolist())}

            genuine, cotwin = make_pairs(
                frame,
                columns["image_id"],
                columns["subject_id"],
                columns["monozygotic_group_id"],
                int(cfg["max_genuine_pairs_per_subject"]),
                int(cfg["max_cotwin_pairs_per_subject_pair"]),
                int(cfg["oof"]["seed"]) + protocol_index * 100 + fold,
            )

            for pair_type, label, pairs in [("genuine", 1, genuine), ("cotwin", 0, cotwin)]:
                for image_1, image_2, subject_1, subject_2 in pairs:
                    if image_1 not in image_to_row or image_2 not in image_to_row:
                        continue
                    score = float(np.dot(features[image_to_row[image_1]], features[image_to_row[image_2]]))
                    rows.append({
                        "protocol": protocol,
                        "fold": fold,
                        "task": "twin_verification",
                        "pair_type": pair_type,
                        "label": label,
                        "image_id_1": image_1,
                        "image_id_2": image_2,
                        "subject_id_1": subject_1,
                        "subject_id_2": subject_2,
                        "similarity": score,
                    })

        score_frame = pd.DataFrame(rows)
        if score_frame.empty or score_frame["label"].nunique() != 2:
            raise ValueError(f"Insufficient OOF pairs for {protocol}.")

        score_frame.to_csv(scores_dir / f"{protocol}.csv", index=False)
        labels = score_frame["label"].to_numpy(dtype=np.int64)
        scores = score_frame["similarity"].to_numpy(dtype=np.float64)
        eer, threshold = eer_and_threshold(labels, scores)
        auc = float(roc_auc_score(labels, scores))

        thresholds[protocol] = {
            "threshold": threshold,
            "oof_eer": eer,
            "oof_auc": auc,
            "genuine_pairs": int((score_frame["label"] == 1).sum()),
            "cotwin_pairs": int((score_frame["label"] == 0).sum()),
        }
        print(f"{protocol}: OOF EER={eer*100:.2f}% threshold={threshold:.6f} AUC={auc*100:.2f}%")

    calibration_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = calibration_dir / cfg["oof"]["thresholds_file"]
    with thresholds_path.open("w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=2)
    print(f"\nSaved OOF thresholds to: {thresholds_path}")


if __name__ == "__main__":
    main()
