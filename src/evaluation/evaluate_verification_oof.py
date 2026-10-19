from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def infer_protocol(path: Path, protocols: list[str]) -> str | None:
    text = str(path).lower()
    for protocol in protocols:
        if protocol.lower() in text:
            return protocol
    return None


def normalize_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "similarity" not in frame.columns and "score" in frame.columns:
        frame = frame.rename(columns={"score": "similarity"})

    if "label" not in frame.columns and "pair_type" in frame.columns:
        pair_type = frame["pair_type"].astype(str).str.lower().str.replace("-", "_")
        frame["label"] = np.where(pair_type.eq("genuine"), 1,
                            np.where(pair_type.isin(["cotwin", "co_twin", "co-twin"]), 0, np.nan))
    return frame


def discover_final_scores(scores_root: Path, feature_name: str, protocols: list[str]) -> dict[str, pd.DataFrame]:
    collected: dict[str, list[pd.DataFrame]] = {p: [] for p in protocols}
    for path in scores_root.rglob("*.csv"):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if frame.empty:
            continue

        path_has_feature = feature_name.lower() in str(path).lower()
        column_has_feature = (
            "feature_name" in frame.columns
            and frame["feature_name"].astype(str).str.lower().eq(feature_name.lower()).any()
        )
        if not path_has_feature and not column_has_feature:
            continue

        frame = normalize_score_frame(frame)
        if "similarity" not in frame.columns:
            continue

        if "feature_name" in frame.columns:
            frame = frame.loc[frame["feature_name"].astype(str).str.lower() == feature_name.lower()].copy()
        if "partition" in frame.columns:
            frame = frame.loc[frame["partition"].astype(str).str.lower().isin(["final_test", "test", "final-test"])].copy()
        if "task" in frame.columns:
            task = frame["task"].astype(str).str.lower()
            frame = frame.loc[task.str.contains("twin")].copy()
        if "pair_type" in frame.columns:
            pt = frame["pair_type"].astype(str).str.lower().str.replace("-", "_")
            frame = frame.loc[pt.isin(["genuine", "cotwin", "co_twin"])].copy()
        if "label" in frame.columns:
            frame = frame.loc[frame["label"].isin([0, 1])].copy()
        if frame.empty:
            continue

        if "protocol" in frame.columns:
            for protocol, group in frame.groupby("protocol"):
                protocol = str(protocol)
                if protocol in collected:
                    collected[protocol].append(group.copy())
        else:
            protocol = infer_protocol(path, protocols)
            if protocol:
                collected[protocol].append(frame.copy())

    result = {}
    for protocol, frames in collected.items():
        if frames:
            result[protocol] = pd.concat(frames, ignore_index=True)
    return result


def eer_value(labels: np.ndarray, scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def tar_at_far(labels: np.ndarray, scores: np.ndarray, target_far: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    valid = np.where(fpr <= target_far)[0]
    return 0.0 if len(valid) == 0 else float(np.max(tpr[valid]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-name", default="twin_aware_facenet_frontal")
    parser.add_argument("--scores-root", default=None,
                        help="Defaults to output.scores_dir from the config.")
    args = parser.parse_args()

    with resolve_path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    cfg = config["twin_aware_facenet"]

    protocols = [
        "p0_controlled_neutral_frontal",
        "p1_controlled_expression",
        "p2_frontal_illumination",
        "p3_frontal_all",
        "p4_moderate_pose",
        "p5_profile",
        "p6_all_usable",
    ]

    calibration_dir = resolve_path(config["output"]["calibration_dir"])
    thresholds_path = calibration_dir / cfg["oof"]["thresholds_file"]
    with thresholds_path.open("r", encoding="utf-8") as file:
        thresholds = json.load(file)

    scores_root = resolve_path(args.scores_root or config["output"]["scores_dir"])
    final_scores = discover_final_scores(scores_root, args.feature_name, protocols)

    missing = [p for p in protocols if p not in final_scores]
    if missing:
        raise FileNotFoundError(
            "Could not find FINAL_TEST score CSVs for these protocols: "
            + ", ".join(missing)
            + f"\nSearched recursively under: {scores_root}\n"
              "Make sure score_pairs has already been run for the feature name, or pass --scores-root explicitly."
        )

    rows = []
    for protocol in protocols:
        frame = normalize_score_frame(final_scores[protocol])
        labels = frame["label"].astype(int).to_numpy()
        scores = frame["similarity"].astype(float).to_numpy()
        threshold = float(thresholds[protocol]["threshold"])
        predictions = (scores >= threshold).astype(int)

        genuine_mask = labels == 1
        impostor_mask = labels == 0
        false_rejects = int(np.sum((predictions == 0) & genuine_mask))
        false_accepts = int(np.sum((predictions == 1) & impostor_mask))
        frr = false_rejects / max(1, int(genuine_mask.sum()))
        far = false_accepts / max(1, int(impostor_mask.sum()))

        test_eer, test_eer_threshold = eer_value(labels, scores)
        row = {
            "feature_name": args.feature_name,
            "protocol": protocol,
            "task": "twin_verification",
            "partition": "final_test",
            "threshold_source": "OOF_DEVELOPMENT",
            "threshold": threshold,
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "far": float(far),
            "frr": float(frr),
            "eer": test_eer,
            "eer_threshold_test_only": test_eer_threshold,
            "auc": float(roc_auc_score(labels, scores)),
            "tar_at_far_1pct": tar_at_far(labels, scores, 0.01),
            "tar_at_far_0_1pct": tar_at_far(labels, scores, 0.001),
            "genuine_pairs": int(genuine_mask.sum()),
            "cotwin_pairs": int(impostor_mask.sum()),
            "oof_development_eer": float(thresholds[protocol]["oof_eer"]),
        }
        rows.append(row)
        print(
            f"{protocol}: accuracy={row['accuracy']*100:.2f}% "
            f"EER={row['eer']*100:.2f}% FAR={row['far']*100:.2f}% FRR={row['frr']*100:.2f}%"
        )

    output = calibration_dir / "verification_summary_oof.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nSaved OOF-calibrated FINAL_TEST summary to: {output}")


if __name__ == "__main__":
    main()
