from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_feature_store(feature_dir: Path):
    index = pd.read_csv(feature_dir / "index.csv", dtype={"image_id": "string"})
    features = np.load(feature_dir / "features.npy", mmap_mode="r")
    mapping = {str(image_id): int(row_index) for image_id, row_index in zip(index["image_id"], index["row_index"])}
    return features, mapping


def load_matching_metric(feature_dir: Path) -> str:
    metadata_path = feature_dir / "metadata.json"
    if not metadata_path.exists():
        return "cosine"
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return str(metadata.get("matching_metric", "cosine")).lower()


def similarity_matrix(probes: np.ndarray, gallery: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        probes = probes / np.maximum(np.linalg.norm(probes, axis=1, keepdims=True), 1e-12)
        gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
        return probes @ gallery.T
    if metric == "manhattan":
        return -cdist(probes, gallery, metric="cityblock")
    raise ValueError(f"Unsupported matching metric: {metric}")


def evaluate_partition(frame: pd.DataFrame, image_column: str, subject_column: str,
                       mapping: dict, features: np.ndarray, matching_metric: str) -> dict:
    counts = frame.groupby(subject_column).size()
    eligible = counts[counts >= 2].index.astype(str).tolist()
    working = frame.loc[frame[subject_column].astype(str).isin(eligible)].copy()
    gallery_rows, probe_rows = [], []

    for _, group in working.groupby(subject_column):
        group = group.sort_values(image_column)
        gallery_rows.append(group.iloc[0])
        probe_rows.extend(row for _, row in group.iloc[1:].iterrows())

    if not gallery_rows or not probe_rows:
        raise ValueError("Not enough images for closed-set identification")

    gallery = pd.DataFrame(gallery_rows)
    probes = pd.DataFrame(probe_rows)
    gallery_idx = np.asarray([mapping[str(x)] for x in gallery[image_column]], dtype=np.int64)
    probe_idx = np.asarray([mapping[str(x)] for x in probes[image_column]], dtype=np.int64)
    similarity = similarity_matrix(
        np.asarray(features[probe_idx], dtype=np.float32),
        np.asarray(features[gallery_idx], dtype=np.float32),
        metric=matching_metric,
    )
    gallery_subjects = gallery[subject_column].astype(str).to_numpy()
    probe_subjects = probes[subject_column].astype(str).to_numpy()
    ranking = np.argsort(-similarity, axis=1)
    rank1 = float(np.mean(gallery_subjects[ranking[:, 0]] == probe_subjects))
    k = min(5, len(gallery_subjects))
    rank5 = float(np.mean([
        probe_subjects[i] in set(gallery_subjects[ranking[i, :k]])
        for i in range(len(probe_subjects))
    ]))
    return {
        "eligible_subjects": int(len(eligible)),
        "gallery_images": int(len(gallery)),
        "probe_images": int(len(probes)),
        "rank_1_recognition_rate": rank1,
        "rank_5_recognition_rate": rank5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-name", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    feature_dir = resolve_path(config["output"]["features_dir"]) / args.feature_name
    protocol_dir = resolve_path(config["input"]["protocol_dir"])
    output_dir = resolve_path(config["output"]["identification_dir"]) / args.feature_name
    output_dir.mkdir(parents=True, exist_ok=True)
    features, mapping = load_feature_store(feature_dir)
    matching_metric = load_matching_metric(feature_dir)
    print(f"Matching metric: {matching_metric}")
    image_column = config["columns"]["image_id"]
    subject_column = config["columns"]["subject_id"]
    split_column = config["columns"]["split"]
    results = {}

    for protocol_name in config["protocols"]:
        frame = pd.read_csv(protocol_dir / f"{protocol_name}.csv", dtype={image_column: "string", subject_column: "string"}, low_memory=False)
        results[protocol_name] = {}
        for partition in ["development", "final_test"]:
            part = frame.loc[frame[split_column] == partition].copy()
            try:
                results[protocol_name][partition] = evaluate_partition(
                    part, image_column, subject_column, mapping, features,
                    matching_metric,
                )
            except ValueError as error:
                results[protocol_name][partition] = {"status": "SKIPPED", "reason": str(error)}

    with (output_dir / "identification_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
