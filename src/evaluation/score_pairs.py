from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

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


def calculate_scores(first_features: np.ndarray, second_features: np.ndarray,
                     metric: str) -> np.ndarray:
    if metric == "cosine":
        denominator = np.maximum(
            np.linalg.norm(first_features, axis=1)
            * np.linalg.norm(second_features, axis=1),
            1e-12,
        )
        return np.sum(first_features * second_features, axis=1) / denominator
    if metric == "manhattan":
        return -np.sum(np.abs(first_features - second_features), axis=1)
    raise ValueError(f"Unsupported matching metric: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-name", required=True)
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))

    feature_dir = resolve_path(config["output"]["features_dir"]) / args.feature_name
    features, mapping = load_feature_store(feature_dir)
    matching_metric = load_matching_metric(feature_dir)
    print(f"Matching metric: {matching_metric}")
    pairs_dir = resolve_path(config["output"]["pairs_dir"])
    scores_root = resolve_path(config["output"]["scores_dir"]) / args.feature_name

    for protocol_name in config["protocols"]:
        out_dir = scores_root / protocol_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for partition in ["development", "final_test"]:
            pairs = pd.read_csv(pairs_dir / protocol_name / f"{partition}.csv", dtype={"image_id_1": "string", "image_id_2": "string"})
            scores = np.empty(len(pairs), dtype=np.float32)
            for start in tqdm(range(0, len(pairs), args.chunk_size), desc=f"{args.feature_name} {protocol_name} {partition}", unit="chunk"):
                stop = min(start + args.chunk_size, len(pairs))
                chunk = pairs.iloc[start:stop]
                first_idx = np.asarray([mapping[str(x)] for x in chunk["image_id_1"]], dtype=np.int64)
                second_idx = np.asarray([mapping[str(x)] for x in chunk["image_id_2"]], dtype=np.int64)
                scores[start:stop] = calculate_scores(
                    np.asarray(features[first_idx], dtype=np.float32),
                    np.asarray(features[second_idx], dtype=np.float32),
                    metric=matching_metric,
                )
            pairs["similarity"] = scores
            pairs.to_csv(out_dir / f"{partition}.csv", index=False)


if __name__ == "__main__":
    main()
