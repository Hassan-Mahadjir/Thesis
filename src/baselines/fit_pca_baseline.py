from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_vector(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return (image.astype(np.float32) / 255.0).reshape(-1)


def batch_vectors(frame: pd.DataFrame, image_column: str, size: int, batch_size: int):
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start:start + batch_size]
        yield np.stack([read_vector(resolve_path(path), size) for path in batch[image_column]], axis=0)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return (matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    cfg = config["pca"]
    if not bool(cfg.get("enabled", False)):
        print("PCA disabled.")
        return

    df = pd.read_csv(resolve_path(config["input"]["assignments_csv"]), low_memory=False)
    image_id_column = config["columns"]["image_id"]
    split_column = config["columns"]["split"]
    image_column = cfg["image_column"]
    size = int(cfg["resize"])
    components = int(cfg["components"])
    batch_size = int(cfg["batch_size"])
    development = df.loc[df[split_column] == "development"].reset_index(drop=True)
    output_dir = resolve_path(config["output"]["features_dir"]) / "pca"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fitting PCA on DEVELOPMENT only: {len(development):,} images")
    pca = IncrementalPCA(n_components=components, batch_size=batch_size)
    for batch in tqdm(batch_vectors(development, image_column, size, batch_size), desc="PCA fit"):
        if len(batch) >= components:
            pca.partial_fit(batch)

    transformed = []
    for batch in tqdm(batch_vectors(df, image_column, size, batch_size), desc="PCA transform"):
        transformed.append(pca.transform(batch).astype(np.float32))
    matrix = normalize_rows(np.concatenate(transformed, axis=0))
    np.save(output_dir / "features.npy", matrix)
    pd.DataFrame({
        "row_index": np.arange(len(df), dtype=np.int64),
        "image_id": df[image_id_column].astype(str).values,
        "image_path": df[image_column].astype(str).values,
    }).to_csv(output_dir / "index.csv", index=False)
    np.save(output_dir / "components.npy", pca.components_.astype(np.float32))
    np.save(output_dir / "mean.npy", pca.mean_.astype(np.float32))
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"method": "pca", "fit_partition": "development", "resize": size, "components": components, "images": len(df)}, f, indent=2)


if __name__ == "__main__":
    main()
