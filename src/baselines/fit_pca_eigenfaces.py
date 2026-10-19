from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def preprocess_image(path: Path, width: int, height: int,
                     equalization: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if equalization == "histogram":
        image = cv2.equalizeHist(image)
    elif equalization == "clahe":
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    elif equalization != "none":
        raise ValueError(f"Unknown equalization method: {equalization}")

    vector = image.astype(np.float32).reshape(-1)
    standard_deviation = max(float(vector.std()), 1e-6)
    return (vector - float(vector.mean())) / standard_deviation


def build_matrix(frame: pd.DataFrame, image_column: str, width: int,
                 height: int, equalization: str, description: str) -> np.ndarray:
    matrix = np.empty((len(frame), width * height), dtype=np.float32)
    paths = frame[image_column].astype(str).tolist()
    for row_index, value in enumerate(tqdm(paths, desc=description, unit="image")):
        matrix[row_index] = preprocess_image(
            resolve_path(value), width, height, equalization,
        )
    return matrix


def fit_variant(df: pd.DataFrame, development_mask: np.ndarray, image_id_column: str,
                image_column: str, name: str, variant: dict, output_root: Path,
                random_state: int) -> None:
    resize = variant["resize"]
    width, height = int(resize["width"]), int(resize["height"])
    equalization = str(variant["equalization"]).lower()
    requested_components = int(variant["n_components"])
    standardize = bool(variant.get("component_standardization", False))

    development_frame = df.loc[development_mask]
    development = build_matrix(
        development_frame, image_column, width, height, equalization,
        f"{name} DEVELOPMENT preprocessing",
    )
    components = min(requested_components, development.shape[0] - 1, development.shape[1])
    if components < 1:
        raise ValueError("PCA needs at least two DEVELOPMENT images")
    if components != requested_components:
        print(f"{name}: reduced components from {requested_components} to {components}")

    print(f"Fitting {name} on DEVELOPMENT only: {len(development):,} images")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=random_state,
    )
    pca.fit(development)
    del development
    matrix = build_matrix(
        df, image_column, width, height, equalization,
        f"{name} transform preprocessing",
    )
    features = pca.transform(matrix).astype(np.float32)
    del matrix
    component_scales = np.ones(components, dtype=np.float32)
    if standardize:
        component_scales = np.sqrt(
            np.maximum(pca.explained_variance_.astype(np.float32), 1e-12),
        )
        features /= component_scales

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "components.npy", pca.components_.astype(np.float32))
    np.save(output_dir / "mean_face.npy", pca.mean_.astype(np.float32))
    np.save(output_dir / "explained_variance.npy", pca.explained_variance_.astype(np.float32))
    if standardize:
        np.save(output_dir / "component_scales.npy", component_scales)
    pd.DataFrame({
        "row_index": np.arange(len(df), dtype=np.int64),
        "image_id": df[image_id_column].astype(str).values,
        "image_path": df[image_column].astype(str).values,
    }).to_csv(output_dir / "index.csv", index=False)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({
            "method": name,
            "fit_partition": "development",
            "images": int(len(df)),
            "development_images": int(development_mask.sum()),
            "image_column": image_column,
            "resize": {"width": width, "height": height},
            "equalization": equalization,
            "per_image_standardization": True,
            "requested_components": requested_components,
            "components": components,
            "component_standardization": standardize,
            "matching_metric": str(variant.get("matching_metric", "manhattan")).lower(),
        }, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    section = config["pca_eigenfaces"]
    df = pd.read_csv(resolve_path(config["input"]["assignments_csv"]), low_memory=False)
    image_id_column = config["columns"]["image_id"]
    split_column = config["columns"]["split"]
    image_column = section["image_column"]
    development_mask = df[split_column].astype(str).eq("development").to_numpy()
    output_root = resolve_path(config["output"]["features_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    for name, variant in section["variants"].items():
        if bool(variant.get("enabled", False)):
            fit_variant(
                df, development_mask, image_id_column, image_column, name,
                variant, output_root, int(section.get("random_state", 42)),
            )


if __name__ == "__main__":
    main()
