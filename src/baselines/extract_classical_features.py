from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return (features / norms).astype(np.float32)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def lbp_feature(image: np.ndarray, radius: int, points: int, method: str) -> np.ndarray:
    lbp = local_binary_pattern(image, P=points, R=radius, method=method)
    bins = points + 2 if method == "uniform" else int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, bins + 1), range=(0, bins))
    hist = hist.astype(np.float32)
    return hist / max(float(hist.sum()), 1.0)


def hog_feature(image: np.ndarray, orientations: int, pixels_per_cell: tuple[int, int], cells_per_block: tuple[int, int]) -> np.ndarray:
    return np.asarray(hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    ), dtype=np.float32)


def gabor_feature(image: np.ndarray, frequencies: list[float], thetas_degrees: list[float]) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    values = []
    for frequency in frequencies:
        for theta_degrees in thetas_degrees:
            real, imaginary = gabor(normalized, frequency=float(frequency), theta=float(np.deg2rad(theta_degrees)))
            magnitude = np.sqrt(real ** 2 + imaginary ** 2)
            values.extend([float(magnitude.mean()), float(magnitude.std())])
    return np.asarray(values, dtype=np.float32)


def extract_method(df: pd.DataFrame, image_id_column: str, image_path_column: str,
                   method_name: str, method_config: dict, output_dir: Path) -> None:
    features = []
    print(f"\nExtracting {method_name}...")

    for row in tqdm(df.itertuples(index=False), total=len(df), unit="image"):
        image = read_gray(resolve_path(getattr(row, image_path_column)))
        if method_name == "lbp":
            feature = lbp_feature(image, int(method_config["radius"]), int(method_config["points"]), str(method_config["method"]))
        elif method_name == "hog":
            feature = hog_feature(
                image,
                int(method_config["orientations"]),
                tuple(int(x) for x in method_config["pixels_per_cell"]),
                tuple(int(x) for x in method_config["cells_per_block"]),
            )
        elif method_name == "gabor":
            feature = gabor_feature(
                image,
                [float(x) for x in method_config["frequencies"]],
                [float(x) for x in method_config["thetas_degrees"]],
            )
        else:
            raise ValueError(method_name)
        features.append(feature)

    matrix = normalize_rows(np.stack(features, axis=0))
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "features.npy", matrix)
    pd.DataFrame({
        "row_index": np.arange(len(df), dtype=np.int64),
        "image_id": df[image_id_column].astype(str).values,
        "image_path": df[image_path_column].astype(str).values,
    }).to_csv(method_dir / "index.csv", index=False)
    with (method_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"method": method_name, "feature_dimension": int(matrix.shape[1]), "images": int(matrix.shape[0]), "image_column": image_path_column}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    df = pd.read_csv(resolve_path(config["input"]["assignments_csv"]), low_memory=False)
    output_dir = resolve_path(config["output"]["features_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    image_id_column = config["columns"]["image_id"]
    image_path_column = config["classical_features"]["image_column"]

    for method_name in ["lbp", "hog", "gabor"]:
        method_config = config["classical_features"][method_name]
        if bool(method_config.get("enabled", False)):
            extract_method(df, image_id_column, image_path_column, method_name, method_config, output_dir)


if __name__ == "__main__":
    main()
