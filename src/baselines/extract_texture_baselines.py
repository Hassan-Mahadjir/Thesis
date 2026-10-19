from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.signal import convolve2d
from skimage.feature import local_binary_pattern
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def preprocess_image(path: Path, width: int, height: int, use_clahe: bool) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if use_clahe:
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    return image


def normalized_histogram(values: np.ndarray, bins: int,
                         root_histogram: bool) -> np.ndarray:
    histogram = np.bincount(values.ravel().astype(np.int64), minlength=bins)[:bins]
    histogram = histogram.astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    if root_histogram:
        histogram = np.sqrt(histogram)
    return histogram


def grid_regions(values: np.ndarray, rows: int, columns: int):
    for row_group in np.array_split(values, rows, axis=0):
        yield from np.array_split(row_group, columns, axis=1)


def lbp_feature(image: np.ndarray, config: dict) -> np.ndarray:
    rows = int(config["grid"]["rows"])
    columns = int(config["grid"]["columns"])
    root_histogram = bool(config.get("root_histogram", False))
    descriptors = []
    for scale in config["scales"]:
        points = int(scale["points"])
        radius = int(scale["radius"])
        method = str(scale["method"])
        codes = local_binary_pattern(image, P=points, R=radius, method=method)
        bins = points + 2 if method == "uniform" else 2 ** points
        descriptors.extend(
            normalized_histogram(region, bins, root_histogram)
            for region in grid_regions(codes, rows, columns)
        )
    return np.concatenate(descriptors).astype(np.float32)


def lpq_codes(image: np.ndarray, window_size: int) -> np.ndarray:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("LPQ window_size must be an odd integer of at least 3")
    coordinates = np.arange(-(window_size // 2), window_size // 2 + 1)
    frequency = 1.0 / window_size
    constant = np.ones(window_size, dtype=np.complex64)
    wave = np.exp(-2j * np.pi * frequency * coordinates).astype(np.complex64)
    filters = (
        np.outer(constant, wave),
        np.outer(wave, constant),
        np.outer(wave, wave),
        np.outer(wave, np.conjugate(wave)),
    )
    normalized = image.astype(np.float32) / 255.0
    responses = [
        convolve2d(normalized, kernel, mode="same", boundary="symm")
        for kernel in filters
    ]
    channels = [part for response in responses for part in (response.real, response.imag)]
    codes = np.zeros(image.shape, dtype=np.uint8)
    for bit, channel in enumerate(channels):
        codes |= ((channel >= 0).astype(np.uint8) << bit)
    return codes


def lpq_feature(image: np.ndarray, config: dict) -> np.ndarray:
    rows = int(config["grid"]["rows"])
    columns = int(config["grid"]["columns"])
    root_histogram = bool(config.get("root_histogram", False))
    codes = lpq_codes(image, int(config["window_size"]))
    return np.concatenate([
        normalized_histogram(region, 256, root_histogram)
        for region in grid_regions(codes, rows, columns)
    ]).astype(np.float32)


def extract_variant(df: pd.DataFrame, image_id_column: str, image_column: str,
                    name: str, config: dict, output_root: Path) -> None:
    resize = config["resize"]
    width, height = int(resize["width"]), int(resize["height"])
    features = []
    for value in tqdm(df[image_column].astype(str), desc=name, unit="image"):
        image = preprocess_image(
            resolve_path(value), width, height, bool(config.get("clahe", False)),
        )
        feature = lpq_feature(image, config) if name == "lpq_partitioned" else lbp_feature(image, config)
        features.append(feature)
    matrix = np.stack(features).astype(np.float32)

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", matrix)
    pd.DataFrame({
        "row_index": np.arange(len(df), dtype=np.int64),
        "image_id": df[image_id_column].astype(str).values,
        "image_path": df[image_column].astype(str).values,
    }).to_csv(output_dir / "index.csv", index=False)
    metadata = {
        "method": name,
        "images": int(len(df)),
        "feature_dimension": int(matrix.shape[1]),
        "image_column": image_column,
        "resize": {"width": width, "height": height},
        "grid": config["grid"],
        "clahe": bool(config.get("clahe", False)),
        "root_histogram": bool(config.get("root_histogram", False)),
        "matching_metric": str(config.get("matching_metric", "manhattan")).lower(),
    }
    if name == "lpq_partitioned":
        metadata["window_size"] = int(config["window_size"])
    else:
        metadata["scales"] = config["scales"]
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    section = config["texture_baselines"]
    df = pd.read_csv(resolve_path(config["input"]["assignments_csv"]), low_memory=False)
    image_id_column = config["columns"]["image_id"]
    image_column = section["image_column"]
    output_root = resolve_path(config["output"]["features_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("lbp_partitioned", "lbp_multiscale", "lpq_partitioned"):
        variant = section[name]
        if bool(variant.get("enabled", False)):
            extract_variant(
                df, image_id_column, image_column, name, variant, output_root,
            )


if __name__ == "__main__":
    main()
