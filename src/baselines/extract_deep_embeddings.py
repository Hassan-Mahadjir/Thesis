from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
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


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    return (vector / max(float(np.linalg.norm(vector)), 1e-12)).astype(np.float32)


def extract_deepface(df: pd.DataFrame, image_id_column: str, image_column: str,
                     model_config: dict, output_dir: Path, checkpoint_every: int) -> None:
    # DeepFace otherwise prefers TensorFlow when both frameworks are installed. On
    # Python 3.14 that currently selects pre-release TensorFlow/Keras packages whose
    # legacy tf-keras compatibility packages cannot be resolved together by pip.
    backend = str(model_config.get("backend", "pytorch")).lower()
    os.environ["DEEPFACE_BACKEND_ENGINE"] = backend

    # DeepFace's import-time warning contains Unicode symbols. Keep redirected
    # Windows output from failing when its default encoding is cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    from deepface import DeepFace

    model_name = str(model_config["model_name"])
    normalization = str(model_config.get("normalization", "base"))
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    progress_path = output_dir / "processed.npy"

    pd.DataFrame({
        "row_index": np.arange(len(df), dtype=np.int64),
        "image_id": df[image_id_column].astype(str).values,
        "image_path": df[image_column].astype(str).values,
    }).to_csv(output_dir / "index.csv", index=False)

    if feature_path.exists() and progress_path.exists():
        features = np.load(feature_path, mmap_mode="r+")
        processed = np.load(progress_path, mmap_mode="r+")
        print(f"Resuming {model_name}: {int(processed.sum()):,}/{len(processed):,}")
    else:
        first_path = resolve_path(df[image_column].iloc[0])
        first = DeepFace.represent(
            img_path=str(first_path), model_name=model_name,
            detector_backend="skip", enforce_detection=False,
            align=False, normalization=normalization,
        )
        embedding = l2_normalize(np.asarray(first[0]["embedding"], dtype=np.float32))
        features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(len(df), len(embedding)))
        processed = np.lib.format.open_memmap(progress_path, mode="w+", dtype=np.bool_, shape=(len(df),))
        processed[:] = False
        features[0] = embedding
        processed[0] = True
        features.flush(); processed.flush()

    remaining = np.where(~processed)[0]
    for counter, row_index in enumerate(tqdm(remaining, desc=model_name, unit="image"), start=1):
        path = resolve_path(df[image_column].iloc[int(row_index)])
        rep = DeepFace.represent(
            img_path=str(path), model_name=model_name,
            detector_backend="skip", enforce_detection=False,
            align=False, normalization=normalization,
        )
        features[row_index] = l2_normalize(np.asarray(rep[0]["embedding"], dtype=np.float32))
        processed[row_index] = True
        if counter % checkpoint_every == 0:
            features.flush(); processed.flush()

    features.flush(); processed.flush()
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "provider": "deepface", "backend": backend, "model_name": model_name,
            "images": int(len(df)), "embedding_dimension": int(features.shape[1]),
            "detector_backend": "skip", "align": False,
            "normalization": normalization,
        }, f, indent=2)


def preprocess_onnx(path: Path, input_size: int, color_order: str, normalization: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    if color_order.lower() == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    if normalization == "minus_one_to_one":
        image = (image - 127.5) / 127.5
    elif normalization == "zero_to_one":
        image = image / 255.0
    elif normalization != "none":
        raise ValueError(f"Unknown normalization: {normalization}")
    return np.transpose(image, (2, 0, 1))


def extract_onnx(df: pd.DataFrame, image_id_column: str, image_column: str,
                 model_config: dict, output_dir: Path, batch_size: int) -> None:
    import onnxruntime as ort

    model_path = resolve_path(model_config["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    input_size = int(model_config.get("input_size", 112))
    color_order = str(model_config.get("color_order", "rgb"))
    normalization = str(model_config.get("normalization", "minus_one_to_one"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "row_index": np.arange(len(df), dtype=np.int64),
        "image_id": df[image_id_column].astype(str).values,
        "image_path": df[image_column].astype(str).values,
    }).to_csv(output_dir / "index.csv", index=False)

    all_embeddings = []
    for start in tqdm(range(0, len(df), batch_size), desc=model_path.stem, unit="batch"):
        batch = df.iloc[start:start + batch_size]
        inputs = np.stack([
            preprocess_onnx(resolve_path(path), input_size, color_order, normalization)
            for path in batch[image_column]
        ])
        output = np.asarray(session.run(None, {input_name: inputs})[0], dtype=np.float32)
        output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
        all_embeddings.append(output.astype(np.float32))

    features = np.concatenate(all_embeddings, axis=0)
    np.save(output_dir / "features.npy", features)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "provider": "onnx", "model_path": str(model_path),
            "providers": providers, "images": int(len(df)),
            "embedding_dimension": int(features.shape[1]),
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    model_config = config["deep_models"].get(args.model)
    if model_config is None:
        raise ValueError(f"Unknown model: {args.model}")
    if not bool(model_config.get("enabled", False)):
        raise ValueError(f"{args.model} is disabled in config.")

    df = pd.read_csv(resolve_path(config["input"]["assignments_csv"]), low_memory=False)
    image_id_column = config["columns"]["image_id"]
    image_column = model_config["image_column"]
    output_dir = resolve_path(config["output"]["features_dir"]) / args.model
    provider = str(model_config["provider"]).lower()

    if provider == "deepface":
        extract_deepface(df, image_id_column, image_column, model_config, output_dir,
                         int(config["processing"].get("checkpoint_every", 100)))
    elif provider == "onnx":
        extract_onnx(df, image_id_column, image_column, model_config, output_dir,
                     int(config["processing"].get("onnx_batch_size", 128)))
    else:
        raise ValueError(f"Unsupported provider: {provider}")


if __name__ == "__main__":
    main()
