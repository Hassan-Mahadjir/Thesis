from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def fold_as_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("Int64")
    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_column: str, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        path = resolve_path(str(self.frame.loc[index, self.image_column]))
        return index, self.transform(Image.open(path).convert("RGB"))


def load_state_dict(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Optional: extract one fold only. Default extracts all configured OOF folds.")
    args = parser.parse_args()

    with resolve_path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    cfg = config["twin_aware_facenet"]
    columns = config["columns"]
    assignments = pd.read_csv(resolve_path(config["input"]["assignments_csv"]), low_memory=False)
    assignments[columns["image_id"]] = normalize_identifier(assignments[columns["image_id"]])
    assignments[columns["fold"]] = fold_as_int(assignments[columns["fold"]])

    folds = [args.fold] if args.fold else [int(x) for x in cfg["oof"]["folds"]]
    checkpoints_root = resolve_path(config["output"]["checkpoints_dir"]) / cfg["oof"]["checkpoints_subdir"]
    features_root = resolve_path(config["output"]["features_dir"]) / cfg["oof"]["features_name"]
    features_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for fold in folds:
        frame = assignments.loc[
            (assignments[columns["split"]] == "development")
            & (assignments[columns["fold"]] == fold)
        ].copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No DEVELOPMENT images found for fold {fold}.")

        checkpoint_path = checkpoints_root / f"fold_{fold}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing OOF checkpoint: {checkpoint_path}")

        model = InceptionResnetV1(pretrained=str(cfg["pretrained"]), classify=False)
        model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
        model = model.to(device).eval()

        dataset = ImageDataset(frame, cfg["image_column"], int(cfg["image_size"]))
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["evaluation_batch_size"]),
            shuffle=False,
            num_workers=int(cfg["num_workers"]),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(cfg["num_workers"]) > 0),
        )

        fold_dir = features_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        features = None

        with torch.no_grad():
            for indices, images in tqdm(loader, desc=f"Extracting held-out fold {fold}", unit="batch"):
                images = images.to(device, non_blocking=True)
                embeddings = F.normalize(model(images), dim=1)
                batch = embeddings.cpu().numpy().astype(np.float32)
                if features is None:
                    features = np.empty((len(frame), batch.shape[1]), dtype=np.float32)
                features[indices.numpy()] = batch

        if features is None:
            raise ValueError(f"No embeddings extracted for fold {fold}.")

        np.save(fold_dir / "features.npy", features)
        frame.to_csv(fold_dir / "index.csv", index=False)
        metadata = {
            "feature_name": cfg["oof"]["features_name"],
            "fold": fold,
            "partition": "development",
            "source_checkpoint": str(checkpoint_path),
            "matching_metric": "cosine",
            "embedding_dimension": int(features.shape[1]),
        }
        with (fold_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print(f"Saved fold {fold}: {fold_dir}")


if __name__ == "__main__":
    main()
