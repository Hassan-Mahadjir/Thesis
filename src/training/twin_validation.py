from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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


def protocol_image_ids(protocols_dir: Path, protocol_name: str, image_id_column: str) -> set[str]:
    path = protocols_dir / f"{protocol_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Protocol manifest not found: {path}")
    frame = pd.read_csv(path, dtype={image_id_column: "string"}, low_memory=False)
    if image_id_column not in frame.columns:
        raise ValueError(f"{path} does not contain column '{image_id_column}'.")
    return set(normalize_identifier(frame[image_id_column]).dropna().tolist())


def union_protocol_ids(protocols_dir: Path, protocol_names: list[str], image_id_column: str) -> set[str]:
    image_ids: set[str] = set()
    for protocol_name in protocol_names:
        image_ids.update(protocol_image_ids(protocols_dir, protocol_name, image_id_column))
    return image_ids


class EvaluationImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_column: str, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        path = resolve_path(str(self.frame.loc[index, self.image_column]))
        image = Image.open(path).convert("RGB")
        return index, self.transform(image)


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    image_column: str,
    image_size: int,
    image_id_column: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, dict[str, int]]:
    dataset = EvaluationImageDataset(frame, image_column, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    model.eval()
    all_features: np.ndarray | None = None
    for indices, images in loader:
        images = images.to(device, non_blocking=True)
        embeddings = F.normalize(model(images), dim=1)
        batch = embeddings.cpu().numpy().astype(np.float32)
        if all_features is None:
            all_features = np.empty((len(frame), batch.shape[1]), dtype=np.float32)
        all_features[indices.numpy()] = batch
    if all_features is None:
        raise ValueError("No images were available for embedding extraction.")
    image_to_row = {
        str(image_id): int(index)
        for index, image_id in enumerate(normalize_identifier(frame[image_id_column]).tolist())
    }
    return all_features, image_to_row


def _sample_pairs(pairs: list[tuple[str, str]], maximum: int, rng: random.Random) -> list[tuple[str, str]]:
    if len(pairs) <= maximum:
        return pairs
    chosen = rng.sample(pairs, maximum)
    chosen.sort()
    return chosen


def create_twin_verification_pairs(
    frame: pd.DataFrame,
    image_id_column: str,
    subject_column: str,
    mz_column: str,
    max_genuine_per_subject: int,
    max_cotwin_per_subject_pair: int,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    frame = frame.copy()
    frame[image_id_column] = normalize_identifier(frame[image_id_column])
    frame[subject_column] = normalize_identifier(frame[subject_column])
    frame[mz_column] = normalize_identifier(frame[mz_column])

    genuine: list[tuple[str, str]] = []
    for _, group in frame.groupby(subject_column):
        image_ids = sorted(group[image_id_column].dropna().tolist())
        if len(image_ids) >= 2:
            genuine.extend(_sample_pairs(
                list(itertools.combinations(image_ids, 2)),
                max_genuine_per_subject,
                rng,
            ))

    cotwin: list[tuple[str, str]] = []
    valid = frame.loc[
        frame[mz_column].notna()
        & (~frame[mz_column].isin(["", "<NA>", "nan", "None"]))
    ].copy()
    for _, mz_group in valid.groupby(mz_column):
        subjects = sorted(mz_group[subject_column].dropna().unique().tolist())
        for first_subject, second_subject in itertools.combinations(subjects, 2):
            first_images = sorted(mz_group.loc[mz_group[subject_column] == first_subject, image_id_column].dropna())
            second_images = sorted(mz_group.loc[mz_group[subject_column] == second_subject, image_id_column].dropna())
            candidates = [(a, b) for a in first_images for b in second_images]
            cotwin.extend(_sample_pairs(candidates, max_cotwin_per_subject_pair, rng))
    return genuine, cotwin


def pair_scores(
    pairs: list[tuple[str, str]],
    embeddings: np.ndarray,
    image_to_row: dict[str, int],
) -> np.ndarray:
    valid_pairs = [(a, b) for a, b in pairs if a in image_to_row and b in image_to_row]
    if not valid_pairs:
        return np.empty((0,), dtype=np.float32)
    first = np.asarray([image_to_row[a] for a, _ in valid_pairs], dtype=np.int64)
    second = np.asarray([image_to_row[b] for _, b in valid_pairs], dtype=np.int64)
    return np.sum(embeddings[first] * embeddings[second], axis=1).astype(np.float32)


def calculate_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def evaluate_validation_protocols(
    model: torch.nn.Module,
    assignments: pd.DataFrame,
    protocol_id_sets: dict[str, set[str]],
    validation_fold: int,
    protocol_weights: dict[str, float],
    image_column: str,
    image_id_column: str,
    subject_column: str,
    mz_column: str,
    split_column: str,
    fold_column: str,
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_genuine_per_subject: int,
    max_cotwin_per_subject_pair: int,
    seed: int,
) -> tuple[dict[str, float], float]:
    frame = assignments.copy()
    frame[image_id_column] = normalize_identifier(frame[image_id_column])
    frame[fold_column] = fold_as_int(frame[fold_column])
    validation = frame.loc[
        (frame[split_column] == "development")
        & (frame[fold_column] == int(validation_fold))
    ].copy()

    union_ids: set[str] = set().union(*protocol_id_sets.values())
    validation_union = validation.loc[validation[image_id_column].isin(union_ids)].copy()
    embeddings, image_to_row = extract_embeddings(
        model, validation_union, image_column, image_size, image_id_column,
        device, batch_size, num_workers,
    )

    eers: dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for protocol_name, weight in protocol_weights.items():
        protocol_frame = validation_union.loc[
            validation_union[image_id_column].isin(protocol_id_sets[protocol_name])
        ].copy()
        genuine, cotwin = create_twin_verification_pairs(
            protocol_frame,
            image_id_column,
            subject_column,
            mz_column,
            max_genuine_per_subject,
            max_cotwin_per_subject_pair,
            seed,
        )
        genuine_scores = pair_scores(genuine, embeddings, image_to_row)
        cotwin_scores = pair_scores(cotwin, embeddings, image_to_row)
        if len(genuine_scores) == 0 or len(cotwin_scores) == 0:
            raise ValueError(f"Protocol {protocol_name} has insufficient validation pairs.")
        labels = np.concatenate([
            np.ones(len(genuine_scores), dtype=np.int64),
            np.zeros(len(cotwin_scores), dtype=np.int64),
        ])
        scores = np.concatenate([genuine_scores, cotwin_scores])
        eer, _ = calculate_eer(labels, scores)
        eers[protocol_name] = eer
        weighted_sum += float(weight) * eer
        weight_sum += float(weight)
    return eers, float(weighted_sum / weight_sum)
