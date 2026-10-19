from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_identifier(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def fold_as_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("Int64")

    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def protocol_image_ids(
    protocols_dir: Path,
    protocol_name: str,
    image_id_column: str,
) -> set[str]:
    path = protocols_dir / f"{protocol_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Protocol manifest not found: {path}")

    frame = pd.read_csv(
        path,
        dtype={image_id_column: "string"},
        low_memory=False,
    )

    if image_id_column not in frame.columns:
        raise ValueError(
            f"{path} does not contain column '{image_id_column}'."
        )

    return set(
        normalize_identifier(frame[image_id_column]).dropna().tolist()
    )


def load_protocol_sets(
    protocols_dir: Path,
    protocol_names: list[str],
    image_id_column: str,
) -> dict[str, set[str]]:
    return {
        name: protocol_image_ids(protocols_dir, name, image_id_column)
        for name in protocol_names
    }


def union_protocol_ids(protocol_sets: dict[str, set[str]]) -> set[str]:
    result: set[str] = set()
    for ids in protocol_sets.values():
        result.update(ids)
    return result


def load_assignments_with_periocular(
    assignments_csv: Path,
    manifest_csv: Path,
    columns: dict,
    local_image_column: str,
) -> pd.DataFrame:
    assignments = pd.read_csv(assignments_csv, low_memory=False)

    image_id_column = columns["image_id"]
    assignments[image_id_column] = normalize_identifier(
        assignments[image_id_column]
    )
    assignments[columns["subject_id"]] = normalize_identifier(
        assignments[columns["subject_id"]]
    )
    assignments[columns["family_id"]] = normalize_identifier(
        assignments[columns["family_id"]]
    )
    assignments[columns["monozygotic_group_id"]] = normalize_identifier(
        assignments[columns["monozygotic_group_id"]]
    )
    assignments[columns["fold"]] = fold_as_int(assignments[columns["fold"]])

    manifest = pd.read_csv(
        manifest_csv,
        dtype={image_id_column: "string"},
        low_memory=False,
    )
    manifest[image_id_column] = normalize_identifier(manifest[image_id_column])

    required_manifest = {image_id_column, local_image_column}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Periocular manifest is missing columns: {sorted(missing)}"
        )

    if "status" in manifest.columns:
        manifest = manifest.loc[manifest["status"] == "ok"].copy()

    manifest = manifest[[image_id_column, local_image_column]].drop_duplicates(
        subset=[image_id_column],
        keep="last",
    )

    merged = assignments.merge(
        manifest,
        on=image_id_column,
        how="left",
        validate="one_to_one",
    )

    return merged


def _sample_pairs(
    pairs: list[tuple[str, str]],
    maximum: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
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
        if len(image_ids) < 2:
            continue

        candidates = list(itertools.combinations(image_ids, 2))
        genuine.extend(
            _sample_pairs(
                candidates,
                max_genuine_per_subject,
                rng,
            )
        )

    cotwin: list[tuple[str, str]] = []

    valid = frame.loc[
        frame[mz_column].notna()
        & (~frame[mz_column].isin(["", "<NA>", "nan", "None"]))
    ].copy()

    for _, mz_group in valid.groupby(mz_column):
        subjects = sorted(
            mz_group[subject_column].dropna().unique().tolist()
        )

        for first_subject, second_subject in itertools.combinations(
            subjects, 2
        ):
            first_images = sorted(
                mz_group.loc[
                    mz_group[subject_column] == first_subject,
                    image_id_column,
                ].dropna().tolist()
            )

            second_images = sorted(
                mz_group.loc[
                    mz_group[subject_column] == second_subject,
                    image_id_column,
                ].dropna().tolist()
            )

            candidates = [
                (a, b)
                for a in first_images
                for b in second_images
            ]

            cotwin.extend(
                _sample_pairs(
                    candidates,
                    max_cotwin_per_subject_pair,
                    rng,
                )
            )

    return genuine, cotwin


def calculate_eer(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))

    eer = float((fpr[index] + fnr[index]) / 2.0)
    threshold = float(thresholds[index])

    return eer, threshold


def calculate_threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    total = max(tp + tn + fp + fn, 1)

    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (
        2.0 * precision * recall / max(precision + recall, 1e-12)
    )
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
    }


def calculate_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    return float(roc_auc_score(labels, scores))


def build_global_transform(
    image_size: int,
    training: bool,
):
    operations = [transforms.Resize((image_size, image_size))]

    if training:
        operations.extend(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.08,
                            contrast=0.08,
                            saturation=0.04,
                        )
                    ],
                    p=0.25,
                ),
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.01, 0.01),
                    scale=(0.99, 1.01),
                ),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    return transforms.Compose(operations)


def build_local_transform(
    height: int,
    width: int,
    training: bool,
):
    operations = [transforms.Resize((height, width))]

    if training:
        # No horizontal flip: left/right asymmetry may be twin-discriminative.
        operations.extend(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.08,
                            contrast=0.10,
                            saturation=0.03,
                        )
                    ],
                    p=0.30,
                ),
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.015, 0.015),
                    scale=(0.98, 1.02),
                ),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return transforms.Compose(operations)


class EncodingDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        global_image_column: str,
        local_image_column: str,
        global_image_size: int,
        local_height: int,
        local_width: int,
    ):
        self.frame = frame.reset_index(drop=True)
        self.global_image_column = global_image_column
        self.local_image_column = local_image_column

        self.global_transform = build_global_transform(
            global_image_size,
            training=False,
        )
        self.local_transform = build_local_transform(
            local_height,
            local_width,
            training=False,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        global_path = resolve_path(
            str(self.frame.loc[index, self.global_image_column])
        )
        local_path = resolve_path(
            str(self.frame.loc[index, self.local_image_column])
        )

        global_image = Image.open(global_path).convert("RGB")
        local_image = Image.open(local_path).convert("RGB")

        return (
            index,
            self.global_transform(global_image),
            self.local_transform(local_image),
        )


@torch.no_grad()
def extract_global_local_embeddings(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    image_id_column: str,
    global_image_column: str,
    local_image_column: str,
    global_image_size: int,
    local_height: int,
    local_width: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    dataset = EncodingDataset(
        frame=frame,
        global_image_column=global_image_column,
        local_image_column=local_image_column,
        global_image_size=global_image_size,
        local_height=local_height,
        local_width=local_width,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model.eval()

    global_features = None
    local_features = None

    for indices, global_images, local_images in loader:
        global_images = global_images.to(device, non_blocking=True)
        local_images = local_images.to(device, non_blocking=True)

        g, l = model.encode(
            global_images,
            local_images,
        )

        g_np = g.cpu().numpy().astype(np.float32)
        l_np = l.cpu().numpy().astype(np.float32)

        if global_features is None:
            global_features = np.empty(
                (len(frame), g_np.shape[1]),
                dtype=np.float32,
            )
            local_features = np.empty(
                (len(frame), l_np.shape[1]),
                dtype=np.float32,
            )

        global_features[indices.numpy()] = g_np
        local_features[indices.numpy()] = l_np

    if global_features is None or local_features is None:
        raise ValueError("No images were available for embedding extraction.")

    image_to_row = {
        str(image_id): int(index)
        for index, image_id in enumerate(
            normalize_identifier(frame[image_id_column]).tolist()
        )
    }

    return global_features, local_features, image_to_row


@torch.no_grad()
def score_pairs_with_head(
    model: torch.nn.Module,
    pairs: list[tuple[str, str]],
    global_features: np.ndarray,
    local_features: np.ndarray,
    image_to_row: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    valid_pairs = [
        (a, b)
        for a, b in pairs
        if a in image_to_row and b in image_to_row
    ]

    if not valid_pairs:
        return np.empty((0,), dtype=np.float32), []

    scores = []

    for start in range(0, len(valid_pairs), batch_size):
        batch_pairs = valid_pairs[start:start + batch_size]

        first_rows = np.asarray(
            [image_to_row[a] for a, _ in batch_pairs],
            dtype=np.int64,
        )
        second_rows = np.asarray(
            [image_to_row[b] for _, b in batch_pairs],
            dtype=np.int64,
        )

        g1 = torch.from_numpy(global_features[first_rows]).to(device)
        g2 = torch.from_numpy(global_features[second_rows]).to(device)
        l1 = torch.from_numpy(local_features[first_rows]).to(device)
        l2 = torch.from_numpy(local_features[second_rows]).to(device)

        logits = model.score_embeddings(g1, g2, l1, l2)
        batch_scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        scores.append(batch_scores)

    return np.concatenate(scores), valid_pairs
