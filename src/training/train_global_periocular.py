from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.global_local.common import (
    build_global_transform,
    build_local_transform,
    calculate_eer,
    extract_global_local_embeddings,
    load_assignments_with_periocular,
    load_protocol_sets,
    normalize_identifier,
    resolve_path,
    score_pairs_with_head,
    union_protocol_ids,
    create_twin_verification_pairs,
)
from src.models.global_periocular import GlobalPeriocularVerifier, VALID_VARIANTS


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class BalancedTwinPairDataset(Dataset):
    """
    On-the-fly pair sampler.

    50% positive same-identity pairs by default.
    Negative pairs are mostly monozygotic co-twins, with a small fraction
    of unrelated negatives to preserve general separation.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        columns: dict,
        global_image_column: str,
        local_image_column: str,
        global_image_size: int,
        local_height: int,
        local_width: int,
        pairs_per_epoch: int,
        positive_probability: float,
        cotwin_negative_probability: float,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.columns = columns
        self.global_image_column = global_image_column
        self.local_image_column = local_image_column
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.positive_probability = float(positive_probability)
        self.cotwin_negative_probability = float(cotwin_negative_probability)

        self.subject_column = columns["subject_id"]
        self.family_column = columns["family_id"]
        self.mz_column = columns["monozygotic_group_id"]

        self.global_transform = build_global_transform(
            global_image_size,
            training=True,
        )
        self.local_transform = build_local_transform(
            local_height,
            local_width,
            training=True,
        )

        self.subject_rows = {
            str(subject): group.index.to_list()
            for subject, group in self.frame.groupby(self.subject_column)
        }

        self.subject_family = {
            str(subject): str(group[self.family_column].iloc[0])
            for subject, group in self.frame.groupby(self.subject_column)
        }

        self.subject_mz = {
            str(subject): (
                None
                if pd.isna(group[self.mz_column].iloc[0])
                else str(group[self.mz_column].iloc[0])
            )
            for subject, group in self.frame.groupby(self.subject_column)
        }

        self.mz_to_subjects: dict[str, list[str]] = {}

        for subject, mz_group in self.subject_mz.items():
            if mz_group in (None, "", "<NA>", "nan", "None"):
                continue
            self.mz_to_subjects.setdefault(mz_group, []).append(subject)

        self.positive_subjects = [
            subject
            for subject, rows in self.subject_rows.items()
            if len(rows) >= 2
        ]

        self.twin_anchor_subjects = []

        for subject in self.subject_rows:
            mz_group = self.subject_mz[subject]
            if mz_group in (None, "", "<NA>", "nan", "None"):
                continue

            twins = [
                other
                for other in self.mz_to_subjects.get(mz_group, [])
                if other != subject and other in self.subject_rows
            ]

            if twins:
                self.twin_anchor_subjects.append(subject)

        self.all_subjects = list(self.subject_rows.keys())

        if not self.positive_subjects:
            raise ValueError("No subjects have at least two training images.")

        if not self.twin_anchor_subjects:
            raise ValueError("No MZ co-twin subjects found in the training frame.")

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _load(self, row_index: int):
        global_path = resolve_path(
            str(self.frame.loc[row_index, self.global_image_column])
        )
        local_path = resolve_path(
            str(self.frame.loc[row_index, self.local_image_column])
        )

        global_image = Image.open(global_path).convert("RGB")
        local_image = Image.open(local_path).convert("RGB")

        return (
            self.global_transform(global_image),
            self.local_transform(local_image),
        )

    def _positive_pair(self):
        subject = random.choice(self.positive_subjects)
        first_row, second_row = random.sample(
            self.subject_rows[subject],
            2,
        )
        return first_row, second_row, 1.0

    def _cotwin_negative_pair(self):
        subject = random.choice(self.twin_anchor_subjects)
        mz_group = self.subject_mz[subject]

        twin_subjects = [
            other
            for other in self.mz_to_subjects[mz_group]
            if other != subject and other in self.subject_rows
        ]

        twin_subject = random.choice(twin_subjects)

        first_row = random.choice(self.subject_rows[subject])
        second_row = random.choice(self.subject_rows[twin_subject])

        return first_row, second_row, 0.0

    def _unrelated_negative_pair(self):
        first_subject = random.choice(self.all_subjects)
        first_family = self.subject_family[first_subject]

        candidates = [
            subject
            for subject in self.all_subjects
            if self.subject_family[subject] != first_family
        ]

        second_subject = random.choice(candidates)

        first_row = random.choice(self.subject_rows[first_subject])
        second_row = random.choice(self.subject_rows[second_subject])

        return first_row, second_row, 0.0

    def __getitem__(self, index: int):
        if random.random() < self.positive_probability:
            first_row, second_row, label = self._positive_pair()
        else:
            if random.random() < self.cotwin_negative_probability:
                first_row, second_row, label = self._cotwin_negative_pair()
            else:
                first_row, second_row, label = self._unrelated_negative_pair()

        global_a, local_a = self._load(first_row)
        global_b, local_b = self._load(second_row)

        return (
            global_a,
            global_b,
            local_a,
            local_b,
            torch.tensor(label, dtype=torch.float32),
        )


def build_model(
    cfg: dict,
    variant: str,
) -> GlobalPeriocularVerifier:
    return GlobalPeriocularVerifier(
        variant=variant,
        global_pretrained=str(cfg["global_pretrained"]),
        local_embedding_dim=int(cfg["local_embedding_dim"]),
        local_pretrained_imagenet=bool(cfg["local_pretrained_imagenet"]),
        hidden_1=int(cfg["pair_head_hidden_1"]),
        hidden_2=int(cfg["pair_head_hidden_2"]),
        dropout=float(cfg["dropout"]),
        freeze_global=bool(cfg["freeze_global"]),
    )


def build_train_frame(
    assignments: pd.DataFrame,
    protocol_sets: dict[str, set[str]],
    config: dict,
    mode: str,
    oof_fold: int | None,
) -> pd.DataFrame:
    columns = config["columns"]
    cfg = config["global_periocular"]

    training_ids = union_protocol_ids(protocol_sets)

    frame = assignments.loc[
        (assignments[columns["split"]] == "development")
        & (assignments[columns["image_id"]].isin(training_ids))
    ].copy()

    if mode == "tuning":
        frame = frame.loc[
            frame[columns["fold"]] != int(cfg["validation_fold"])
        ].copy()

    elif mode == "oof":
        frame = frame.loc[
            frame[columns["fold"]] != int(oof_fold)
        ].copy()

    frame = frame.loc[
        frame[cfg["global_image_column"]].notna()
        & frame[cfg["local_image_column"]].notna()
    ].copy()

    if frame.empty:
        raise ValueError("Training frame is empty after filtering.")

    return frame


def evaluate_validation(
    model: GlobalPeriocularVerifier,
    assignments: pd.DataFrame,
    validation_protocol_sets: dict[str, set[str]],
    config: dict,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    columns = config["columns"]
    cfg = config["global_periocular"]

    validation = assignments.loc[
        (assignments[columns["split"]] == "development")
        & (
            assignments[columns["fold"]]
            == int(cfg["validation_fold"])
        )
    ].copy()

    validation_ids = union_protocol_ids(validation_protocol_sets)

    validation = validation.loc[
        validation[columns["image_id"]].isin(validation_ids)
        & validation[cfg["global_image_column"]].notna()
        & validation[cfg["local_image_column"]].notna()
    ].copy()

    global_features, local_features, image_to_row = (
        extract_global_local_embeddings(
            model=model,
            frame=validation,
            image_id_column=columns["image_id"],
            global_image_column=cfg["global_image_column"],
            local_image_column=cfg["local_image_column"],
            global_image_size=int(cfg["global_image_size"]),
            local_height=int(cfg["local_height"]),
            local_width=int(cfg["local_width"]),
            device=device,
            batch_size=int(cfg["evaluation_batch_size"]),
            num_workers=int(cfg["num_workers"]),
        )
    )

    eers = {}
    weighted_sum = 0.0
    weight_sum = 0.0

    for protocol_name, weight in cfg["validation_protocols"].items():
        protocol_frame = validation.loc[
            validation[columns["image_id"]].isin(
                validation_protocol_sets[protocol_name]
            )
        ].copy()

        genuine, cotwin = create_twin_verification_pairs(
            frame=protocol_frame,
            image_id_column=columns["image_id"],
            subject_column=columns["subject_id"],
            mz_column=columns["monozygotic_group_id"],
            max_genuine_per_subject=int(
                cfg["max_genuine_pairs_per_subject"]
            ),
            max_cotwin_per_subject_pair=int(
                cfg["max_cotwin_pairs_per_subject_pair"]
            ),
            seed=int(cfg["seed"]),
        )

        genuine_scores, _ = score_pairs_with_head(
            model,
            genuine,
            global_features,
            local_features,
            image_to_row,
            device,
            int(cfg["pair_score_batch_size"]),
        )

        cotwin_scores, _ = score_pairs_with_head(
            model,
            cotwin,
            global_features,
            local_features,
            image_to_row,
            device,
            int(cfg["pair_score_batch_size"]),
        )

        if len(genuine_scores) == 0 or len(cotwin_scores) == 0:
            raise ValueError(
                f"Insufficient validation pairs for {protocol_name}."
            )

        labels = np.concatenate(
            [
                np.ones(len(genuine_scores), dtype=np.int64),
                np.zeros(len(cotwin_scores), dtype=np.int64),
            ]
        )
        scores = np.concatenate([genuine_scores, cotwin_scores])

        eer, _ = calculate_eer(labels, scores)

        eers[protocol_name] = float(eer)
        weighted_sum += float(weight) * float(eer)
        weight_sum += float(weight)

    return eers, float(weighted_sum / weight_sum)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(VALID_VARIANTS),
        help=(
            "Ablation variant. Use global_pair and local_pair for diagnosis; "
            "global_local_pair reproduces the original architecture."
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--final",
        action="store_true",
        help="Train the frozen configuration on all DEVELOPMENT.",
    )

    mode_group.add_argument(
        "--oof-fold",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Train a fixed-epoch OOF model excluding this fold.",
    )

    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    cfg = config["global_periocular"]
    columns = config["columns"]

    seed = int(cfg["seed"])
    seed_everything(seed)

    if args.final:
        mode = "final"
    elif args.oof_fold is not None:
        mode = "oof"
    else:
        mode = "tuning"

    assignments = load_assignments_with_periocular(
        assignments_csv=resolve_path(config["input"]["assignments_csv"]),
        manifest_csv=resolve_path(
            config["input"]["periocular_manifest_csv"]
        ),
        columns=columns,
        local_image_column=cfg["local_image_column"],
    )

    protocols_dir = resolve_path(config["input"]["protocols_dir"])

    training_protocol_sets = load_protocol_sets(
        protocols_dir,
        list(cfg["training_protocols"]),
        columns["image_id"],
    )

    validation_protocol_sets = load_protocol_sets(
        protocols_dir,
        list(cfg["validation_protocols"].keys()),
        columns["image_id"],
    )

    train_frame = build_train_frame(
        assignments=assignments,
        protocol_sets=training_protocol_sets,
        config=config,
        mode=mode,
        oof_fold=args.oof_fold,
    )

    print(f"Mode: {mode}")
    print(f"Training images: {len(train_frame):,}")
    print(
        f"Training subjects: "
        f"{train_frame[columns['subject_id']].nunique():,}"
    )
    print(
        f"Training families: "
        f"{train_frame[columns['family_id']].nunique():,}"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    variant = str(args.variant)
    print(f"Variant: {variant}")

    model = build_model(
        cfg,
        variant=variant,
    ).to(device)

    dataset = BalancedTwinPairDataset(
        frame=train_frame,
        columns=columns,
        global_image_column=cfg["global_image_column"],
        local_image_column=cfg["local_image_column"],
        global_image_size=int(cfg["global_image_size"]),
        local_height=int(cfg["local_height"]),
        local_width=int(cfg["local_width"]),
        pairs_per_epoch=int(cfg["train_pairs_per_epoch"]),
        positive_probability=float(cfg["positive_probability"]),
        cotwin_negative_probability=float(
            cfg["cotwin_negative_probability"]
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(cfg["num_workers"]) > 0),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )

    parameter_groups = []

    if model.uses_local:
        local_parameters = (
            list(model.local_backbone.parameters())
            + list(model.local_projection.parameters())
        )
        parameter_groups.append(
            {
                "params": [
                    parameter
                    for parameter in local_parameters
                    if parameter.requires_grad
                ],
                "lr": float(cfg["local_learning_rate"]),
            }
        )

    parameter_groups.append(
        {
            "params": [
                parameter
                for parameter in model.pair_head.parameters()
                if parameter.requires_grad
            ],
            "lr": float(cfg["head_learning_rate"]),
        }
    )

    if model.uses_global and not bool(cfg["freeze_global"]):
        parameter_groups.append(
            {
                "params": [
                    parameter
                    for parameter in model.global_backbone.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(cfg["local_learning_rate"]) * 0.1,
            }
        )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(cfg["weight_decay"]),
    )

    criterion = nn.BCEWithLogitsLoss()

    if mode == "tuning":
        epochs = int(cfg["epochs"])
    elif mode == "oof":
        epochs = int(cfg["oof_fixed_epochs"])
    else:
        epochs = int(cfg["final_epochs"])

    checkpoint_root = (
        resolve_path(config["output"]["checkpoints_dir"])
        / "ablation"
        / variant
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if mode == "tuning":
        checkpoint_path = checkpoint_root / "best.pt"
    elif mode == "final":
        checkpoint_path = checkpoint_root / "final.pt"
    else:
        oof_dir = checkpoint_root / cfg["oof"]["checkpoints_subdir"]
        oof_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = oof_dir / f"fold_{args.oof_fold}.pt"

    report_dir = (
        resolve_path(config["output"]["training_reports_dir"])
        / "ablation"
        / variant
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_objective = float("inf")
    best_epoch = None
    without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()

        losses = []

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
        )

        for (
            global_a,
            global_b,
            local_a,
            local_b,
            labels,
        ) in progress:
            global_a = global_a.to(device, non_blocking=True)
            global_b = global_b.to(device, non_blocking=True)
            local_a = local_a.to(device, non_blocking=True)
            local_b = local_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                global_a,
                global_b,
                local_a,
                local_b,
            )

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=5.0,
            )

            optimizer.step()

            losses.append(float(loss.item()))

            progress.set_postfix(
                loss=f"{np.mean(losses[-50:]):.4f}"
            )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
        }

        if mode == "tuning":
            eers, objective = evaluate_validation(
                model=model,
                assignments=assignments,
                validation_protocol_sets=validation_protocol_sets,
                config=config,
                device=device,
            )

            for name, eer in eers.items():
                row[f"eer::{name}"] = eer

            row["validation_objective"] = objective

            print(
                "\nValidation EERs: "
                + " | ".join(
                    f"{name}={eer * 100:.2f}%"
                    for name, eer in eers.items()
                )
            )
            print(
                f"Weighted EER objective: "
                f"{objective * 100:.2f}%"
            )

            improved = (
                objective
                < best_objective - float(cfg["objective_min_delta"])
            )

            if improved:
                best_objective = objective
                best_epoch = epoch
                without_improvement = 0

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "variant": variant,
                        "epoch": epoch,
                        "validation_objective": objective,
                        "validation_eers": eers,
                    },
                    checkpoint_path,
                )

                print(f"Saved new best: {checkpoint_path}")

            else:
                without_improvement += 1

        else:
            # OOF and final modes train a fixed number of epochs.
            # Held-out OOF folds are NOT used for checkpoint selection.
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "epoch": epoch,
                    "mode": mode,
                    "oof_fold": args.oof_fold,
                },
                checkpoint_path,
            )

        history.append(row)

        history_name = {
            "tuning": "tuning_history.csv",
            "final": "final_training_history.csv",
            "oof": f"oof_fold_{args.oof_fold}_history.csv",
        }[mode]

        pd.DataFrame(history).to_csv(
            report_dir / history_name,
            index=False,
        )

        if (
            mode == "tuning"
            and without_improvement >= int(cfg["early_stopping_patience"])
        ):
            print("Early stopping.")
            break

    summary = {
        "variant": variant,
        "mode": mode,
        "epochs_completed": len(history),
        "best_epoch": best_epoch if mode == "tuning" else None,
        "best_validation_objective": (
            best_objective if mode == "tuning" else None
        ),
        "checkpoint": str(checkpoint_path),
    }

    summary_name = {
        "tuning": "tuning_summary.json",
        "final": "final_training_summary.json",
        "oof": f"oof_fold_{args.oof_fold}_summary.json",
    }[mode]

    with (report_dir / summary_name).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nTraining complete.")
    print(f"Checkpoint: {checkpoint_path}")

    if mode == "tuning":
        print(f"Best epoch: {best_epoch}")
        print(
            f"Best weighted validation EER: "
            f"{best_objective * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
