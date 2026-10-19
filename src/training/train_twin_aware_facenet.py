from __future__ import annotations

import argparse
import copy
import json
import random
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

from src.training.twin_validation import (
    evaluate_validation_protocols,
    fold_as_int,
    normalize_identifier,
    protocol_image_ids,
    resolve_path,
    union_protocol_ids,
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TwinTripletDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_column: str,
        subject_column: str,
        family_column: str,
        mz_column: str,
        transform,
        samples_per_epoch: int | None = None,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.image_column = image_column
        self.subject_column = subject_column
        self.family_column = family_column
        self.mz_column = mz_column
        self.transform = transform

        self.subject_rows = {
            str(subject): group.index.to_list()
            for subject, group in self.frame.groupby(subject_column)
        }
        self.subject_family = {
            str(subject): str(group[family_column].iloc[0])
            for subject, group in self.frame.groupby(subject_column)
        }
        self.subject_mz = {
            str(subject): (None if pd.isna(group[mz_column].iloc[0]) else str(group[mz_column].iloc[0]))
            for subject, group in self.frame.groupby(subject_column)
        }

        self.mz_to_subjects: dict[str, list[str]] = {}
        for subject, mz in self.subject_mz.items():
            if mz is None or mz in {"", "<NA>", "nan", "None"}:
                continue
            self.mz_to_subjects.setdefault(mz, []).append(subject)

        self.eligible_subjects = []
        for subject, rows in self.subject_rows.items():
            if len(rows) < 2:
                continue
            mz = self.subject_mz[subject]
            if mz is None or mz not in self.mz_to_subjects:
                continue
            twins = [s for s in self.mz_to_subjects[mz] if s != subject and s in self.subject_rows]
            if twins:
                self.eligible_subjects.append(subject)

        if not self.eligible_subjects:
            raise ValueError("No eligible monozygotic twin subjects were found in the training frame.")

        self.all_subjects = list(self.subject_rows.keys())
        self.samples_per_epoch = (
            int(samples_per_epoch)
            if samples_per_epoch is not None
            else max(len(self.frame), len(self.eligible_subjects) * 20)
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _read(self, row_index: int):
        path = resolve_path(str(self.frame.loc[row_index, self.image_column]))
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, _: int):
        subject = random.choice(self.eligible_subjects)
        anchor_row, positive_row = random.sample(self.subject_rows[subject], 2)

        mz = self.subject_mz[subject]
        twin_subjects = [s for s in self.mz_to_subjects[mz] if s != subject and s in self.subject_rows]
        twin_subject = random.choice(twin_subjects)
        twin_row = random.choice(self.subject_rows[twin_subject])

        family = self.subject_family[subject]
        unrelated_candidates = [
            s for s in self.all_subjects if self.subject_family[s] != family
        ]
        unrelated_subject = random.choice(unrelated_candidates)
        unrelated_row = random.choice(self.subject_rows[unrelated_subject])

        return (
            self._read(anchor_row),
            self._read(positive_row),
            self._read(twin_row),
            self._read(unrelated_row),
        )


def build_transform(image_size: int, training: bool):
    ops = [transforms.Resize((image_size, image_size))]
    if training:
        ops += [
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05)
            ], p=0.30),
            transforms.RandomAffine(degrees=2, translate=(0.01, 0.01), scale=(0.99, 1.01)),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)


def configure_trainable_layers(model: torch.nn.Module, targets: list[str]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    matched = set()
    for name, module in model.named_modules():
        for target in targets:
            if name == target or name.startswith(f"{target}."):
                matched.add(target)
                for parameter in module.parameters():
                    parameter.requires_grad = True
    missing = set(targets) - matched
    if missing:
        print(f"[WARNING] Unmatched trainable modules: {sorted(missing)}")


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(first, second, dim=1)


def forward_four(model, anchor, positive, twin, unrelated):
    combined = torch.cat([anchor, positive, twin, unrelated], dim=0)
    embeddings = F.normalize(model(combined), dim=1)
    return torch.chunk(embeddings, chunks=4, dim=0)


def calculate_loss(student_embeddings, teacher_embeddings, cfg: dict):
    anchor, positive, twin, unrelated = student_embeddings
    d_ap = cosine_distance(anchor, positive)
    d_at = cosine_distance(anchor, twin)
    d_an = cosine_distance(anchor, unrelated)

    twin_loss = F.relu(d_ap - d_at + float(cfg["twin_margin"])).mean()
    unrelated_loss = F.relu(d_ap - d_an + float(cfg["unrelated_margin"])).mean()
    positive_loss = d_ap.mean()

    preserve = torch.stack([
        (1.0 - F.cosine_similarity(student, teacher, dim=1)).mean()
        for student, teacher in zip(student_embeddings, teacher_embeddings)
    ]).mean()

    total = (
        float(cfg["twin_loss_weight"]) * twin_loss
        + float(cfg["unrelated_loss_weight"]) * unrelated_loss
        + float(cfg["positive_compactness_weight"]) * positive_loss
        + float(cfg["preserve_pretrained_weight"]) * preserve
    )
    return {
        "total": total,
        "twin": twin_loss,
        "unrelated": unrelated_loss,
        "positive": positive_loss,
        "preserve": preserve,
    }


def prepare_assignments(config: dict, cfg: dict):
    columns = config["columns"]
    assignments = pd.read_csv(resolve_path(config["input"]["assignments_csv"]), low_memory=False)
    for column in [
        columns["image_id"], columns["subject_id"], columns["family_id"],
        columns["monozygotic_group_id"],
    ]:
        assignments[column] = normalize_identifier(assignments[column])
    assignments[columns["fold"]] = fold_as_int(assignments[columns["fold"]])

    protocols_dir = resolve_path(config["input"]["protocols_dir"])
    validation_weights = {str(k): float(v) for k, v in cfg["validation_protocols"].items()}
    protocol_id_sets = {
        name: protocol_image_ids(protocols_dir, name, columns["image_id"])
        for name in validation_weights
    }
    return assignments, protocol_id_sets, validation_weights


def build_training_frame(assignments: pd.DataFrame, config: dict, cfg: dict, mode: str, oof_fold: int | None):
    columns = config["columns"]
    protocols_dir = resolve_path(config["input"]["protocols_dir"])
    training_ids = union_protocol_ids(protocols_dir, list(cfg["training_protocols"]), columns["image_id"])

    frame = assignments.loc[
        (assignments[columns["split"]] == "development")
        & assignments[columns["image_id"]].isin(training_ids)
    ].copy()

    if mode == "tuning":
        frame = frame.loc[frame[columns["fold"]] != int(cfg["validation_fold"])].copy()
    elif mode == "oof":
        frame = frame.loc[frame[columns["fold"]] != int(oof_fold)].copy()
    elif mode != "final":
        raise ValueError(f"Unknown mode: {mode}")

    if frame.empty:
        raise ValueError("Training frame is empty after filtering.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--final", action="store_true")
    group.add_argument("--oof-fold", type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    cfg = config["twin_aware_facenet"]
    seed = int(cfg["seed"])
    seed_everything(seed)

    if args.oof_fold is not None:
        mode = "oof"
    elif args.final:
        mode = "final"
    else:
        mode = "tuning"

    assignments, protocol_id_sets, validation_weights = prepare_assignments(config, cfg)
    train_frame = build_training_frame(assignments, config, cfg, mode, args.oof_fold)
    columns = config["columns"]

    dataset = TwinTripletDataset(
        train_frame,
        cfg["image_column"],
        columns["subject_id"],
        columns["family_id"],
        columns["monozygotic_group_id"],
        build_transform(int(cfg["image_size"]), training=True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
    else:
        print("WARNING: CUDA is not available; training will use CPU.")

    num_workers = int(cfg["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    student = InceptionResnetV1(pretrained=str(cfg["pretrained"]), classify=False).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    configure_trainable_layers(student, list(cfg["unfreeze_modules"]))
    trainable_parameters = [p for p in student.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters):,}")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    if mode == "oof":
        epochs = int(cfg["oof_fixed_epochs"])
    elif mode == "final":
        epochs = int(cfg["final_epochs"])
    else:
        epochs = int(cfg["epochs"])

    checkpoints_dir = resolve_path(config["output"]["checkpoints_dir"])
    reports_dir = resolve_path(config["output"]["training_reports_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if mode == "oof":
        oof_dir = checkpoints_dir / cfg["oof"]["checkpoints_subdir"]
        oof_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = oof_dir / f"fold_{args.oof_fold}.pt"
        history_path = reports_dir / f"oof_fold_{args.oof_fold}_history.csv"
    elif mode == "final":
        checkpoint_path = checkpoints_dir / cfg["final_checkpoint_name"]
        history_path = reports_dir / "final_training_history.csv"
    else:
        checkpoint_path = checkpoints_dir / cfg["checkpoint_name"]
        history_path = reports_dir / "tuning_history.csv"

    history = []
    best_objective = float("inf")
    best_epoch = None
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        student.train()
        running = {k: [] for k in ["total", "twin", "unrelated", "positive", "preserve"]}
        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")

        for anchor, positive, twin, unrelated in progress:
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            twin = twin.to(device, non_blocking=True)
            unrelated = unrelated.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            student_embeddings = forward_four(student, anchor, positive, twin, unrelated)
            with torch.no_grad():
                teacher_embeddings = forward_four(teacher, anchor, positive, twin, unrelated)
            losses = calculate_loss(student_embeddings, teacher_embeddings, cfg)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            optimizer.step()

            for key in running:
                running[key].append(float(losses[key].item()))
            progress.set_postfix(
                loss=f"{np.mean(running['total'][-50:]):.4f}",
                twin=f"{np.mean(running['twin'][-50:]):.4f}",
            )

        row = {
            "epoch": epoch,
            **{f"train_{key}_loss": float(np.mean(values)) for key, values in running.items()},
        }

        if mode == "tuning":
            eers, objective = evaluate_validation_protocols(
                model=student,
                assignments=assignments,
                protocol_id_sets=protocol_id_sets,
                validation_fold=int(cfg["validation_fold"]),
                protocol_weights=validation_weights,
                image_column=cfg["image_column"],
                image_id_column=columns["image_id"],
                subject_column=columns["subject_id"],
                mz_column=columns["monozygotic_group_id"],
                split_column=columns["split"],
                fold_column=columns["fold"],
                image_size=int(cfg["image_size"]),
                device=device,
                batch_size=int(cfg["evaluation_batch_size"]),
                num_workers=num_workers,
                max_genuine_per_subject=int(cfg["max_genuine_pairs_per_subject"]),
                max_cotwin_per_subject_pair=int(cfg["max_cotwin_pairs_per_subject_pair"]),
                seed=seed,
            )
            row.update({f"eer::{name}": value for name, value in eers.items()})
            row["validation_objective"] = objective
            print("Validation EERs: " + " | ".join(f"{k}={v*100:.2f}%" for k, v in eers.items()))
            print(f"Weighted validation objective: {objective*100:.2f}%")

            if objective < best_objective - float(cfg["objective_min_delta"]):
                best_objective = objective
                best_epoch = epoch
                no_improvement = 0
                torch.save({
                    "state_dict": student.state_dict(),
                    "epoch": epoch,
                    "validation_objective": objective,
                    "validation_eers": eers,
                    "config": cfg,
                }, checkpoint_path)
                print(f"Saved best checkpoint: {checkpoint_path}")
            else:
                no_improvement += 1
        else:
            # OOF and final modes deliberately use a FIXED number of epochs.
            # The held-out OOF fold is never used for early stopping or model selection.
            if epoch == epochs:
                torch.save({
                    "state_dict": student.state_dict(),
                    "epoch": epoch,
                    "mode": mode,
                    "oof_fold": args.oof_fold,
                    "config": cfg,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if mode == "tuning" and no_improvement >= int(cfg["early_stopping_patience"]):
            print("Early stopping triggered.")
            break

    if mode == "tuning":
        summary = {
            "mode": mode,
            "best_epoch": best_epoch,
            "best_validation_objective": best_objective,
            "checkpoint": str(checkpoint_path),
        }
        with (reports_dir / "tuning_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(f"Best epoch: {best_epoch}")
    else:
        print(f"Completed {mode} training for {epochs} fixed epochs.")


if __name__ == "__main__":
    main()
