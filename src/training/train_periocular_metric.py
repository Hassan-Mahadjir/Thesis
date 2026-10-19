from __future__ import annotations

import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.global_local.common import (
    build_local_transform,
    load_assignments_with_periocular,
    load_protocol_sets,
    resolve_path,
    union_protocol_ids,
)
from src.global_local.late_fusion_common import (
    build_global_backbone,
    build_protocol_score_data,
    extract_dual_embeddings,
    search_alpha,
)
from src.models.periocular_metric import PeriocularMetricEncoder


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class TwinMetricDataset(Dataset):
    def __init__(self, frame, columns, local_image_column,
                 local_height, local_width, samples_per_epoch):
        self.frame = frame.reset_index(drop=True).copy()
        self.columns = columns
        self.local_image_column = local_image_column
        self.samples_per_epoch = int(samples_per_epoch)
        self.subject_column = columns["subject_id"]
        self.family_column = columns["family_id"]
        self.mz_column = columns["monozygotic_group_id"]
        self.transform = build_local_transform(local_height, local_width, training=True)

        self.subject_rows = {
            str(s): g.index.to_list()
            for s, g in self.frame.groupby(self.subject_column)
        }
        self.subject_family = {
            str(s): str(g[self.family_column].iloc[0])
            for s, g in self.frame.groupby(self.subject_column)
        }
        self.subject_mz = {
            str(s): None if pd.isna(g[self.mz_column].iloc[0])
            else str(g[self.mz_column].iloc[0])
            for s, g in self.frame.groupby(self.subject_column)
        }

        self.mz_to_subjects = {}
        for subject, mz in self.subject_mz.items():
            if mz in (None, "", "<NA>", "nan", "None"):
                continue
            self.mz_to_subjects.setdefault(mz, []).append(subject)

        self.anchor_subjects = []
        for subject, rows in self.subject_rows.items():
            if len(rows) < 2:
                continue
            mz = self.subject_mz[subject]
            if mz in (None, "", "<NA>", "nan", "None"):
                continue
            twins = [
                other for other in self.mz_to_subjects.get(mz, [])
                if other != subject and other in self.subject_rows
            ]
            if twins:
                self.anchor_subjects.append(subject)

        self.all_subjects = list(self.subject_rows.keys())

        if not self.anchor_subjects:
            raise ValueError("No valid twin anchors found.")

    def __len__(self):
        return self.samples_per_epoch

    def _load(self, row_index):
        path = resolve_path(str(self.frame.loc[row_index, self.local_image_column]))
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def __getitem__(self, index):
        anchor_subject = random.choice(self.anchor_subjects)
        anchor_row, positive_row = random.sample(self.subject_rows[anchor_subject], 2)

        mz = self.subject_mz[anchor_subject]
        cotwin_subjects = [
            s for s in self.mz_to_subjects[mz]
            if s != anchor_subject and s in self.subject_rows
        ]
        cotwin_subject = random.choice(cotwin_subjects)
        cotwin_row = random.choice(self.subject_rows[cotwin_subject])

        family = self.subject_family[anchor_subject]
        unrelated_subject = random.choice([
            s for s in self.all_subjects
            if self.subject_family[s] != family
        ])
        unrelated_row = random.choice(self.subject_rows[unrelated_subject])

        return (
            self._load(anchor_row),
            self._load(positive_row),
            self._load(cotwin_row),
            self._load(unrelated_row),
        )


def metric_loss(a, p, t, n, cfg):
    d_ap = 1.0 - F.cosine_similarity(a, p, dim=1)
    d_at = 1.0 - F.cosine_similarity(a, t, dim=1)
    d_an = 1.0 - F.cosine_similarity(a, n, dim=1)

    twin_rank = F.relu(d_ap - d_at + float(cfg["twin_margin"])).mean()
    unrelated_rank = F.relu(d_at - d_an + float(cfg["unrelated_margin"])).mean()
    positive = d_ap.mean()

    return (
        float(cfg["twin_loss_weight"]) * twin_rank
        + float(cfg["unrelated_loss_weight"]) * unrelated_rank
        + float(cfg["positive_compactness_weight"]) * positive
    )


def build_model(cfg):
    return PeriocularMetricEncoder(
        embedding_dim=int(cfg["local_embedding_dim"]),
        pretrained_imagenet=bool(cfg["local_pretrained_imagenet"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--final", action="store_true")
    group.add_argument("--oof-fold", type=int, choices=[1,2,3,4], default=None)
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    cfg = config["late_fusion"]
    columns = config["columns"]

    seed_everything(int(cfg["seed"]))

    mode = "final" if args.final else ("oof" if args.oof_fold is not None else "tuning")

    assignments = load_assignments_with_periocular(
        assignments_csv=resolve_path(config["input"]["assignments_csv"]),
        manifest_csv=resolve_path(config["input"]["periocular_manifest_csv"]),
        columns=columns,
        local_image_column=cfg["local_image_column"],
    )

    protocols_dir = resolve_path(config["input"]["protocols_dir"])
    training_sets = load_protocol_sets(
        protocols_dir, list(cfg["training_protocols"]), columns["image_id"]
    )
    training_ids = union_protocol_ids(training_sets)

    train_frame = assignments.loc[
        (assignments[columns["split"]] == "development")
        & assignments[columns["image_id"]].isin(training_ids)
        & assignments[cfg["local_image_column"]].notna()
    ].copy()

    if mode == "tuning":
        train_frame = train_frame.loc[
            train_frame[columns["fold"]] != int(cfg["validation_fold"])
        ].copy()
    elif mode == "oof":
        train_frame = train_frame.loc[
            train_frame[columns["fold"]] != int(args.oof_fold)
        ].copy()

    print(f"Mode: {mode}")
    print(f"Training images: {len(train_frame):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA build: {torch.version.cuda}")

    local_model = build_model(cfg).to(device)

    global_model = None
    validation_sets = None
    if mode == "tuning":
        global_model = build_global_backbone(str(cfg["global_pretrained"])).to(device)
        validation_sets = load_protocol_sets(
            protocols_dir,
            list(cfg["validation_protocols"].keys()),
            columns["image_id"],
        )

    dataset = TwinMetricDataset(
        train_frame, columns, cfg["local_image_column"],
        int(cfg["local_height"]), int(cfg["local_width"]),
        int(cfg["samples_per_epoch"])
    )

    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))

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

    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    epochs = (
        int(cfg["epochs"]) if mode == "tuning"
        else int(cfg["oof_fixed_epochs"]) if mode == "oof"
        else int(cfg["final_epochs"])
    )

    checkpoint_root = resolve_path(config["output"]["checkpoints_dir"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if mode == "tuning":
        checkpoint_path = checkpoint_root / cfg["checkpoint_name"]
    elif mode == "final":
        checkpoint_path = checkpoint_root / cfg["final_checkpoint_name"]
    else:
        oof_dir = checkpoint_root / cfg["oof"]["checkpoints_subdir"]
        oof_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = oof_dir / f"fold_{args.oof_fold}.pt"

    report_dir = resolve_path(config["output"]["training_reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_objective = float("inf")
    best_epoch = None
    best_alpha = None
    patience = 0

    for epoch in range(1, epochs + 1):
        local_model.train()
        losses = []

        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for anchor, positive, cotwin, unrelated in progress:
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            cotwin = cotwin.to(device, non_blocking=True)
            unrelated = unrelated.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            embeddings = local_model(
                torch.cat([anchor, positive, cotwin, unrelated], dim=0)
            )
            a, p, t, n = torch.chunk(embeddings, 4, dim=0)

            loss = metric_loss(a, p, t, n, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
            optimizer.step()

            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}

        if mode == "tuning":
            validation_ids = union_protocol_ids(validation_sets)
            vf = assignments.loc[
                (assignments[columns["split"]] == "development")
                & (assignments[columns["fold"]] == int(cfg["validation_fold"]))
                & assignments[columns["image_id"]].isin(validation_ids)
                & assignments[cfg["global_image_column"]].notna()
                & assignments[cfg["local_image_column"]].notna()
            ].copy()

            gf, lf, image_to_row = extract_dual_embeddings(
                global_model, local_model, vf, columns["image_id"],
                cfg["global_image_column"], cfg["local_image_column"],
                int(cfg["global_image_size"]), int(cfg["local_height"]),
                int(cfg["local_width"]), device,
                int(cfg["evaluation_batch_size"]),
                int(cfg["num_workers"]),
            )

            protocol_data = build_protocol_score_data(
                vf, validation_sets,
                list(cfg["validation_protocols"].keys()),
                columns, cfg, gf, lf, image_to_row
            )

            alpha, objective, eers = search_alpha(
                protocol_data,
                cfg["validation_protocols"],
                float(cfg["alpha_step"]),
            )

            row["validation_objective"] = objective
            row["validation_alpha"] = alpha
            for name, eer in eers.items():
                row[f"eer::{name}"] = eer

            print(
                "\nValidation fused EERs: "
                + " | ".join(f"{k}={v*100:.2f}%" for k, v in eers.items())
            )
            print(f"Best alpha: {alpha:.2f}")
            print(f"Weighted fused EER: {objective*100:.2f}%")

            if objective < best_objective - float(cfg["objective_min_delta"]):
                best_objective = objective
                best_epoch = epoch
                best_alpha = alpha
                patience = 0
                torch.save(
                    {
                        "model_state": local_model.state_dict(),
                        "epoch": epoch,
                        "validation_objective": objective,
                        "validation_alpha": alpha,
                    },
                    checkpoint_path,
                )
                print(f"Saved new best: {checkpoint_path}")
            else:
                patience += 1
        else:
            torch.save(
                {
                    "model_state": local_model.state_dict(),
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
        pd.DataFrame(history).to_csv(report_dir / history_name, index=False)

        if mode == "tuning" and patience >= int(cfg["early_stopping_patience"]):
            print("Early stopping.")
            break

    summary = {
        "mode": mode,
        "epochs_completed": len(history),
        "best_epoch": best_epoch if mode == "tuning" else None,
        "best_validation_objective": best_objective if mode == "tuning" else None,
        "best_validation_alpha": best_alpha if mode == "tuning" else None,
        "checkpoint": str(checkpoint_path),
    }

    summary_name = {
        "tuning": "tuning_summary.json",
        "final": "final_training_summary.json",
        "oof": f"oof_fold_{args.oof_fold}_summary.json",
    }[mode]

    with (report_dir / summary_name).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    if mode == "tuning":
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation alpha: {best_alpha:.2f}")
        print(f"Best weighted fused EER: {best_objective*100:.2f}%")


if __name__ == "__main__":
    main()
