from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.global_local.common import (
    calculate_auc,
    calculate_eer,
    create_twin_verification_pairs,
    extract_global_local_embeddings,
    load_assignments_with_periocular,
    load_protocol_sets,
    resolve_path,
    score_pairs_with_head,
    union_protocol_ids,
)
from src.models.global_periocular import GlobalPeriocularVerifier


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_model(cfg: dict) -> GlobalPeriocularVerifier:
    return GlobalPeriocularVerifier(
        global_pretrained=str(cfg["global_pretrained"]),
        local_embedding_dim=int(cfg["local_embedding_dim"]),
        local_pretrained_imagenet=bool(cfg["local_pretrained_imagenet"]),
        hidden_1=int(cfg["pair_head_hidden_1"]),
        hidden_2=int(cfg["pair_head_hidden_2"]),
        dropout=float(cfg["dropout"]),
        freeze_global=bool(cfg["freeze_global"]),
    )


def load_checkpoint(
    model: GlobalPeriocularVerifier,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    cfg = config["global_periocular"]
    columns = config["columns"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    assignments = load_assignments_with_periocular(
        assignments_csv=resolve_path(config["input"]["assignments_csv"]),
        manifest_csv=resolve_path(
            config["input"]["periocular_manifest_csv"]
        ),
        columns=columns,
        local_image_column=cfg["local_image_column"],
    )

    protocol_sets = load_protocol_sets(
        resolve_path(config["input"]["protocols_dir"]),
        list(cfg["evaluation_protocols"]),
        columns["image_id"],
    )

    all_protocol_ids = union_protocol_ids(protocol_sets)

    checkpoint_root = resolve_path(
        config["output"]["checkpoints_dir"]
    ) / cfg["oof"]["checkpoints_subdir"]

    calibration_dir = resolve_path(
        config["output"]["calibration_dir"]
    )
    score_dir = calibration_dir / "oof_scores"

    calibration_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    per_protocol_rows: dict[str, list[dict]] = {
        name: []
        for name in cfg["evaluation_protocols"]
    }

    for fold in cfg["oof"]["folds"]:
        fold = int(fold)

        checkpoint_path = checkpoint_root / f"fold_{fold}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing OOF checkpoint: {checkpoint_path}"
            )

        model = build_model(cfg).to(device)
        load_checkpoint(model, checkpoint_path, device)
        model.eval()

        held_out = assignments.loc[
            (assignments[columns["split"]] == "development")
            & (assignments[columns["fold"]] == fold)
            & assignments[columns["image_id"]].isin(all_protocol_ids)
            & assignments[cfg["global_image_column"]].notna()
            & assignments[cfg["local_image_column"]].notna()
        ].copy()

        print(
            f"\nFold {fold}: "
            f"{len(held_out):,} held-out images"
        )

        global_features, local_features, image_to_row = (
            extract_global_local_embeddings(
                model=model,
                frame=held_out,
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

        for protocol_name in cfg["evaluation_protocols"]:
            protocol_frame = held_out.loc[
                held_out[columns["image_id"]].isin(
                    protocol_sets[protocol_name]
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

            genuine_scores, genuine_pairs = score_pairs_with_head(
                model,
                genuine,
                global_features,
                local_features,
                image_to_row,
                device,
                int(cfg["pair_score_batch_size"]),
            )

            cotwin_scores, cotwin_pairs = score_pairs_with_head(
                model,
                cotwin,
                global_features,
                local_features,
                image_to_row,
                device,
                int(cfg["pair_score_batch_size"]),
            )

            for (a, b), score in zip(
                genuine_pairs,
                genuine_scores,
            ):
                per_protocol_rows[protocol_name].append(
                    {
                        "fold": fold,
                        "image_id_1": a,
                        "image_id_2": b,
                        "pair_type": "genuine",
                        "label": 1,
                        "score": float(score),
                    }
                )

            for (a, b), score in zip(
                cotwin_pairs,
                cotwin_scores,
            ):
                per_protocol_rows[protocol_name].append(
                    {
                        "fold": fold,
                        "image_id_1": a,
                        "image_id_2": b,
                        "pair_type": "cotwin",
                        "label": 0,
                        "score": float(score),
                    }
                )

    thresholds = {}

    for protocol_name, rows in per_protocol_rows.items():
        frame = pd.DataFrame(rows)

        if frame.empty:
            raise ValueError(
                f"No OOF scores generated for {protocol_name}."
            )

        labels = frame["label"].to_numpy(dtype=np.int64)
        scores = frame["score"].to_numpy(dtype=np.float32)

        if len(np.unique(labels)) < 2:
            raise ValueError(
                f"{protocol_name} does not contain both classes."
            )

        eer, threshold = calculate_eer(labels, scores)
        auc = calculate_auc(labels, scores)

        thresholds[protocol_name] = {
            "threshold": float(threshold),
            "oof_eer": float(eer),
            "oof_auc": float(auc),
            "genuine_pairs": int(np.sum(labels == 1)),
            "cotwin_pairs": int(np.sum(labels == 0)),
        }

        frame.to_csv(
            score_dir / f"{protocol_name}.csv",
            index=False,
        )

        print(
            f"{protocol_name}: "
            f"EER={eer * 100:.2f}% | "
            f"AUC={auc * 100:.2f}% | "
            f"threshold={threshold:.6f}"
        )

    thresholds_path = (
        calibration_dir
        / cfg["oof"]["thresholds_file"]
    )

    with thresholds_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(thresholds, file, indent=2)

    print(f"\nSaved:\n{thresholds_path}")


if __name__ == "__main__":
    main()
