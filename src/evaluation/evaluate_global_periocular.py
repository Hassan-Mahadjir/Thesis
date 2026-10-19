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
    calculate_threshold_metrics,
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional explicit final checkpoint. "
            "Defaults to output.checkpoints_dir/final_checkpoint_name."
        ),
    )

    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    cfg = config["global_periocular"]
    columns = config["columns"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint
        else (
            resolve_path(config["output"]["checkpoints_dir"])
            / cfg["final_checkpoint_name"]
        )
    )

    thresholds_path = (
        resolve_path(config["output"]["calibration_dir"])
        / cfg["oof"]["thresholds_file"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Final checkpoint not found: {checkpoint_path}"
        )

    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"OOF thresholds not found: {thresholds_path}"
        )

    with thresholds_path.open("r", encoding="utf-8") as file:
        thresholds = json.load(file)

    model = build_model(cfg).to(device)

    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(payload["model_state"])
    model.eval()

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

    final_test = assignments.loc[
        (assignments[columns["split"]] == "final_test")
        & assignments[columns["image_id"]].isin(
            union_protocol_ids(protocol_sets)
        )
        & assignments[cfg["global_image_column"]].notna()
        & assignments[cfg["local_image_column"]].notna()
    ].copy()

    print(f"Final-test images: {len(final_test):,}")

    global_features, local_features, image_to_row = (
        extract_global_local_embeddings(
            model=model,
            frame=final_test,
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

    report_rows = []

    for protocol_name in cfg["evaluation_protocols"]:
        protocol_frame = final_test.loc[
            final_test[columns["image_id"]].isin(
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

        labels = np.concatenate(
            [
                np.ones(len(genuine_scores), dtype=np.int64),
                np.zeros(len(cotwin_scores), dtype=np.int64),
            ]
        )
        scores = np.concatenate([genuine_scores, cotwin_scores])

        eer, test_eer_threshold = calculate_eer(labels, scores)
        auc = calculate_auc(labels, scores)

        oof_threshold = float(
            thresholds[protocol_name]["threshold"]
        )

        threshold_metrics = calculate_threshold_metrics(
            labels,
            scores,
            oof_threshold,
        )

        row = {
            "feature_name": "global_periocular",
            "protocol": protocol_name,
            "task": "twin_verification",
            "eer": float(eer),
            "roc_auc": float(auc),
            "oof_threshold": oof_threshold,
            "test_eer_threshold": float(test_eer_threshold),
            "genuine_pairs": int(len(genuine_scores)),
            "cotwin_pairs": int(len(cotwin_scores)),
            **threshold_metrics,
        }

        report_rows.append(row)

        print(
            f"{protocol_name}: "
            f"EER={eer * 100:.2f}% | "
            f"AUC={auc * 100:.2f}% | "
            f"Acc@OOF={threshold_metrics['accuracy'] * 100:.2f}%"
        )

    output_dir = resolve_path(
        config["output"]["final_reports_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "verification_summary.csv"

    pd.DataFrame(report_rows).to_csv(
        summary_path,
        index=False,
    )

    print(f"\nSaved:\n{summary_path}")


if __name__ == "__main__":
    main()
