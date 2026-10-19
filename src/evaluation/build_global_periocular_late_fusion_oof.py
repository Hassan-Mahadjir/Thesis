from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import torch
import yaml

from src.global_local.common import (
    calculate_auc,
    calculate_eer,
    load_assignments_with_periocular,
    load_protocol_sets,
    resolve_path,
    union_protocol_ids,
)
from src.global_local.late_fusion_common import (
    build_global_backbone,
    build_protocol_score_data,
    extract_dual_embeddings,
    fused_scores,
    search_alpha,
)
from src.models.periocular_metric import PeriocularMetricEncoder


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_local_model(cfg):
    return PeriocularMetricEncoder(
        embedding_dim=int(cfg["local_embedding_dim"]),
        pretrained_imagenet=bool(cfg["local_pretrained_imagenet"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    cfg = config["late_fusion"]
    columns = config["columns"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    assignments = load_assignments_with_periocular(
        assignments_csv=resolve_path(config["input"]["assignments_csv"]),
        manifest_csv=resolve_path(config["input"]["periocular_manifest_csv"]),
        columns=columns,
        local_image_column=cfg["local_image_column"],
    )

    protocol_sets = load_protocol_sets(
        resolve_path(config["input"]["protocols_dir"]),
        list(cfg["evaluation_protocols"]),
        columns["image_id"],
    )

    all_ids = union_protocol_ids(protocol_sets)
    global_model = build_global_backbone(str(cfg["global_pretrained"])).to(device)

    checkpoint_root = (
        resolve_path(config["output"]["checkpoints_dir"])
        / cfg["oof"]["checkpoints_subdir"]
    )
    calibration_dir = resolve_path(config["output"]["calibration_dir"])
    score_dir = calibration_dir / "oof_scores"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    rows_by_protocol = {name: [] for name in cfg["evaluation_protocols"]}

    for fold in cfg["oof"]["folds"]:
        fold = int(fold)
        local_model = build_local_model(cfg).to(device)
        payload = torch.load(
            checkpoint_root / f"fold_{fold}.pt",
            map_location=device,
            weights_only=False,
        )
        local_model.load_state_dict(payload["model_state"])
        local_model.eval()

        held_out = assignments.loc[
            (assignments[columns["split"]] == "development")
            & (assignments[columns["fold"]] == fold)
            & assignments[columns["image_id"]].isin(all_ids)
            & assignments[cfg["global_image_column"]].notna()
            & assignments[cfg["local_image_column"]].notna()
        ].copy()

        print(f"Fold {fold}: {len(held_out):,} images")

        gf, lf, image_to_row = extract_dual_embeddings(
            global_model, local_model, held_out, columns["image_id"],
            cfg["global_image_column"], cfg["local_image_column"],
            int(cfg["global_image_size"]), int(cfg["local_height"]),
            int(cfg["local_width"]), device,
            int(cfg["evaluation_batch_size"]), int(cfg["num_workers"]),
        )

        protocol_data = build_protocol_score_data(
            held_out, protocol_sets, list(cfg["evaluation_protocols"]),
            columns, cfg, gf, lf, image_to_row
        )

        for protocol_name, data in protocol_data.items():
            for label, gs, ls in zip(
                data["labels"], data["global_scores"], data["local_scores"]
            ):
                rows_by_protocol[protocol_name].append({
                    "fold": fold,
                    "label": int(label),
                    "global_score": float(gs),
                    "local_score": float(ls),
                })

    combined = {}
    for protocol_name, rows in rows_by_protocol.items():
        frame = pd.DataFrame(rows)
        combined[protocol_name] = {
            "labels": frame["label"].to_numpy(dtype=np.int64),
            "global_scores": frame["global_score"].to_numpy(dtype=np.float32),
            "local_scores": frame["local_score"].to_numpy(dtype=np.float32),
        }
        frame.to_csv(score_dir / f"{protocol_name}_raw.csv", index=False)

    alpha, objective, _ = search_alpha(
        combined, cfg["validation_protocols"], float(cfg["alpha_step"])
    )

    print(f"Selected OOF alpha: {alpha:.2f}")
    print(f"Weighted OOF EER: {objective*100:.2f}%")

    fusion_config = {
        "alpha": float(alpha),
        "weighted_oof_objective": float(objective),
        "protocols": {},
    }

    for protocol_name, data in combined.items():
        fused = fused_scores(
            data["global_scores"], data["local_scores"], alpha
        )

        fused_eer, threshold = calculate_eer(data["labels"], fused)
        fused_auc = calculate_auc(data["labels"], fused)
        global_eer, _ = calculate_eer(data["labels"], data["global_scores"])
        local_eer, _ = calculate_eer(data["labels"], data["local_scores"])

        fusion_config["protocols"][protocol_name] = {
            "threshold": float(threshold),
            "oof_eer": float(fused_eer),
            "oof_auc": float(fused_auc),
            "global_only_oof_eer": float(global_eer),
            "local_only_oof_eer": float(local_eer),
            "pairs": int(len(data["labels"])),
        }

        pd.DataFrame({
            "label": data["labels"],
            "global_score": data["global_scores"],
            "local_score": data["local_scores"],
            "fused_score": fused,
        }).to_csv(score_dir / f"{protocol_name}.csv", index=False)

        print(
            f"{protocol_name}: "
            f"global={global_eer*100:.2f}% | "
            f"local={local_eer*100:.2f}% | "
            f"fused={fused_eer*100:.2f}%"
        )

    output_path = calibration_dir / cfg["oof"]["fusion_file"]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(fusion_config, f, indent=2)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
