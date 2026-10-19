from __future__ import annotations

import argparse
import json
import pandas as pd
import torch
import yaml

from src.global_local.common import (
    calculate_auc,
    calculate_eer,
    calculate_threshold_metrics,
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
)
from src.models.periocular_metric import PeriocularMetricEncoder


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    fusion_path = (
        resolve_path(config["output"]["calibration_dir"])
        / cfg["oof"]["fusion_file"]
    )
    with fusion_path.open("r", encoding="utf-8") as f:
        fusion_config = json.load(f)

    alpha = float(fusion_config["alpha"])

    global_model = build_global_backbone(str(cfg["global_pretrained"])).to(device)
    local_model = PeriocularMetricEncoder(
        embedding_dim=int(cfg["local_embedding_dim"]),
        pretrained_imagenet=bool(cfg["local_pretrained_imagenet"]),
    ).to(device)

    checkpoint_path = (
        resolve_path(config["output"]["checkpoints_dir"])
        / cfg["final_checkpoint_name"]
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    local_model.load_state_dict(payload["model_state"])
    local_model.eval()

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

    final_test = assignments.loc[
        (assignments[columns["split"]] == "final_test")
        & assignments[columns["image_id"]].isin(union_protocol_ids(protocol_sets))
        & assignments[cfg["global_image_column"]].notna()
        & assignments[cfg["local_image_column"]].notna()
    ].copy()

    gf, lf, image_to_row = extract_dual_embeddings(
        global_model, local_model, final_test, columns["image_id"],
        cfg["global_image_column"], cfg["local_image_column"],
        int(cfg["global_image_size"]), int(cfg["local_height"]),
        int(cfg["local_width"]), device,
        int(cfg["evaluation_batch_size"]), int(cfg["num_workers"]),
    )

    protocol_data = build_protocol_score_data(
        final_test, protocol_sets, list(cfg["evaluation_protocols"]),
        columns, cfg, gf, lf, image_to_row
    )

    rows = []

    for protocol_name, data in protocol_data.items():
        fused = fused_scores(
            data["global_scores"], data["local_scores"], alpha
        )
        eer, test_eer_threshold = calculate_eer(data["labels"], fused)
        auc = calculate_auc(data["labels"], fused)

        global_eer, _ = calculate_eer(data["labels"], data["global_scores"])
        local_eer, _ = calculate_eer(data["labels"], data["local_scores"])

        threshold = float(
            fusion_config["protocols"][protocol_name]["threshold"]
        )
        tm = calculate_threshold_metrics(
            data["labels"], fused, threshold
        )

        rows.append({
            "feature_name": "global_periocular_late_fusion",
            "protocol": protocol_name,
            "task": "twin_verification",
            "alpha_global": alpha,
            "oof_threshold": threshold,
            "eer": float(eer),
            "roc_auc": float(auc),
            "global_only_eer": float(global_eer),
            "local_only_eer": float(local_eer),
            "test_eer_threshold": float(test_eer_threshold),
            **tm,
        })

        print(
            f"{protocol_name}: global={global_eer*100:.2f}% | "
            f"local={local_eer*100:.2f}% | "
            f"FUSED={eer*100:.2f}% | "
            f"Acc@OOF={tm['accuracy']*100:.2f}%"
        )

    out_dir = resolve_path(config["output"]["final_reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "verification_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
