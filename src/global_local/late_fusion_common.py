from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.global_local.common import (
    build_global_transform,
    build_local_transform,
    calculate_eer,
    create_twin_verification_pairs,
    normalize_identifier,
    resolve_path,
)


class DualEncodingDataset(Dataset):
    def __init__(self, frame, global_image_column, local_image_column,
                 global_image_size, local_height, local_width):
        self.frame = frame.reset_index(drop=True)
        self.global_image_column = global_image_column
        self.local_image_column = local_image_column
        self.global_transform = build_global_transform(global_image_size, training=False)
        self.local_transform = build_local_transform(local_height, local_width, training=False)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        gp = resolve_path(str(self.frame.loc[index, self.global_image_column]))
        lp = resolve_path(str(self.frame.loc[index, self.local_image_column]))
        gi = Image.open(gp).convert("RGB")
        li = Image.open(lp).convert("RGB")
        return index, self.global_transform(gi), self.local_transform(li)


def build_global_backbone(pretrained: str):
    model = InceptionResnetV1(pretrained=pretrained, classify=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_dual_embeddings(global_model, local_model, frame, image_id_column,
                            global_image_column, local_image_column,
                            global_image_size, local_height, local_width,
                            device, batch_size, num_workers):
    ds = DualEncodingDataset(
        frame, global_image_column, local_image_column,
        global_image_size, local_height, local_width
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0)
    )

    global_model.eval()
    local_model.eval()

    gf = lf = None

    for indices, global_images, local_images in dl:
        global_images = global_images.to(device, non_blocking=True)
        local_images = local_images.to(device, non_blocking=True)

        g = F.normalize(global_model(global_images), dim=1)
        l = F.normalize(local_model(local_images), dim=1)

        g = g.cpu().numpy().astype(np.float32)
        l = l.cpu().numpy().astype(np.float32)

        if gf is None:
            gf = np.empty((len(frame), g.shape[1]), dtype=np.float32)
            lf = np.empty((len(frame), l.shape[1]), dtype=np.float32)

        gf[indices.numpy()] = g
        lf[indices.numpy()] = l

    image_to_row = {
        str(image_id): int(i)
        for i, image_id in enumerate(normalize_identifier(frame[image_id_column]).tolist())
    }

    return gf, lf, image_to_row


def cosine_pair_scores(pairs, features, image_to_row):
    valid = [(a, b) for a, b in pairs if a in image_to_row and b in image_to_row]
    if not valid:
        return np.empty((0,), dtype=np.float32), []

    a_idx = np.asarray([image_to_row[a] for a, _ in valid], dtype=np.int64)
    b_idx = np.asarray([image_to_row[b] for _, b in valid], dtype=np.int64)

    scores = np.sum(features[a_idx] * features[b_idx], axis=1).astype(np.float32)
    return scores, valid


def fused_scores(global_scores, local_scores, alpha):
    return (
        float(alpha) * global_scores
        + (1.0 - float(alpha)) * local_scores
    ).astype(np.float32)


def build_protocol_score_data(frame, protocol_sets, protocol_names, columns, cfg,
                              global_features, local_features, image_to_row):
    out = {}

    for protocol_name in protocol_names:
        pf = frame.loc[
            frame[columns["image_id"]].isin(protocol_sets[protocol_name])
        ].copy()

        genuine, cotwin = create_twin_verification_pairs(
            frame=pf,
            image_id_column=columns["image_id"],
            subject_column=columns["subject_id"],
            mz_column=columns["monozygotic_group_id"],
            max_genuine_per_subject=int(cfg["max_genuine_pairs_per_subject"]),
            max_cotwin_per_subject_pair=int(cfg["max_cotwin_pairs_per_subject_pair"]),
            seed=int(cfg["seed"]),
        )

        gg, _ = cosine_pair_scores(genuine, global_features, image_to_row)
        cg, _ = cosine_pair_scores(cotwin, global_features, image_to_row)
        gl, _ = cosine_pair_scores(genuine, local_features, image_to_row)
        cl, _ = cosine_pair_scores(cotwin, local_features, image_to_row)

        out[protocol_name] = {
            "labels": np.concatenate([
                np.ones(len(gg), dtype=np.int64),
                np.zeros(len(cg), dtype=np.int64)
            ]),
            "global_scores": np.concatenate([gg, cg]),
            "local_scores": np.concatenate([gl, cl]),
        }

    return out


def search_alpha(protocol_data, protocol_weights, alpha_step):
    values = np.arange(0.0, 1.0 + alpha_step / 2.0, alpha_step)
    best_alpha = None
    best_objective = float("inf")
    best_eers = None

    for alpha in values:
        weighted = 0.0
        weight_sum = 0.0
        eers = {}

        for protocol_name, weight in protocol_weights.items():
            data = protocol_data[protocol_name]
            scores = fused_scores(
                data["global_scores"],
                data["local_scores"],
                alpha,
            )
            eer, _ = calculate_eer(data["labels"], scores)
            eers[protocol_name] = float(eer)
            weighted += float(weight) * float(eer)
            weight_sum += float(weight)

        objective = weighted / weight_sum

        if objective < best_objective:
            best_alpha = float(alpha)
            best_objective = float(objective)
            best_eers = eers

    return best_alpha, best_objective, best_eers
