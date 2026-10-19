from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ThresholdMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    true_accept_rate: float
    true_reject_rate: float
    tp: int
    tn: int
    fp: int
    fn: int


def calculate_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    return eer, float(thresholds[index])


def calculate_tar_at_far(labels: np.ndarray, scores: np.ndarray, target_far: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> ThresholdMetrics:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    far = float(fp / (fp + tn)) if (fp + tn) else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    return ThresholdMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        far=far,
        frr=frr,
        true_accept_rate=1.0 - frr,
        true_reject_rate=1.0 - far,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


def intrinsic_metrics(labels: np.ndarray, scores: np.ndarray, far_targets: list[float]) -> dict:
    if len(np.unique(labels)) < 2:
        raise ValueError("Both positive and negative classes are required.")
    auc = float(roc_auc_score(labels, scores))
    eer, eer_threshold = calculate_eer(labels, scores)
    return {
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "tar_at_far": {
            str(target): calculate_tar_at_far(labels, scores, target)
            for target in far_targets
        },
    }
