from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import intrinsic_metrics, metrics_at_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_task(frame: pd.DataFrame, task_name: str, seed: int) -> pd.DataFrame:
    genuine = frame.loc[frame["pair_type"] == "genuine"].copy()

    if task_name == "twin_verification":
        negative = frame.loc[frame["pair_type"] == "co_twin"].copy()
    elif task_name == "ordinary_verification":
        negative = frame.loc[frame["pair_type"] == "unrelated"].copy()
    elif task_name == "combined_verification":
        cotwin = frame.loc[frame["pair_type"] == "co_twin"].copy()
        unrelated = frame.loc[frame["pair_type"] == "unrelated"].copy()
        target_negative = len(genuine)
        target_cotwin = min(len(cotwin), target_negative // 2)
        target_unrelated = min(len(unrelated), target_negative - target_cotwin)
        rng = np.random.default_rng(seed)
        sampled = []
        if target_cotwin:
            sampled.append(cotwin.sample(n=target_cotwin, random_state=int(rng.integers(0, 2**31 - 1))))
        if target_unrelated:
            sampled.append(unrelated.sample(n=target_unrelated, random_state=int(rng.integers(0, 2**31 - 1))))
        negative = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame(columns=frame.columns)
    else:
        raise ValueError(task_name)

    if len(genuine) == 0 or len(negative) == 0:
        raise ValueError(f"{task_name} lacks both classes")
    return pd.concat([genuine, negative], ignore_index=True)


def distribution_summary(frame: pd.DataFrame) -> dict:
    result = {}
    for pair_type, group in frame.groupby("pair_type"):
        scores = group["similarity"].astype(float).to_numpy()
        result[str(pair_type)] = {
            "count": int(len(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "minimum": float(np.min(scores)),
            "maximum": float(np.max(scores)),
        }
    return result


def evaluate_task(development: pd.DataFrame, final_test: pd.DataFrame, task_name: str,
                  far_targets: list[float], seed: int) -> dict:
    dev = build_task(development, task_name, seed)
    test = build_task(final_test, task_name, seed + 1)
    dev_labels = dev["label"].astype(int).to_numpy()
    dev_scores = dev["similarity"].astype(float).to_numpy()
    test_labels = test["label"].astype(int).to_numpy()
    test_scores = test["similarity"].astype(float).to_numpy()

    dev_intrinsic = intrinsic_metrics(dev_labels, dev_scores, far_targets)
    selected_threshold = float(dev_intrinsic["eer_threshold"])
    dev_operating = metrics_at_threshold(dev_labels, dev_scores, selected_threshold)
    test_intrinsic = intrinsic_metrics(test_labels, test_scores, far_targets)
    test_operating = metrics_at_threshold(test_labels, test_scores, selected_threshold)

    return {
        "development": {"pairs": int(len(dev)), "intrinsic": dev_intrinsic, "operating": vars(dev_operating)},
        "selected_threshold": selected_threshold,
        "final_test": {
            "pairs": int(len(test)),
            "intrinsic": test_intrinsic,
            "operating_at_development_threshold": vars(test_operating),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-name", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))

    scores_root = resolve_path(config["output"]["scores_dir"]) / args.feature_name
    report_root = resolve_path(config["output"]["report_dir"]) / args.feature_name
    report_root.mkdir(parents=True, exist_ok=True)
    far_targets = [float(x) for x in config["evaluation"]["far_targets"]]
    seed = int(config["evaluation"]["seed"])
    summary_records = []

    for protocol_index, protocol_name in enumerate(config["protocols"]):
        development = pd.read_csv(scores_root / protocol_name / "development.csv", low_memory=False)
        final_test = pd.read_csv(scores_root / protocol_name / "final_test.csv", low_memory=False)
        result = {
            "feature_name": args.feature_name,
            "protocol": protocol_name,
            "similarity_distributions": {
                "development": distribution_summary(development),
                "final_test": distribution_summary(final_test),
            },
            "tasks": {},
        }

        for task_index, task_name in enumerate(["twin_verification", "ordinary_verification", "combined_verification"]):
            try:
                task_result = evaluate_task(
                    development, final_test, task_name, far_targets,
                    seed + protocol_index * 100 + task_index,
                )
                result["tasks"][task_name] = task_result
                intrinsic = task_result["final_test"]["intrinsic"]
                operating = task_result["final_test"]["operating_at_development_threshold"]
                summary_records.append({
                    "feature_name": args.feature_name,
                    "protocol": protocol_name,
                    "task": task_name,
                    "eer": intrinsic["eer"],
                    "roc_auc": intrinsic["roc_auc"],
                    "accuracy": operating["accuracy"],
                    "precision": operating["precision"],
                    "recall": operating["recall"],
                    "f1": operating["f1"],
                    "far": operating["far"],
                    "frr": operating["frr"],
                    "tar_at_far_1_percent": intrinsic["tar_at_far"].get("0.01"),
                    "tar_at_far_0_1_percent": intrinsic["tar_at_far"].get("0.001"),
                    "development_threshold": task_result["selected_threshold"],
                })
            except ValueError as error:
                result["tasks"][task_name] = {"status": "SKIPPED", "reason": str(error)}

        with (report_root / f"{protocol_name}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    pd.DataFrame(summary_records).to_csv(report_root / "verification_summary.csv", index=False)
    print(f"Saved verification results to: {report_root}")


if __name__ == "__main__":
    main()
