from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import calculate_eer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAIR_KEYS = [
    "image_id_1",
    "image_id_2",
    "pair_type",
    "label",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_scores(
    scores_root: Path,
    model_name: str,
    protocol: str,
    partition: str,
) -> pd.DataFrame:
    path = scores_root / model_name / protocol / f"{partition}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        dtype={
            "image_id_1": "string",
            "image_id_2": "string",
        },
        low_memory=False,
    )

    required = set(PAIR_KEYS + ["similarity"])
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return frame


def merge_model_scores(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge model scores while preserving pair metadata from the first model."""
    merged = None

    for model_name, frame in frames.items():
        if merged is None:
            # The first model is the canonical source for protocol, partition,
            # subject, family, pair-type, label, and any future metadata fields.
            part = frame.copy()
        else:
            # Later models contribute only their score and the keys needed to
            # align it with the canonical pair rows.
            part = frame[PAIR_KEYS + ["similarity"]].copy()

        part = part.rename(
            columns={
                "similarity": f"score::{model_name}"
            }
        )

        if merged is None:
            merged = part
        else:
            merged = merged.merge(
                part,
                on=PAIR_KEYS,
                how="inner",
                validate="one_to_one",
            )

    if merged is None:
        raise RuntimeError("No score frames were supplied.")

    expected_score_columns = [
        f"score::{model_name}"
        for model_name in frames.keys()
    ]
    missing_score_columns = [
        column
        for column in expected_score_columns
        if column not in merged.columns
    ]
    if missing_score_columns:
        raise RuntimeError(
            "Merged score table is missing "
            f"model scores: {missing_score_columns}"
        )

    return merged


def calculate_normalization(
    development: pd.DataFrame,
    model_names: list[str],
) -> dict[str, tuple[float, float]]:
    result = {}

    for model_name in model_names:
        column = f"score::{model_name}"

        mean = float(
            development[column].mean()
        )

        std = float(
            development[column].std()
        )

        std = max(
            std,
            1e-8,
        )

        result[model_name] = (
            mean,
            std,
        )

    return result


def normalize_scores(
    frame: pd.DataFrame,
    model_names: list[str],
    normalization: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    result = frame.copy()

    for model_name in model_names:
        source = f"score::{model_name}"
        target = f"z::{model_name}"

        mean, std = normalization[
            model_name
        ]

        result[target] = (
            result[source] - mean
        ) / std

    return result


def twin_task(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return frame.loc[
        frame["pair_type"].isin(
            [
                "genuine",
                "co_twin",
            ]
        )
    ].copy()


def choose_alpha(
    development: pd.DataFrame,
    first_model: str,
    second_model: str,
    alpha_step: float,
) -> tuple[float, float]:
    task = twin_task(
        development
    )

    labels = (
        task["label"]
        .astype(int)
        .to_numpy()
    )

    first = (
        task[f"z::{first_model}"]
        .astype(float)
        .to_numpy()
    )

    second = (
        task[f"z::{second_model}"]
        .astype(float)
        .to_numpy()
    )

    best_alpha = 0.5
    best_eer = float("inf")

    alphas = np.arange(
        0.0,
        1.0 + alpha_step / 2.0,
        alpha_step,
    )

    for alpha in alphas:
        fused = (
            alpha * first
            + (1.0 - alpha) * second
        )

        eer, _ = calculate_eer(
            labels,
            fused,
        )

        if eer < best_eer:
            best_eer = float(eer)
            best_alpha = float(alpha)

    return (
        best_alpha,
        best_eer,
    )


def fuse(
    frame: pd.DataFrame,
    first_model: str,
    second_model: str,
    alpha: float,
) -> pd.DataFrame:
    result = frame.copy()

    result["similarity"] = (
        alpha
        * result[
            f"z::{first_model}"
        ]
        + (
            1.0
            - alpha
        )
        * result[
            f"z::{second_model}"
        ]
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--baseline-config",
        required=True,
        help=(
            "Your existing baseline_benchmark.yaml. "
            "Used to read the protocol names and score root."
        ),
    )

    args = parser.parse_args()

    improvement_config = load_yaml(
        resolve_path(
            args.config
        )
    )

    baseline_config = load_yaml(
        resolve_path(
            args.baseline_config
        )
    )

    fusion_config = improvement_config[
        "score_fusion"
    ]

    model_names = list(
        fusion_config[
            "source_models"
        ]
    )

    if len(model_names) != 2:
        raise ValueError(
            "This implementation currently expects exactly two source models."
        )

    first_model = model_names[0]
    second_model = model_names[1]

    scores_root = resolve_path(
        baseline_config[
            "output"
        ][
            "scores_dir"
        ]
    )

    output_name = str(
        fusion_config[
            "output_name"
        ]
    )

    output_root = (
        scores_root
        / output_name
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    alpha_step = float(
        fusion_config.get(
            "alpha_step",
            0.01,
        )
    )

    fusion_summary = {}

    for protocol in baseline_config[
        "protocols"
    ]:
        print(
            f"\nProtocol: {protocol}"
        )

        development_frames = {
            model_name:
                load_scores(
                    scores_root=scores_root,
                    model_name=model_name,
                    protocol=protocol,
                    partition="development",
                )
            for model_name
            in model_names
        }

        final_test_frames = {
            model_name:
                load_scores(
                    scores_root=scores_root,
                    model_name=model_name,
                    protocol=protocol,
                    partition="final_test",
                )
            for model_name
            in model_names
        }

        development = merge_model_scores(
            development_frames
        )

        final_test = merge_model_scores(
            final_test_frames
        )

        normalization = (
            calculate_normalization(
                development=development,
                model_names=model_names,
            )
        )

        development = normalize_scores(
            development,
            model_names=model_names,
            normalization=normalization,
        )

        final_test = normalize_scores(
            final_test,
            model_names=model_names,
            normalization=normalization,
        )

        alpha, development_eer = (
            choose_alpha(
                development=development,
                first_model=first_model,
                second_model=second_model,
                alpha_step=alpha_step,
            )
        )

        print(
            f"  Best alpha for {first_model}: "
            f"{alpha:.2f}"
        )

        print(
            f"  DEVELOPMENT twin EER: "
            f"{development_eer:.4f}"
        )

        fused_development = fuse(
            development,
            first_model=first_model,
            second_model=second_model,
            alpha=alpha,
        )

        fused_final_test = fuse(
            final_test,
            first_model=first_model,
            second_model=second_model,
            alpha=alpha,
        )

        protocol_dir = (
            output_root
            / protocol
        )

        protocol_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        keep_columns = list(
            development_frames[
                first_model
            ].columns
        )

        if "similarity" not in keep_columns:
            keep_columns.append(
                "similarity"
            )

        fused_development[
            keep_columns
        ].to_csv(
            protocol_dir
            / "development.csv",
            index=False,
        )

        fused_final_test[
            keep_columns
        ].to_csv(
            protocol_dir
            / "final_test.csv",
            index=False,
        )

        fusion_summary[
            protocol
        ] = {
            "first_model":
                first_model,

            "second_model":
                second_model,

            "alpha_first_model":
                alpha,

            "alpha_second_model":
                1.0 - alpha,

            "development_twin_eer":
                development_eer,

            "normalization":
                {
                    model_name:
                        {
                            "mean":
                                normalization[
                                    model_name
                                ][
                                    0
                                ],

                            "std":
                                normalization[
                                    model_name
                                ][
                                    1
                                ],
                        }
                    for model_name
                    in model_names
                },
        }

    with (
        output_root
        / "fusion_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            fusion_summary,
            file,
            indent=2,
        )

    print(
        f"\nSaved fused scores to:\n"
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
