from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class PartitionResult:
    assignments: dict[str, int]
    score: float
    quotas: list[int]


# ============================================================
# PATH / CONFIG
# ============================================================


def resolve_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def load_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# GENERAL HELPERS
# ============================================================


def normalize_bool(
    series: pd.Series,
) -> pd.Series:
    """
    Convert common boolean representations into real bool values.
    """

    if pd.api.types.is_bool_dtype(
        series
    ):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }

    return (
        normalized
        .isin(
            true_values
        )
        .fillna(False)
        .astype(bool)
    )


def ensure_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required metadata columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


def normalize_identifier(
    series: pd.Series,
) -> pd.Series:
    """
    Keep identifiers as strings without accidental '.0' suffixes.
    """

    result = (
        series
        .astype("string")
        .str.strip()
    )

    result = result.str.replace(
        r"\.0$",
        "",
        regex=True,
    )

    return result


# ============================================================
# VALIDATE SOURCE METADATA
# ============================================================


def validate_metadata(
    df: pd.DataFrame,
    image_column: str,
    subject_column: str,
    family_column: str,
    require_unique_image_ids: bool,
) -> None:

    if df[
        family_column
    ].isna().any():
        count = int(
            df[
                family_column
            ].isna().sum()
        )

        raise ValueError(
            f"{count:,} usable images have no family ID."
        )

    if df[
        subject_column
    ].isna().any():
        count = int(
            df[
                subject_column
            ].isna().sum()
        )

        raise ValueError(
            f"{count:,} usable images have no subject ID."
        )

    # --------------------------------------------------------
    # Every subject must map to exactly one family.
    # --------------------------------------------------------

    subject_family_counts = (
        df.groupby(
            subject_column
        )[
            family_column
        ]
        .nunique()
    )

    problematic_subjects = (
        subject_family_counts[
            subject_family_counts > 1
        ]
    )

    if not problematic_subjects.empty:
        raise ValueError(
            "Some subjects belong to more than one family:\n"
            + problematic_subjects.to_string()
        )

    # --------------------------------------------------------
    # Image IDs should be unique.
    # --------------------------------------------------------

    if (
        require_unique_image_ids
        and image_column in df.columns
    ):
        duplicated = df[
            image_column
        ].duplicated(
            keep=False
        )

        if duplicated.any():
            examples = (
                df.loc[
                    duplicated,
                    image_column,
                ]
                .head(20)
                .tolist()
            )

            raise ValueError(
                "Duplicate image IDs were found. Examples:\n"
                f"{examples}"
            )


# ============================================================
# INTEGER QUOTAS
# ============================================================


def calculate_quotas(
    number_of_groups: int,
    ratios: list[float],
) -> list[int]:
    """
    Largest-remainder allocation.

    Example:
        227 families with 80/20
        -> [182, 45]
    """

    ratios_array = np.asarray(
        ratios,
        dtype=np.float64,
    )

    ratios_array = (
        ratios_array
        / ratios_array.sum()
    )

    raw = (
        number_of_groups
        * ratios_array
    )

    base = np.floor(
        raw
    ).astype(int)

    remaining = (
        number_of_groups
        - int(
            base.sum()
        )
    )

    remainders = (
        raw
        - base
    )

    order = np.argsort(
        -remainders
    )

    for index in order[
        :remaining
    ]:
        base[
            index
        ] += 1

    return base.tolist()


# ============================================================
# FAMILY FEATURE MATRIX
# ============================================================


def build_family_features(
    df: pd.DataFrame,
    family_column: str,
    subject_column: str,
    condition_columns: list[str],
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    """
    Build one row per family.

    Features include:
        subject count
        image count
        counts for each configured condition/category
    """

    base = (
        df.groupby(
            family_column
        )
        .agg(
            subject_count=(
                subject_column,
                "nunique",
            ),
            image_count=(
                subject_column,
                "size",
            ),
        )
    )

    feature_frames = [
        base
    ]

    condition_feature_columns = []

    for column in condition_columns:

        if column not in df.columns:
            print(
                f"[WARNING] Condition column "
                f"'{column}' not found. Skipping."
            )
            continue

        values = (
            df[
                column
            ]
            .astype("string")
            .fillna("<missing>")
        )

        temporary = pd.DataFrame(
            {
                family_column:
                    df[
                        family_column
                    ].values,

                "_condition":
                    values.values,
            },
            index=df.index,
        )

        table = pd.crosstab(
            temporary[
                family_column
            ],
            temporary[
                "_condition"
            ],
        )

        renamed = {}

        for category in table.columns:
            new_name = (
                f"condition::{column}::{category}"
            )

            renamed[
                category
            ] = new_name

            condition_feature_columns.append(
                new_name
            )

        table = table.rename(
            columns=renamed
        )

        feature_frames.append(
            table
        )

    features = pd.concat(
        feature_frames,
        axis=1,
    ).fillna(0)

    features = features.sort_index()

    return (
        features,
        condition_feature_columns,
    )


# ============================================================
# OBJECTIVE
# ============================================================


def build_feature_weights(
    feature_columns: list[str],
    subject_weight: float,
    image_weight: float,
    condition_weight: float,
) -> np.ndarray:

    weights = []

    for column in feature_columns:

        if column == "subject_count":
            weights.append(
                subject_weight
            )

        elif column == "image_count":
            weights.append(
                image_weight
            )

        elif column.startswith(
            "condition::"
        ):
            weights.append(
                condition_weight
            )

        else:
            weights.append(
                1.0
            )

    return np.asarray(
        weights,
        dtype=np.float64,
    )


def normalized_partition_score(
    totals: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Relative squared error.

    Normalization prevents large-count dimensions from completely
    dominating the objective.
    """

    denominator = np.maximum(
        targets,
        1.0,
    )

    error = (
        totals
        - targets
    ) / denominator

    weighted = (
        error ** 2
        * weights[
            np.newaxis,
            :
        ]
    )

    return float(
        weighted.sum()
    )


# ============================================================
# PARTITION OPTIMIZATION
# ============================================================


def partition_families(
    family_features: pd.DataFrame,
    ratios: list[float],
    seed: int,
    restarts: int,
    subject_weight: float,
    image_weight: float,
    condition_weight: float,
    local_search_iterations: int = 5000,
) -> PartitionResult:
    """
    Partition complete families while preserving exact family quotas.

    Each restart begins with a complete random assignment. The local
    search then swaps whole families between partitions, accepting a
    swap only when it improves the objective. Family quotas therefore
    remain exact throughout optimization.
    """

    family_ids = (
        family_features
        .index
        .astype(str)
        .tolist()
    )

    number_of_families = len(
        family_ids
    )

    number_of_partitions = len(
        ratios
    )

    if number_of_families == 0:
        raise ValueError(
            "No families were supplied to partition_families()."
        )

    if number_of_partitions < 2:
        raise ValueError(
            "At least two partitions are required."
        )

    if restarts < 1:
        raise ValueError(
            "restarts must be at least 1."
        )

    if local_search_iterations < 0:
        raise ValueError(
            "local_search_iterations cannot be negative."
        )

    ratios_array = np.asarray(
        ratios,
        dtype=np.float64,
    )

    if (
        not np.isfinite(
            ratios_array
        ).all()
        or (
            ratios_array < 0
        ).any()
        or ratios_array.sum() <= 0
    ):
        raise ValueError(
            "Partition ratios must be finite, non-negative, "
            "and have a positive sum."
        )

    quotas = calculate_quotas(
        number_of_groups=number_of_families,
        ratios=ratios,
    )

    proportions = (
        np.asarray(
            quotas,
            dtype=np.float64,
        )
        / number_of_families
    )

    feature_columns = (
        family_features.columns.tolist()
    )

    values = (
        family_features[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    global_totals = (
        values.sum(
            axis=0
        )
    )

    targets = (
        proportions[
            :,
            np.newaxis,
        ]
        * global_totals[
            np.newaxis,
            :
        ]
    )

    weights = build_feature_weights(
        feature_columns=feature_columns,
        subject_weight=subject_weight,
        image_weight=image_weight,
        condition_weight=condition_weight,
    )

    best_score = math.inf
    best_assignments = None

    rng = np.random.default_rng(
        seed
    )

    nonempty_partitions = np.flatnonzero(
        np.asarray(
            quotas
        )
        > 0
    )

    for _ in range(
        restarts
    ):
        permutation = rng.permutation(
            number_of_families
        )
        assignments_array = np.full(
            number_of_families,
            -1,
            dtype=np.int64,
        )

        partition_members: list[list[int]] = []
        start = 0

        for partition, quota in enumerate(
            quotas
        ):
            stop = start + quota
            members = permutation[
                start:stop
            ].tolist()
            partition_members.append(
                members
            )
            assignments_array[
                members
            ] = partition
            start = stop

        partition_totals = np.vstack(
            [
                values[
                    assignments_array == partition
                ].sum(
                    axis=0
                )
                for partition in range(
                    number_of_partitions
                )
            ]
        )

        current_score = (
            normalized_partition_score(
                totals=partition_totals,
                targets=targets,
                weights=weights,
            )
        )

        for _ in range(
            local_search_iterations
        ):
            if len(
                nonempty_partitions
            ) < 2:
                break

            first_partition, second_partition = (
                rng.choice(
                    nonempty_partitions,
                    size=2,
                    replace=False,
                )
            )

            first_position = int(
                rng.integers(
                    len(
                        partition_members[
                            first_partition
                        ]
                    )
                )
            )
            second_position = int(
                rng.integers(
                    len(
                        partition_members[
                            second_partition
                        ]
                    )
                )
            )

            first_family = partition_members[
                first_partition
            ][
                first_position
            ]
            second_family = partition_members[
                second_partition
            ][
                second_position
            ]

            candidate_totals = partition_totals.copy()
            candidate_totals[
                first_partition
            ] += (
                values[
                    second_family
                ]
                - values[
                    first_family
                ]
            )
            candidate_totals[
                second_partition
            ] += (
                values[
                    first_family
                ]
                - values[
                    second_family
                ]
            )

            candidate_score = normalized_partition_score(
                totals=candidate_totals,
                targets=targets,
                weights=weights,
            )

            if candidate_score >= current_score:
                continue

            partition_members[
                first_partition
            ][
                first_position
            ] = second_family
            partition_members[
                second_partition
            ][
                second_position
            ] = first_family
            assignments_array[
                first_family
            ] = second_partition
            assignments_array[
                second_family
            ] = first_partition
            partition_totals = candidate_totals
            current_score = candidate_score

        if current_score < best_score:

            best_score = current_score

            best_assignments = {
                family_ids[index]:
                    int(
                        assignments_array[
                            index
                        ]
                    )
                for index in range(
                    number_of_families
                )
            }

    if best_assignments is None:
        raise RuntimeError(
            "Could not create a valid partition."
        )

    return PartitionResult(
        assignments=best_assignments,
        score=float(
            best_score
        ),
        quotas=quotas,
    )


# ============================================================
# ASSIGNMENT HELPERS
# ============================================================


def map_family_assignments(
    df: pd.DataFrame,
    family_column: str,
    assignments: dict[str, int],
) -> pd.Series:

    family_values = (
        df[
            family_column
        ]
        .astype(str)
    )

    result = family_values.map(
        assignments
    )

    if result.isna().any():
        missing = (
            family_values[
                result.isna()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Some families did not receive an assignment:\n"
            f"{missing[:20]}"
        )

    return result.astype(int)


# ============================================================
# LEAKAGE AUDIT
# ============================================================


def set_overlap(
    first: pd.Series,
    second: pd.Series,
) -> set[str]:

    first_set = set(
        first
        .dropna()
        .astype(str)
    )

    second_set = set(
        second
        .dropna()
        .astype(str)
    )

    return (
        first_set
        & second_set
    )


def run_leakage_audit(
    development: pd.DataFrame,
    final_test: pd.DataFrame,
    folds: dict[int, pd.DataFrame],
    family_column: str,
    subject_column: str,
    total_usable_rows: int,
) -> dict:

    dev_test_family_overlap = set_overlap(
        development[
            family_column
        ],
        final_test[
            family_column
        ],
    )

    dev_test_subject_overlap = set_overlap(
        development[
            subject_column
        ],
        final_test[
            subject_column
        ],
    )

    fold_family_overlaps = {}
    fold_subject_overlaps = {}

    fold_numbers = sorted(
        folds.keys()
    )

    for i in range(
        len(
            fold_numbers
        )
    ):
        for j in range(
            i + 1,
            len(
                fold_numbers
            ),
        ):
            first_number = fold_numbers[
                i
            ]

            second_number = fold_numbers[
                j
            ]

            pair_name = (
                f"fold_{first_number}"
                f"_vs_fold_{second_number}"
            )

            first = folds[
                first_number
            ]

            second = folds[
                second_number
            ]

            family_overlap = set_overlap(
                first[
                    family_column
                ],
                second[
                    family_column
                ],
            )

            subject_overlap = set_overlap(
                first[
                    subject_column
                ],
                second[
                    subject_column
                ],
            )

            fold_family_overlaps[
                pair_name
            ] = sorted(
                family_overlap
            )

            fold_subject_overlaps[
                pair_name
            ] = sorted(
                subject_overlap
            )

    assigned_rows = (
        len(
            development
        )
        + len(
            final_test
        )
    )

    fold_rows = sum(
        len(
            fold
        )
        for fold
        in folds.values()
    )

    development_family_set = set(
        development[
            family_column
        ].astype(str)
    )

    fold_family_union = set()

    for fold in folds.values():
        fold_family_union.update(
            fold[
                family_column
            ].astype(str)
        )

    development_subject_set = set(
        development[
            subject_column
        ].astype(str)
    )

    fold_subject_union = set()

    for fold in folds.values():
        fold_subject_union.update(
            fold[
                subject_column
            ].astype(str)
        )

    passed = (
        len(
            dev_test_family_overlap
        )
        == 0
        and len(
            dev_test_subject_overlap
        )
        == 0
        and all(
            len(
                overlap
            )
            == 0
            for overlap
            in fold_family_overlaps.values()
        )
        and all(
            len(
                overlap
            )
            == 0
            for overlap
            in fold_subject_overlaps.values()
        )
        and assigned_rows
        == total_usable_rows
        and fold_rows
        == len(
            development
        )
        and fold_family_union
        == development_family_set
        and fold_subject_union
        == development_subject_set
    )

    return {
        "status":
            (
                "PASS"
                if passed
                else "FAIL"
            ),

        "development_final_test":
            {
                "family_overlap_count":
                    len(
                        dev_test_family_overlap
                    ),

                "family_overlap":
                    sorted(
                        dev_test_family_overlap
                    ),

                "subject_overlap_count":
                    len(
                        dev_test_subject_overlap
                    ),

                "subject_overlap":
                    sorted(
                        dev_test_subject_overlap
                    ),
            },

        "development_folds":
            {
                "family_overlaps":
                    fold_family_overlaps,

                "subject_overlaps":
                    fold_subject_overlaps,

                "fold_rows":
                    fold_rows,

                "development_rows":
                    len(
                        development
                    ),

                "all_development_families_present":
                    (
                        fold_family_union
                        == development_family_set
                    ),

                "all_development_subjects_present":
                    (
                        fold_subject_union
                        == development_subject_set
                    ),
            },

        "assignment":
            {
                "total_usable_rows":
                    total_usable_rows,

                "assigned_rows":
                    assigned_rows,

                "all_usable_rows_assigned":
                    (
                        assigned_rows
                        == total_usable_rows
                    ),
            },
    }


# ============================================================
# REPORTS
# ============================================================


def split_statistics(
    df: pd.DataFrame,
    family_column: str,
    subject_column: str,
) -> dict:

    return {
        "images":
            int(
                len(
                    df
                )
            ),

        "subjects":
            int(
                df[
                    subject_column
                ].nunique()
            ),

        "families":
            int(
                df[
                    family_column
                ].nunique()
            ),
    }


def build_condition_distribution(
    partitions: dict[str, pd.DataFrame],
    condition_columns: list[str],
) -> pd.DataFrame:

    records = []

    for partition_name, partition_df in (
        partitions.items()
    ):

        for column in condition_columns:

            if column not in partition_df.columns:
                continue

            values = (
                partition_df[
                    column
                ]
                .astype("string")
                .fillna("<missing>")
            )

            counts = values.value_counts(
                dropna=False
            )

            total = len(
                partition_df
            )

            for value, count in counts.items():

                records.append(
                    {
                        "partition":
                            partition_name,

                        "condition":
                            column,

                        "value":
                            str(
                                value
                            ),

                        "count":
                            int(
                                count
                            ),

                        "percentage":
                            (
                                float(
                                    count
                                    / total
                                    * 100.0
                                )
                                if total > 0
                                else 0.0
                            ),
                    }
                )

    return pd.DataFrame(
        records
    )


def build_partition_summary_table(
    partitions: dict[str, pd.DataFrame],
    family_column: str,
    subject_column: str,
) -> pd.DataFrame:

    records = []

    for name, frame in (
        partitions.items()
    ):

        stats = split_statistics(
            frame,
            family_column=family_column,
            subject_column=subject_column,
        )

        records.append(
            {
                "partition":
                    name,

                **stats,
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# OUTPUT
# ============================================================


def save_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )


def save_json(
    value: dict,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            value,
            file,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create family-disjoint ND-TWINS "
            "development/final-test splits and "
            "family-disjoint development folds."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to split YAML configuration."
        ),
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    config = load_yaml(
        config_path
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    metadata_path = resolve_path(
        config[
            "input"
        ][
            "metadata_csv"
        ]
    )

    split_dir = resolve_path(
        config[
            "output"
        ][
            "split_dir"
        ]
    )

    report_dir = resolve_path(
        config[
            "output"
        ][
            "report_dir"
        ]
    )

    image_column = config[
        "columns"
    ][
        "image_id"
    ]

    subject_column = config[
        "columns"
    ][
        "subject_id"
    ]

    family_column = config[
        "columns"
    ][
        "family_id"
    ]

    usable_column = config[
        "columns"
    ][
        "usable"
    ]

    seed = int(
        config[
            "split"
        ][
            "seed"
        ]
    )

    development_ratio = float(
        config[
            "split"
        ][
            "development_ratio"
        ]
    )

    final_test_ratio = float(
        config[
            "split"
        ][
            "final_test_ratio"
        ]
    )

    number_of_folds = int(
        config[
            "split"
        ][
            "number_of_folds"
        ]
    )

    restarts = int(
        config[
            "split"
        ].get(
            "optimization_restarts",
            250,
        )
    )

    local_search_iterations = int(
        config[
            "split"
        ].get(
            "local_search_iterations",
            5000,
        )
    )

    subject_weight = float(
        config[
            "balance"
        ].get(
            "subject_weight",
            3.0,
        )
    )

    image_weight = float(
        config[
            "balance"
        ].get(
            "image_weight",
            3.0,
        )
    )

    condition_weight = float(
        config[
            "balance"
        ].get(
            "condition_weight",
            0.5,
        )
    )

    condition_columns = list(
        config[
            "balance"
        ].get(
            "condition_columns",
            [],
        )
    )

    require_unique_image_ids = bool(
        config[
            "validation"
        ].get(
            "require_unique_image_ids",
            True,
        )
    )

    # --------------------------------------------------------
    # Load QC v1 metadata
    # --------------------------------------------------------

    print(
        "Loading frozen QC metadata:"
    )

    print(
        metadata_path
    )

    df = pd.read_csv(
        metadata_path,
        low_memory=False,
    )

    ensure_columns(
        df,
        [
            subject_column,
            family_column,
            usable_column,
        ]
        + (
            [
                image_column
            ]
            if require_unique_image_ids
            else []
        ),
    )

    # --------------------------------------------------------
    # Filter usable images
    # --------------------------------------------------------

    usable_mask = normalize_bool(
        df[
            usable_column
        ]
    )

    usable_df = (
        df.loc[
            usable_mask
        ]
        .copy()
    )

    usable_df[
        "_source_row"
    ] = usable_df.index

    usable_df[
        subject_column
    ] = normalize_identifier(
        usable_df[
            subject_column
        ]
    )

    usable_df[
        family_column
    ] = normalize_identifier(
        usable_df[
            family_column
        ]
    )

    if image_column in usable_df.columns:
        usable_df[
            image_column
        ] = normalize_identifier(
            usable_df[
                image_column
            ]
        )

    print(
        f"\nUsable images: "
        f"{len(usable_df):,}"
    )

    print(
        f"Subjects: "
        f"{usable_df[subject_column].nunique():,}"
    )

    print(
        f"Families: "
        f"{usable_df[family_column].nunique():,}"
    )

    # --------------------------------------------------------
    # Validate family structure
    # --------------------------------------------------------

    validate_metadata(
        df=usable_df,
        image_column=image_column,
        subject_column=subject_column,
        family_column=family_column,
        require_unique_image_ids=(
            require_unique_image_ids
        ),
    )

    print(
        "\nMetadata validation: PASS"
    )

    # --------------------------------------------------------
    # Build family-level feature table
    # --------------------------------------------------------

    family_features, _ = (
        build_family_features(
            df=usable_df,
            family_column=family_column,
            subject_column=subject_column,
            condition_columns=condition_columns,
        )
    )

    print(
        f"\nFamily feature matrix: "
        f"{family_features.shape[0]:,} families x "
        f"{family_features.shape[1]:,} features"
    )

    # ========================================================
    # OUTER SPLIT
    # ========================================================

    print(
        "\nCreating family-disjoint outer split..."
    )

    outer_result = partition_families(
        family_features=family_features,
        ratios=[
            development_ratio,
            final_test_ratio,
        ],
        seed=seed,
        restarts=restarts,
        subject_weight=subject_weight,
        image_weight=image_weight,
        condition_weight=condition_weight,
        local_search_iterations=(
            local_search_iterations
        ),
    )

    print(
        f"Outer family quotas: "
        f"{outer_result.quotas}"
    )

    print(
        f"Outer balance score: "
        f"{outer_result.score:.6f}"
    )

    outer_assignment = (
        map_family_assignments(
            df=usable_df,
            family_column=family_column,
            assignments=(
                outer_result.assignments
            ),
        )
    )

    usable_df[
        "split"
    ] = np.where(
        outer_assignment == 0,
        "development",
        "final_test",
    )

    development = (
        usable_df.loc[
            usable_df[
                "split"
            ]
            == "development"
        ]
        .copy()
    )

    final_test = (
        usable_df.loc[
            usable_df[
                "split"
            ]
            == "final_test"
        ]
        .copy()
    )

    # ========================================================
    # DEVELOPMENT FOLDS
    # ========================================================

    print(
        "\nCreating development folds..."
    )

    development_family_features = (
        family_features.loc[
            sorted(
                development[
                    family_column
                ]
                .unique()
            )
        ]
    )

    fold_result = partition_families(
        family_features=(
            development_family_features
        ),
        ratios=[
            1.0
            for _ in range(
                number_of_folds
            )
        ],
        seed=(
            seed
            + 1000
        ),
        restarts=restarts,
        subject_weight=subject_weight,
        image_weight=image_weight,
        condition_weight=condition_weight,
        local_search_iterations=(
            local_search_iterations
        ),
    )

    print(
        f"Fold family quotas: "
        f"{fold_result.quotas}"
    )

    print(
        f"Fold balance score: "
        f"{fold_result.score:.6f}"
    )

    development[
        "fold"
    ] = (
        map_family_assignments(
            df=development,
            family_column=family_column,
            assignments=(
                fold_result.assignments
            ),
        )
        + 1
    )

    final_test[
        "fold"
    ] = pd.NA

    # --------------------------------------------------------
    # Put fold assignments back into complete usable dataframe.
    # --------------------------------------------------------

    fold_map = (
        development[
            [
                family_column,
                "fold",
            ]
        ]
        .drop_duplicates(
            subset=[
                family_column
            ]
        )
        .set_index(
            family_column
        )[
            "fold"
        ]
        .to_dict()
    )

    usable_df[
        "fold"
    ] = (
        usable_df[
            family_column
        ].map(
            fold_map
        )
    )

    # ========================================================
    # CREATE FOLD DATAFRAMES
    # ========================================================

    folds = {}

    for fold_number in range(
        1,
        number_of_folds + 1,
    ):

        fold_df = (
            development.loc[
                development[
                    "fold"
                ]
                == fold_number
            ]
            .copy()
        )

        folds[
            fold_number
        ] = fold_df

    # ========================================================
    # LEAKAGE AUDIT
    # ========================================================

    print(
        "\nRunning leakage audit..."
    )

    leakage_audit = (
        run_leakage_audit(
            development=development,
            final_test=final_test,
            folds=folds,
            family_column=family_column,
            subject_column=subject_column,
            total_usable_rows=len(
                usable_df
            ),
        )
    )

    print(
        f"Leakage audit: "
        f"{leakage_audit['status']}"
    )

    if (
        leakage_audit[
            "status"
        ]
        != "PASS"
    ):
        raise RuntimeError(
            "Split leakage audit FAILED. "
            "Outputs will not be finalized."
        )

    # ========================================================
    # OUTPUT CSV FILES
    # ========================================================

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_dir = (
        split_dir
        / "folds"
    )

    folds_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_path = (
        split_dir
        / "development.csv"
    )

    final_test_path = (
        split_dir
        / "final_test.csv"
    )

    full_assignment_path = (
        split_dir
        / "usable_with_split_assignments.csv"
    )

    save_csv(
        development,
        development_path,
    )

    save_csv(
        final_test,
        final_test_path,
    )

    save_csv(
        usable_df,
        full_assignment_path,
    )

    fold_paths = {}

    for (
        fold_number,
        fold_df,
    ) in folds.items():

        path = (
            folds_dir
            / f"fold_{fold_number}.csv"
        )

        save_csv(
            fold_df,
            path,
        )

        fold_paths[
            fold_number
        ] = path

    # --------------------------------------------------------
    # Family assignment table
    # --------------------------------------------------------

    family_assignment = (
        usable_df[
            [
                family_column,
                "split",
                "fold",
            ]
        ]
        .drop_duplicates(
            subset=[
                family_column
            ]
        )
        .sort_values(
            family_column
        )
    )

    family_assignment_path = (
        split_dir
        / "family_assignments.csv"
    )

    save_csv(
        family_assignment,
        family_assignment_path,
    )

    # ========================================================
    # REPORTS
    # ========================================================

    partitions = {
        "development":
            development,

        "final_test":
            final_test,
    }

    for (
        fold_number,
        fold_df,
    ) in folds.items():
        partitions[
            f"fold_{fold_number}"
        ] = fold_df

    summary_table = (
        build_partition_summary_table(
            partitions=partitions,
            family_column=family_column,
            subject_column=subject_column,
        )
    )

    save_csv(
        summary_table,
        report_dir
        / "partition_summary.csv",
    )

    condition_distribution = (
        build_condition_distribution(
            partitions=partitions,
            condition_columns=condition_columns,
        )
    )

    save_csv(
        condition_distribution,
        report_dir
        / "condition_distribution.csv",
    )

    leakage_path = (
        report_dir
        / "leakage_audit.json"
    )

    save_json(
        leakage_audit,
        leakage_path,
    )

    split_summary = {
        "input":
            {
                "metadata_csv":
                    str(
                        metadata_path
                    ),

                "metadata_sha256":
                    sha256_file(
                        metadata_path
                    ),
            },

        "seed":
            seed,

        "configuration":
            {
                "development_ratio":
                    development_ratio,

                "final_test_ratio":
                    final_test_ratio,

                "number_of_folds":
                    number_of_folds,

                "optimization_restarts":
                    restarts,

                "local_search_iterations":
                    local_search_iterations,

                "subject_weight":
                    subject_weight,

                "image_weight":
                    image_weight,

                "condition_weight":
                    condition_weight,

                "condition_columns":
                    condition_columns,
            },

        "source":
            {
                "usable_images":
                    int(
                        len(
                            usable_df
                        )
                    ),

                "subjects":
                    int(
                        usable_df[
                            subject_column
                        ].nunique()
                    ),

                "families":
                    int(
                        usable_df[
                            family_column
                        ].nunique()
                    ),
            },

        "outer_split":
            {
                "balance_score":
                    outer_result.score,

                "family_quotas":
                    outer_result.quotas,

                "development":
                    split_statistics(
                        development,
                        family_column,
                        subject_column,
                    ),

                "final_test":
                    split_statistics(
                        final_test,
                        family_column,
                        subject_column,
                    ),
            },

        "folds":
            {
                "balance_score":
                    fold_result.score,

                "family_quotas":
                    fold_result.quotas,

                **{
                    f"fold_{fold_number}":
                        split_statistics(
                            fold_df,
                            family_column,
                            subject_column,
                        )

                    for (
                        fold_number,
                        fold_df,
                    )
                    in folds.items()
                },
            },

        "leakage_audit":
            leakage_audit,

        "outputs":
            {
                "development":
                    str(
                        development_path
                    ),

                "final_test":
                    str(
                        final_test_path
                    ),

                "family_assignments":
                    str(
                        family_assignment_path
                    ),

                "usable_with_assignments":
                    str(
                        full_assignment_path
                    ),

                "folds":
                    {
                        str(
                            key
                        ):
                            str(
                                value
                            )

                        for (
                            key,
                            value,
                        )
                        in fold_paths.items()
                    },
            },
    }

    summary_json_path = (
        report_dir
        / "split_summary.json"
    )

    save_json(
        split_summary,
        summary_json_path,
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "ND-TWINS FAMILY-DISJOINT SPLIT COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Usable images: "
        f"{len(usable_df):,}"
    )

    print(
        f"Families: "
        f"{usable_df[family_column].nunique():,}"
    )

    print(
        f"Subjects: "
        f"{usable_df[subject_column].nunique():,}"
    )

    print(
        "\nOUTER SPLIT"
    )

    dev_stats = split_statistics(
        development,
        family_column,
        subject_column,
    )

    test_stats = split_statistics(
        final_test,
        family_column,
        subject_column,
    )

    print(
        f"Development: "
        f"{dev_stats['images']:,} images | "
        f"{dev_stats['subjects']:,} subjects | "
        f"{dev_stats['families']:,} families"
    )

    print(
        f"Final test:  "
        f"{test_stats['images']:,} images | "
        f"{test_stats['subjects']:,} subjects | "
        f"{test_stats['families']:,} families"
    )

    print(
        "\nDEVELOPMENT FOLDS"
    )

    for (
        fold_number,
        fold_df,
    ) in folds.items():

        stats = split_statistics(
            fold_df,
            family_column,
            subject_column,
        )

        print(
            f"Fold {fold_number}: "
            f"{stats['images']:,} images | "
            f"{stats['subjects']:,} subjects | "
            f"{stats['families']:,} families"
        )

    print(
        "\nBALANCE"
    )

    print(
        f"Outer score: "
        f"{outer_result.score:.6f}"
    )

    print(
        f"Fold score:  "
        f"{fold_result.score:.6f}"
    )

    print(
        "\nLEAKAGE AUDIT"
    )

    print(
        f"Family overlap "
        f"development/final-test: "
        f"{leakage_audit['development_final_test']['family_overlap_count']}"
    )

    print(
        f"Subject overlap "
        f"development/final-test: "
        f"{leakage_audit['development_final_test']['subject_overlap_count']}"
    )

    print(
        f"All usable images assigned: "
        f"{leakage_audit['assignment']['all_usable_rows_assigned']}"
    )

    print(
        f"Status: "
        f"{leakage_audit['status']}"
    )

    print(
        "\nOutputs:"
    )

    print(
        split_dir
    )

    print(
        "\nReports:"
    )

    print(
        report_dir
    )


if __name__ == "__main__":
    main()
