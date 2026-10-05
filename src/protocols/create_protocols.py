from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# ============================================================
# CONFIG / PATHS
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
# DATA HELPERS
# ============================================================


def normalize_bool(
    series: pd.Series,
) -> pd.Series:
    """
    Convert common boolean representations into Python bool.
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

    return (
        normalized
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
                "t",
            }
        )
        .fillna(False)
        .astype(bool)
    )


def normalize_identifier(
    series: pd.Series,
) -> pd.Series:
    result = (
        series
        .astype("string")
        .str.strip()
    )

    return result.str.replace(
        r"\.0$",
        "",
        regex=True,
    )


def ensure_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


# ============================================================
# PROTOCOL MASKS
# ============================================================


def create_protocol_mask(
    df: pd.DataFrame,
    protocol_config: dict,
    yaw_column: str,
) -> pd.Series:

    protocol_type = str(
        protocol_config[
            "type"
        ]
    ).strip().lower()

    # --------------------------------------------------------
    # ALL USABLE
    # --------------------------------------------------------

    if protocol_type == "all":
        return pd.Series(
            True,
            index=df.index,
            dtype=bool,
        )

    # --------------------------------------------------------
    # BOOLEAN ANY
    #
    # Row is included if ANY configured boolean column is true.
    # --------------------------------------------------------

    if protocol_type == "boolean_any":

        columns = protocol_config.get(
            "columns",
            [],
        )

        if not columns:
            raise ValueError(
                "boolean_any protocol requires at least "
                "one column."
            )

        ensure_columns(
            df,
            columns,
        )

        mask = pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

        for column in columns:
            mask = (
                mask
                | normalize_bool(
                    df[
                        column
                    ]
                )
            )

        return mask

    # --------------------------------------------------------
    # YAW VALUES
    # --------------------------------------------------------

    if protocol_type == "yaw_values":

        ensure_columns(
            df,
            [
                yaw_column,
            ],
        )

        allowed_values = {
            float(
                value
            )
            for value
            in protocol_config.get(
                "values",
                [],
            )
        }

        yaw = pd.to_numeric(
            df[
                yaw_column
            ],
            errors="coerce",
        )

        return (
            yaw
            .isin(
                allowed_values
            )
            .fillna(False)
        )

    raise ValueError(
        f"Unknown protocol type: "
        f"{protocol_type}"
    )


# ============================================================
# VALIDATE FROZEN SPLIT
# ============================================================


def validate_assignments(
    df: pd.DataFrame,
    image_column: str,
    subject_column: str,
    family_column: str,
    split_column: str,
) -> None:

    # --------------------------------------------------------
    # Image uniqueness
    # --------------------------------------------------------

    if df[
        image_column
    ].duplicated().any():

        duplicates = (
            df.loc[
                df[
                    image_column
                ].duplicated(
                    keep=False
                ),
                image_column,
            ]
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Duplicate image IDs found:\n"
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Expected outer partitions
    # --------------------------------------------------------

    split_values = set(
        df[
            split_column
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    expected = {
        "development",
        "final_test",
    }

    if split_values != expected:

        raise ValueError(
            "Unexpected split values.\n"
            f"Expected: {expected}\n"
            f"Found:    {split_values}"
        )

    development = df.loc[
        df[
            split_column
        ]
        == "development"
    ]

    final_test = df.loc[
        df[
            split_column
        ]
        == "final_test"
    ]

    # --------------------------------------------------------
    # Family leakage
    # --------------------------------------------------------

    development_families = set(
        development[
            family_column
        ].astype(str)
    )

    final_test_families = set(
        final_test[
            family_column
        ].astype(str)
    )

    family_overlap = (
        development_families
        & final_test_families
    )

    if family_overlap:

        raise ValueError(
            "Family leakage detected between "
            "development and final_test:\n"
            f"{sorted(family_overlap)}"
        )

    # --------------------------------------------------------
    # Subject leakage
    # --------------------------------------------------------

    development_subjects = set(
        development[
            subject_column
        ].astype(str)
    )

    final_test_subjects = set(
        final_test[
            subject_column
        ].astype(str)
    )

    subject_overlap = (
        development_subjects
        & final_test_subjects
    )

    if subject_overlap:

        raise ValueError(
            "Subject leakage detected between "
            "development and final_test:\n"
            f"{sorted(subject_overlap)}"
        )


# ============================================================
# PROTOCOL STATISTICS
# ============================================================


def partition_statistics(
    df: pd.DataFrame,
    subject_column: str,
    family_column: str,
) -> dict:

    subject_counts = (
        df.groupby(
            subject_column
        )
        .size()
    )

    subjects_with_two_or_more = int(
        (
            subject_counts >= 2
        ).sum()
    )

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

        "subjects_with_2plus_images":
            subjects_with_two_or_more,
    }


def fold_statistics(
    development: pd.DataFrame,
    fold_column: str,
    subject_column: str,
    family_column: str,
) -> dict:

    result = {}

    fold_values = (
        pd.to_numeric(
            development[
                fold_column
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
    )

    for fold_number in fold_values:

        fold_df = development.loc[
            pd.to_numeric(
                development[
                    fold_column
                ],
                errors="coerce",
            )
            == fold_number
        ]

        result[
            f"fold_{fold_number}"
        ] = partition_statistics(
            df=fold_df,
            subject_column=subject_column,
            family_column=family_column,
        )

    return result


# ============================================================
# CONDITION DISTRIBUTION
# ============================================================


def build_condition_distribution(
    protocol_name: str,
    protocol_df: pd.DataFrame,
    split_column: str,
    condition_columns: list[str],
) -> list[dict]:

    records = []

    partitions = {
        "all":
            protocol_df,

        "development":
            protocol_df.loc[
                protocol_df[
                    split_column
                ]
                == "development"
            ],

        "final_test":
            protocol_df.loc[
                protocol_df[
                    split_column
                ]
                == "final_test"
            ],
    }

    for partition_name, frame in (
        partitions.items()
    ):

        for column in condition_columns:

            if column not in frame.columns:
                continue

            values = (
                frame[
                    column
                ]
                .astype("string")
                .fillna(
                    "<missing>"
                )
            )

            counts = values.value_counts(
                dropna=False
            )

            total = len(
                frame
            )

            for value, count in (
                counts.items()
            ):

                records.append(
                    {
                        "protocol":
                            protocol_name,

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

    return records


# ============================================================
# PROTOCOL LEAKAGE CHECK
# ============================================================


def audit_protocol_leakage(
    protocol_df: pd.DataFrame,
    split_column: str,
    subject_column: str,
    family_column: str,
) -> dict:

    development = protocol_df.loc[
        protocol_df[
            split_column
        ]
        == "development"
    ]

    final_test = protocol_df.loc[
        protocol_df[
            split_column
        ]
        == "final_test"
    ]

    family_overlap = (
        set(
            development[
                family_column
            ].astype(str)
        )
        & set(
            final_test[
                family_column
            ].astype(str)
        )
    )

    subject_overlap = (
        set(
            development[
                subject_column
            ].astype(str)
        )
        & set(
            final_test[
                subject_column
            ].astype(str)
        )
    )

    passed = (
        len(
            family_overlap
        )
        == 0
        and len(
            subject_overlap
        )
        == 0
    )

    return {
        "status":
            (
                "PASS"
                if passed
                else "FAIL"
            ),

        "family_overlap_count":
            len(
                family_overlap
            ),

        "subject_overlap_count":
            len(
                subject_overlap
            ),

        "family_overlap":
            sorted(
                family_overlap
            ),

        "subject_overlap":
            sorted(
                subject_overlap
            ),
    }


# ============================================================
# SAVE HELPERS
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
            "Create fixed ND-TWINS experimental "
            "protocol manifests."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to protocols YAML configuration."
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
    # Paths
    # --------------------------------------------------------

    assignments_path = resolve_path(
        config[
            "input"
        ][
            "assignments_csv"
        ]
    )

    protocol_dir = resolve_path(
        config[
            "output"
        ][
            "protocol_dir"
        ]
    )

    report_dir = resolve_path(
        config[
            "output"
        ][
            "report_dir"
        ]
    )

    protocol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Column configuration
    # --------------------------------------------------------

    columns = config[
        "columns"
    ]

    image_column = columns[
        "image_id"
    ]

    subject_column = columns[
        "subject_id"
    ]

    family_column = columns[
        "family_id"
    ]

    split_column = columns[
        "split"
    ]

    fold_column = columns[
        "fold"
    ]

    usable_column = columns[
        "usable"
    ]

    yaw_column = columns[
        "yaw"
    ]

    # --------------------------------------------------------
    # Load assignments
    # --------------------------------------------------------

    print(
        "Loading frozen split assignments:"
    )

    print(
        assignments_path
    )

    df = pd.read_csv(
        assignments_path,
        low_memory=False,
    )

    ensure_columns(
        df,
        [
            image_column,
            subject_column,
            family_column,
            split_column,
            fold_column,
            usable_column,
            yaw_column,
        ],
    )

    # --------------------------------------------------------
    # Only usable images should exist here, but enforce it.
    # --------------------------------------------------------

    usable_mask = normalize_bool(
        df[
            usable_column
        ]
    )

    if not usable_mask.all():

        excluded_count = int(
            (
                ~usable_mask
            ).sum()
        )

        print(
            f"[WARNING] Removing "
            f"{excluded_count:,} non-usable rows."
        )

        df = (
            df.loc[
                usable_mask
            ]
            .copy()
        )

    # --------------------------------------------------------
    # Normalize identifiers
    # --------------------------------------------------------

    df[
        image_column
    ] = normalize_identifier(
        df[
            image_column
        ]
    )

    df[
        subject_column
    ] = normalize_identifier(
        df[
            subject_column
        ]
    )

    df[
        family_column
    ] = normalize_identifier(
        df[
            family_column
        ]
    )

    df[
        split_column
    ] = (
        df[
            split_column
        ]
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Validate frozen split before creating any protocol.
    # --------------------------------------------------------

    validate_assignments(
        df=df,
        image_column=image_column,
        subject_column=subject_column,
        family_column=family_column,
        split_column=split_column,
    )

    print(
        "\nFrozen split validation: PASS"
    )

    print(
        f"Images:   {len(df):,}"
    )

    print(
        f"Subjects: "
        f"{df[subject_column].nunique():,}"
    )

    print(
        f"Families: "
        f"{df[family_column].nunique():,}"
    )

    protocols_config = config[
        "protocols"
    ]

    condition_columns = list(
        config.get(
            "report",
            {},
        ).get(
            "condition_columns",
            [],
        )
    )

    summary = {
        "input":
            {
                "assignments_csv":
                    str(
                        assignments_path
                    ),

                "assignments_sha256":
                    sha256_file(
                        assignments_path
                    ),
            },

        "source":
            {
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
            },

        "protocols":
            {},
    }

    condition_records = []

    # ========================================================
    # CREATE EACH PROTOCOL
    # ========================================================

    for (
        protocol_name,
        protocol_config,
    ) in protocols_config.items():

        print(
            "\n----------------------------------------"
        )

        print(
            protocol_name
        )

        print(
            "----------------------------------------"
        )

        mask = create_protocol_mask(
            df=df,
            protocol_config=(
                protocol_config
            ),
            yaw_column=yaw_column,
        )

        protocol_df = (
            df.loc[
                mask
            ]
            .copy()
        )

        protocol_df.insert(
            0,
            "protocol",
            protocol_name,
        )

        # ----------------------------------------------------
        # Stable deterministic ordering.
        # ----------------------------------------------------

        sort_columns = [
            split_column,
            family_column,
            subject_column,
            image_column,
        ]

        protocol_df = (
            protocol_df
            .sort_values(
                sort_columns,
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Split partitions
        # ----------------------------------------------------

        development = (
            protocol_df.loc[
                protocol_df[
                    split_column
                ]
                == "development"
            ]
            .copy()
        )

        final_test = (
            protocol_df.loc[
                protocol_df[
                    split_column
                ]
                == "final_test"
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Verify no family/subject leakage was introduced.
        # ----------------------------------------------------

        leakage = (
            audit_protocol_leakage(
                protocol_df=(
                    protocol_df
                ),
                split_column=(
                    split_column
                ),
                subject_column=(
                    subject_column
                ),
                family_column=(
                    family_column
                ),
            )
        )

        if (
            leakage[
                "status"
            ]
            != "PASS"
        ):
            raise RuntimeError(
                f"Protocol {protocol_name} "
                f"failed leakage validation."
            )

        # ----------------------------------------------------
        # Save manifest
        # ----------------------------------------------------

        output_path = (
            protocol_dir
            / f"{protocol_name}.csv"
        )

        save_csv(
            protocol_df,
            output_path,
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        all_stats = (
            partition_statistics(
                df=protocol_df,
                subject_column=(
                    subject_column
                ),
                family_column=(
                    family_column
                ),
            )
        )

        development_stats = (
            partition_statistics(
                df=development,
                subject_column=(
                    subject_column
                ),
                family_column=(
                    family_column
                ),
            )
        )

        final_test_stats = (
            partition_statistics(
                df=final_test,
                subject_column=(
                    subject_column
                ),
                family_column=(
                    family_column
                ),
            )
        )

        folds = fold_statistics(
            development=development,
            fold_column=fold_column,
            subject_column=subject_column,
            family_column=family_column,
        )

        summary[
            "protocols"
        ][
            protocol_name
        ] = {
            "description":
                protocol_config.get(
                    "description",
                    "",
                ),

            "manifest":
                str(
                    output_path
                ),

            "manifest_sha256":
                sha256_file(
                    output_path
                ),

            "all":
                all_stats,

            "development":
                development_stats,

            "final_test":
                final_test_stats,

            "folds":
                folds,

            "leakage_audit":
                leakage,
        }

        condition_records.extend(
            build_condition_distribution(
                protocol_name=(
                    protocol_name
                ),
                protocol_df=(
                    protocol_df
                ),
                split_column=(
                    split_column
                ),
                condition_columns=(
                    condition_columns
                ),
            )
        )

        print(
            f"All:         "
            f"{all_stats['images']:,} images | "
            f"{all_stats['subjects']:,} subjects | "
            f"{all_stats['families']:,} families"
        )

        print(
            f"Development: "
            f"{development_stats['images']:,} images | "
            f"{development_stats['subjects']:,} subjects | "
            f"{development_stats['families']:,} families"
        )

        print(
            f"Final test:  "
            f"{final_test_stats['images']:,} images | "
            f"{final_test_stats['subjects']:,} subjects | "
            f"{final_test_stats['families']:,} families"
        )

        print(
            f"Subjects with >=2 images: "
            f"{all_stats['subjects_with_2plus_images']:,}"
        )

        print(
            f"Leakage: "
            f"{leakage['status']}"
        )

    # ========================================================
    # REPORTS
    # ========================================================

    summary_path = (
        report_dir
        / "protocol_summary.json"
    )

    save_json(
        summary,
        summary_path,
    )

    condition_df = pd.DataFrame(
        condition_records
    )

    condition_path = (
        report_dir
        / "protocol_condition_distribution.csv"
    )

    save_csv(
        condition_df,
        condition_path,
    )

    # --------------------------------------------------------
    # Compact summary CSV
    # --------------------------------------------------------

    summary_records = []

    for (
        protocol_name,
        protocol_summary,
    ) in summary[
        "protocols"
    ].items():

        for partition in [
            "all",
            "development",
            "final_test",
        ]:

            stats = protocol_summary[
                partition
            ]

            summary_records.append(
                {
                    "protocol":
                        protocol_name,

                    "partition":
                        partition,

                    "images":
                        stats[
                            "images"
                        ],

                    "subjects":
                        stats[
                            "subjects"
                        ],

                    "families":
                        stats[
                            "families"
                        ],

                    "subjects_with_2plus_images":
                        stats[
                            "subjects_with_2plus_images"
                        ],
                }
            )

    summary_csv = pd.DataFrame(
        summary_records
    )

    save_csv(
        summary_csv,
        report_dir
        / "protocol_summary.csv",
    )

    print(
        "\n========================================"
    )

    print(
        "EXPERIMENTAL PROTOCOLS COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Protocol manifests:\n"
        f"{protocol_dir}"
    )

    print(
        f"\nReports:\n"
        f"{report_dir}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()