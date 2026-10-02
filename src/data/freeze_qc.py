from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


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
    sha256 = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


def count_files(
    directory: Path,
    suffix: str = ".png",
) -> int:
    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.rglob(
            f"*{suffix}"
        )
        if path.is_file()
    )


def bool_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    return (
        df[
            column
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )


def build_counts(
    df: pd.DataFrame,
) -> dict:
    readable = bool_series(
        df,
        "image_readable",
    )

    face_detected = bool_series(
        df,
        "face_detected",
    )

    landmarks_detected = bool_series(
        df,
        "landmarks_detected",
    )

    alignment_success = bool_series(
        df,
        "alignment_success",
    )

    usable = bool_series(
        df,
        "usable",
    )

    face_counts = pd.to_numeric(
        df[
            "number_of_faces_detected"
        ],
        errors="coerce",
    )

    reasons = (
        df[
            "exclusion_reason"
        ]
        .dropna()
        .astype(
            str
        )
        .value_counts()
        .to_dict()
    )

    attempts = pd.to_numeric(
        df.get(
            "qc_attempts",
            pd.Series(
                index=df.index,
                dtype="float64",
            ),
        ),
        errors="coerce",
    )

    return {
        "total_images":
            int(
                len(
                    df
                )
            ),

        "readable":
            int(
                readable.sum()
            ),

        "unreadable":
            int(
                (
                    ~readable
                ).sum()
            ),

        "face_detected":
            int(
                face_detected.sum()
            ),

        "no_face_detected":
            int(
                (
                    readable
                    & ~face_detected
                ).sum()
            ),

        "landmarks_detected":
            int(
                landmarks_detected.sum()
            ),

        "alignment_success":
            int(
                alignment_success.sum()
            ),

        "multiple_faces":
            int(
                (
                    face_counts > 1
                ).sum()
            ),

        "usable":
            int(
                usable.sum()
            ),

        "unusable":
            int(
                (
                    ~usable
                ).sum()
            ),

        "retried_images":
            int(
                (
                    attempts > 1
                ).sum()
            ),

        "exclusion_reasons":
            {
                str(
                    key
                ): int(
                    value
                )
                for key, value
                in reasons.items()
            },
    }


def validate_counts(
    counts: dict,
    expected: dict,
) -> None:
    """
    Prevent freezing if the dataset no longer matches the
    audited values.
    """

    for key, expected_value in expected.items():
        actual_value = counts.get(
            key
        )

        if actual_value != expected_value:
            raise ValueError(
                f"QC validation failed for '{key}'. "
                f"Expected {expected_value:,}, "
                f"found {actual_value:,}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the audited ND-TWINS "
            "QC dataset as an immutable version."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to QC freeze YAML config."
        ),
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    config = load_yaml(
        config_path
    )

    version = str(
        config[
            "freeze"
        ][
            "version"
        ]
    ).strip()

    source_metadata = resolve_path(
        config[
            "input"
        ][
            "metadata_csv"
        ]
    )

    qc_config_path = resolve_path(
        config[
            "input"
        ][
            "qc_config"
        ]
    )

    audit_json_path = resolve_path(
        config[
            "input"
        ][
            "audit_json"
        ]
    )

    frozen_metadata = resolve_path(
        config[
            "output"
        ][
            "metadata_csv"
        ]
    )

    freeze_report = resolve_path(
        config[
            "output"
        ][
            "freeze_report_json"
        ]
    )

    frozen_qc_config = resolve_path(
        config[
            "output"
        ][
            "qc_config_copy"
        ]
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    required_files = [
        source_metadata,
        qc_config_path,
        audit_json_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    if frozen_metadata.exists():
        raise FileExistsError(
            f"Frozen metadata already exists:\n"
            f"{frozen_metadata}\n\n"
            "QC freezes are immutable. "
            "Use a new version instead of overwriting it."
        )

    if freeze_report.exists():
        raise FileExistsError(
            f"Freeze report already exists:\n"
            f"{freeze_report}"
        )

    # --------------------------------------------------------
    # Load current QC metadata
    # --------------------------------------------------------

    print(
        "Loading audited QC metadata:"
    )

    print(
        source_metadata
    )

    df = pd.read_csv(
        source_metadata,
        low_memory=False,
    )

    counts = build_counts(
        df
    )

    # --------------------------------------------------------
    # Validate against known audited QC v1 result
    # --------------------------------------------------------

    expected = config[
        "validation"
    ][
        "expected_counts"
    ]

    validate_counts(
        counts=counts,
        expected=expected,
    )

    print(
        "\nQC counts validated."
    )

    # --------------------------------------------------------
    # Validate audit summary as well
    # --------------------------------------------------------

    with audit_json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        audit_summary = json.load(
            file
        )

    for key in [
        "total_images",
        "readable",
        "unreadable",
        "face_detected",
        "no_face_detected",
        "alignment_success",
        "multiple_faces",
        "usable",
        "unusable",
    ]:
        if (
            audit_summary.get(
                key
            )
            != counts.get(
                key
            )
        ):
            raise ValueError(
                f"Audit summary mismatch for '{key}'. "
                f"Metadata={counts.get(key)}, "
                f"Audit={audit_summary.get(key)}"
            )

    print(
        "Audit summary validated."
    )

    # --------------------------------------------------------
    # Check aligned output counts
    # --------------------------------------------------------

    aligned_112_arcface_dir = resolve_path(
        config[
            "aligned_outputs"
        ][
            "aligned_112_arcface"
        ]
    )

    aligned_112_fullface_dir = resolve_path(
        config[
            "aligned_outputs"
        ][
            "aligned_112_fullface"
        ]
    )

    aligned_448_fullface_dir = resolve_path(
        config[
            "aligned_outputs"
        ][
            "aligned_448_fullface"
        ]
    )

    print(
        "\nCounting aligned images..."
    )

    aligned_counts = {
        "aligned_112_arcface":
            count_files(
                aligned_112_arcface_dir
            ),

        "aligned_112_fullface":
            count_files(
                aligned_112_fullface_dir
            ),

        "aligned_448_fullface":
            count_files(
                aligned_448_fullface_dir
            ),
    }

    for name, count in aligned_counts.items():
        print(
            f"{name}: {count:,}"
        )

        if count != counts[
            "usable"
        ]:
            raise ValueError(
                f"{name} contains {count:,} images, "
                f"but QC metadata reports "
                f"{counts['usable']:,} usable images."
            )

    # --------------------------------------------------------
    # Create immutable metadata snapshot
    # --------------------------------------------------------

    frozen_metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    freeze_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_qc_config.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_metadata,
        frozen_metadata,
    )

    shutil.copy2(
        qc_config_path,
        frozen_qc_config,
    )

    # --------------------------------------------------------
    # Compute hashes AFTER copies are created
    # --------------------------------------------------------

    hashes = {
        "frozen_metadata_sha256":
            sha256_file(
                frozen_metadata
            ),

        "source_metadata_sha256":
            sha256_file(
                source_metadata
            ),

        "qc_config_sha256":
            sha256_file(
                frozen_qc_config
            ),

        "audit_summary_sha256":
            sha256_file(
                audit_json_path
            ),
    }

    if (
        hashes[
            "frozen_metadata_sha256"
        ]
        != hashes[
            "source_metadata_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen metadata hash does not match "
            "the source QC metadata."
        )

    # --------------------------------------------------------
    # Freeze manifest
    # --------------------------------------------------------

    freeze_manifest = {
        "freeze_version":
            version,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "frozen",

        "dataset":
            "ND-TWINS-2009-2010",

        "counts":
            counts,

        "aligned_output_counts":
            aligned_counts,

        "audit":
            {
                "multiple_face_visual_check":
                    True,

                "multiple_face_primary_selection":
                    "confirmed_correct",

                "audit_summary":
                    str(
                        audit_json_path
                    ),
            },

        "files":
            {
                "source_qc_metadata":
                    str(
                        source_metadata
                    ),

                "frozen_qc_metadata":
                    str(
                        frozen_metadata
                    ),

                "frozen_qc_config":
                    str(
                        frozen_qc_config
                    ),

                "audit_summary":
                    str(
                        audit_json_path
                    ),
            },

        "aligned_outputs":
            {
                "aligned_112_arcface":
                    str(
                        aligned_112_arcface_dir
                    ),

                "aligned_112_fullface":
                    str(
                        aligned_112_fullface_dir
                    ),

                "aligned_448_fullface":
                    str(
                        aligned_448_fullface_dir
                    ),
            },

        "hashes":
            hashes,

        "git_commit":
            get_git_commit(),

        "notes": [
            (
                "QC v1 is immutable and must not be "
                "modified after freezing."
            ),
            (
                "Future preprocessing changes must use "
                "a new QC version."
            ),
            (
                "Full-profile preprocessing may be "
                "introduced later as a separate protocol."
            ),
        ],
    }

    with freeze_report.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            freeze_manifest,
            file,
            indent=2,
        )

    print(
        "\n========================================"
    )

    print(
        f"ND-TWINS QC {version.upper()} FROZEN"
    )

    print(
        "========================================"
    )

    print(
        f"Total images:       "
        f"{counts['total_images']:,}"
    )

    print(
        f"Usable:             "
        f"{counts['usable']:,}"
    )

    print(
        f"Unusable:           "
        f"{counts['unusable']:,}"
    )

    print(
        f"Multiple faces:     "
        f"{counts['multiple_faces']:,}"
    )

    print(
        f"\nFrozen metadata:\n"
        f"{frozen_metadata}"
    )

    print(
        f"\nFrozen QC config:\n"
        f"{frozen_qc_config}"
    )

    print(
        f"\nFreeze manifest:\n"
        f"{freeze_report}"
    )

    print(
        "\nSHA-256:"
    )

    print(
        hashes[
            "frozen_metadata_sha256"
        ]
    )


if __name__ == "__main__":
    main()