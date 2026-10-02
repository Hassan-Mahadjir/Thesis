from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from facenet_pytorch import MTCNN
from tqdm import tqdm

from src.data.image_qc import read_image_rgb


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


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    """
    Return the first matching column.

    Matching is case-insensitive.
    """

    lookup = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        match = lookup.get(
            candidate.lower()
        )

        if match is not None:
            return match

    return None


def resolve_device(
    value: str,
) -> torch.device:
    value = str(
        value
    ).strip().lower()

    if value == "auto":
        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    return torch.device(
        value
    )


def create_detector(
    qc_config: dict,
    device: torch.device,
) -> MTCNN:
    detector_config = qc_config[
        "detector"
    ]

    return MTCNN(
        keep_all=True,
        device=device,
        min_face_size=int(
            detector_config.get(
                "min_face_size",
                40,
            )
        ),
        thresholds=list(
            detector_config.get(
                "thresholds",
                [
                    0.6,
                    0.7,
                    0.7,
                ],
            )
        ),
        factor=float(
            detector_config.get(
                "factor",
                0.709,
            )
        ),
        post_process=False,
    )


def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def condition_report(
    df: pd.DataFrame,
    failure_mask: pd.Series,
    column: str,
) -> pd.DataFrame:
    """
    Report no-face failures for a metadata condition.

    Missing values in the failure mask are treated as False.
    """

    # pandas StringDtype comparisons may produce pd.NA rather
    # than False when exclusion_reason itself is missing.
    normalized_failure_mask = (
        failure_mask
        .reindex(
            df.index
        )
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    working = df[
        [column]
    ].copy()

    working[
        "_failure"
    ] = normalized_failure_mask.astype(
        int
    )

    working[
        column
    ] = (
        working[
            column
        ]
        .astype(
            "string"
        )
        .fillna(
            "<missing>"
        )
    )

    result = (
        working
        .groupby(
            column,
            dropna=False,
        )
        .agg(
            total_images=(
                "_failure",
                "size",
            ),
            no_face_count=(
                "_failure",
                "sum",
            ),
        )
        .reset_index()
    )

    result[
        "no_face_rate_percent"
    ] = (
        result[
            "no_face_count"
        ]
        / result[
            "total_images"
        ]
        * 100.0
    )

    total_failures = int(
        normalized_failure_mask.sum()
    )

    if total_failures > 0:
        result[
            "share_of_all_no_face_percent"
        ] = (
            result[
                "no_face_count"
            ]
            / total_failures
            * 100.0
        )
    else:
        result[
            "share_of_all_no_face_percent"
        ] = 0.0

    result.insert(
        0,
        "condition",
        column,
    )

    result = result.rename(
        columns={
            column: "value",
        }
    )

    return result.sort_values(
        [
            "no_face_rate_percent",
            "no_face_count",
        ],
        ascending=[
            False,
            False,
        ],
    )


def box_area(
    box: np.ndarray,
) -> float:
    x1, y1, x2, y2 = box

    width = max(
        0.0,
        float(
            x2 - x1
        ),
    )

    height = max(
        0.0,
        float(
            y2 - y1
        ),
    )

    return width * height


def box_iou(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    x1 = max(
        ax1,
        bx1,
    )

    y1 = max(
        ay1,
        by1,
    )

    x2 = min(
        ax2,
        bx2,
    )

    y2 = min(
        ay2,
        by2,
    )

    intersection_width = max(
        0.0,
        x2 - x1,
    )

    intersection_height = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    union = (
        box_area(
            first
        )
        + box_area(
            second
        )
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection
        / union
    )


def valid_detections(
    boxes,
    probabilities,
    minimum_confidence: float,
) -> list[dict]:
    detections = []

    if (
        boxes is None
        or probabilities is None
    ):
        return detections

    for box, probability in zip(
        boxes,
        probabilities,
    ):
        if (
            box is None
            or probability is None
        ):
            continue

        probability = float(
            probability
        )

        if probability < minimum_confidence:
            continue

        box_array = np.asarray(
            box,
            dtype=np.float32,
        )

        detections.append(
            {
                "box": box_array,
                "probability": probability,
                "area": box_area(
                    box_array
                ),
            }
        )

    return detections


def select_primary_detection(
    detections: list[dict],
) -> Optional[int]:
    """
    Match the QC logic:
    largest valid face, with confidence as tie-breaker.
    """

    if not detections:
        return None

    return max(
        range(
            len(
                detections
            )
        ),
        key=lambda index: (
            detections[
                index
            ][
                "area"
            ],
            detections[
                index
            ][
                "probability"
            ],
        ),
    )


def draw_detections(
    image_rgb: np.ndarray,
    detections: list[dict],
    primary_index: int,
) -> np.ndarray:
    image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    for index, detection in enumerate(
        detections
    ):
        box = detection[
            "box"
        ]

        confidence = detection[
            "probability"
        ]

        x1, y1, x2, y2 = (
            box.astype(
                int
            )
        )

        if index == primary_index:
            color = (
                0,
                255,
                0,
            )

            thickness = 4

            label = (
                f"PRIMARY "
                f"{confidence:.4f}"
            )

        else:
            color = (
                0,
                0,
                255,
            )

            thickness = 2

            label = (
                f"candidate "
                f"{confidence:.4f}"
            )

        cv2.rectangle(
            image,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            color,
            thickness,
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return image


def audit_multiple_faces(
    df: pd.DataFrame,
    qc_config: dict,
    audit_config: dict,
    report_dir: Path,
    image_path_column: str,
    id_column: Optional[str],
) -> pd.DataFrame:
    count_column = find_column(
        df,
        [
            "number_of_faces_detected",
        ],
    )

    if count_column is None:
        raise ValueError(
            "number_of_faces_detected "
            "column was not found."
        )

    counts = safe_numeric(
        df[
            count_column
        ]
    )

    multiple = df.loc[
        counts > 1
    ].copy()

    sample_size = audit_config[
        "multiple_faces"
    ].get(
        "sample_size",
        100,
    )

    random_seed = int(
        audit_config[
            "multiple_faces"
        ].get(
            "random_seed",
            42,
        )
    )

    if sample_size is None:
        sample = multiple.copy()
    else:
        sample_size = min(
            int(
                sample_size
            ),
            len(
                multiple
            ),
        )

        sample = multiple.sample(
            n=sample_size,
            random_state=random_seed,
        )

    processing_config = qc_config[
        "processing"
    ]

    device = resolve_device(
        processing_config.get(
            "device",
            "auto",
        )
    )

    print(
        f"\nMultiple-face images: "
        f"{len(multiple):,}"
    )

    print(
        f"Auditing sample:      "
        f"{len(sample):,}"
    )

    print(
        f"Device:                "
        f"{device}"
    )

    detector = create_detector(
        qc_config=qc_config,
        device=device,
    )

    minimum_confidence = float(
        qc_config[
            "detector"
        ].get(
            "min_detection_confidence",
            0.90,
        )
    )

    save_visualizations = bool(
        audit_config[
            "multiple_faces"
        ].get(
            "save_visualizations",
            True,
        )
    )

    visual_dir = (
        report_dir
        / "multiple_face_visualizations"
    )

    if save_visualizations:
        visual_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    records = []

    for dataframe_index, row in tqdm(
        sample.iterrows(),
        total=len(
            sample
        ),
        desc="Multiple-face audit",
        unit="image",
    ):
        image_path = Path(
            str(
                row[
                    image_path_column
                ]
            )
        )

        if not image_path.is_absolute():
            image_path = resolve_path(
                image_path
            )

        record = {
            "dataframe_index":
                dataframe_index,

            "image_path":
                str(
                    image_path
                ),

            "original_qc_face_count":
                row.get(
                    count_column,
                    pd.NA,
                ),

            "audit_status":
                "",

            "audit_detection_count":
                0,

            "primary_confidence":
                pd.NA,

            "primary_area":
                pd.NA,

            "maximum_secondary_iou":
                pd.NA,

            "automatic_interpretation":
                "",

            # Intentionally left empty.
            # You can fill these manually if required.
            "manual_primary_correct":
                "",

            "manual_notes":
                "",
        }

        if id_column is not None:
            record[
                id_column
            ] = row.get(
                id_column,
                pd.NA,
            )

        try:
            image_rgb = read_image_rgb(
                image_path
            )

            if image_rgb is None:
                raise RuntimeError(
                    "read_image_rgb returned None."
                )

            boxes, probabilities = (
                detector.detect(
                    image_rgb,
                    landmarks=False,
                )
            )

            detections = valid_detections(
                boxes=boxes,
                probabilities=probabilities,
                minimum_confidence=(
                    minimum_confidence
                ),
            )

            record[
                "audit_detection_count"
            ] = len(
                detections
            )

            if len(
                detections
            ) == 0:
                record[
                    "audit_status"
                ] = "no_valid_detection"

                record[
                    "automatic_interpretation"
                ] = (
                    "detection_changed_on_audit"
                )

                records.append(
                    record
                )

                continue

            primary_index = (
                select_primary_detection(
                    detections
                )
            )

            primary = detections[
                primary_index
            ]

            record[
                "primary_confidence"
            ] = primary[
                "probability"
            ]

            record[
                "primary_area"
            ] = primary[
                "area"
            ]

            secondary_ious = []

            for index, detection in enumerate(
                detections
            ):
                if index == primary_index:
                    continue

                secondary_ious.append(
                    box_iou(
                        primary[
                            "box"
                        ],
                        detection[
                            "box"
                        ],
                    )
                )

            maximum_secondary_iou = (
                max(
                    secondary_ious
                )
                if secondary_ious
                else 0.0
            )

            record[
                "maximum_secondary_iou"
            ] = (
                maximum_secondary_iou
            )

            if len(
                detections
            ) == 1:
                interpretation = (
                    "single_detection_on_audit"
                )

            elif maximum_secondary_iou >= 0.50:
                interpretation = (
                    "overlapping_detection_candidate"
                )

            else:
                interpretation = (
                    "separate_face_candidate"
                )

            record[
                "automatic_interpretation"
            ] = interpretation

            record[
                "audit_status"
            ] = "success"

            if save_visualizations:
                rendered = draw_detections(
                    image_rgb=image_rgb,
                    detections=detections,
                    primary_index=primary_index,
                )

                if id_column is not None:
                    image_id = str(
                        row.get(
                            id_column,
                            dataframe_index,
                        )
                    )
                else:
                    image_id = str(
                        dataframe_index
                    )

                output_path = (
                    visual_dir
                    / f"{image_id}.jpg"
                )

                cv2.imwrite(
                    str(
                        output_path
                    ),
                    rendered,
                )

        except Exception as error:
            record[
                "audit_status"
            ] = "error"

            record[
                "automatic_interpretation"
            ] = (
                type(
                    error
                ).__name__
            )

            record[
                "manual_notes"
            ] = str(
                error
            )[:500]

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


def build_summary(
    df: pd.DataFrame,
) -> dict:
    processed = (
        df[
            "qc_processed"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    readable = (
        df[
            "image_readable"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    face_detected = (
        df[
            "face_detected"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    alignment_success = (
        df[
            "alignment_success"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    usable = (
        df[
            "usable"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    face_counts = safe_numeric(
        df[
            "number_of_faces_detected"
        ]
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

    return {
        "total_images":
            int(
                len(
                    df
                )
            ),

        "processed":
            int(
                processed.sum()
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ND-TWINS QC failures "
            "and multiple-face detections."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to qc_audit.yaml",
    )

    args = parser.parse_args()

    audit_config_path = resolve_path(
        args.config
    )

    audit_config = load_yaml(
        audit_config_path
    )

    metadata_path = resolve_path(
        audit_config[
            "input"
        ][
            "metadata_csv"
        ]
    )

    qc_config_path = resolve_path(
        audit_config[
            "input"
        ][
            "qc_config"
        ]
    )

    report_dir = resolve_path(
        audit_config[
            "output"
        ][
            "report_dir"
        ]
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading QC metadata:\n"
        f"{metadata_path}"
    )

    df = pd.read_csv(
        metadata_path,
        low_memory=False,
    )

    qc_config = load_yaml(
        qc_config_path
    )

    image_path_column = find_column(
        df,
        [
            "image_path",
            "filepath",
            "file_path",
            "path",
        ],
    )

    if image_path_column is None:
        raise ValueError(
            "Could not locate the image path column."
        )

    id_column = find_column(
        df,
        [
            "image_id",
            "imageid",
            "id",
            "sequenceid",
        ],
    )

    summary = build_summary(
        df
    )

    # ========================================================
    # FAILURE AUDIT
    # ========================================================

    exclusion_reason = (
        df[
            "exclusion_reason"
        ]
        .astype(
            "string"
        )
        .fillna(
            ""
        )
    )

    unreadable_mask = (
        exclusion_reason
        .eq(
            "unreadable_image"
        )
        .fillna(
            False
        )
    )

    no_face_mask = (
        exclusion_reason
        .eq(
            "no_face_detected"
        )
        .fillna(
            False
        )
    )

    failed_mask = (
        unreadable_mask
        | no_face_mask
    )

    failure_audit = df.loc[
        failed_mask
    ].copy()

    failure_audit.to_csv(
        report_dir
        / "qc_failure_audit.csv",
        index=False,
    )

    df.loc[
        unreadable_mask
    ].to_csv(
        report_dir
        / "unreadable_images.csv",
        index=False,
    )

    df.loc[
        no_face_mask
    ].to_csv(
        report_dir
        / "no_face_images.csv",
        index=False,
    )

    # ========================================================
    # CONDITION AUDIT
    # ========================================================

    candidate_conditions = [
        "yaw",
        "pose",
        "pose_category",
        "emotion",
        "expression",
        "glasses",
        "weather",
        "illumination",
        "illuminantid",
        "illuminantid1",
        "illuminantid2",
        "environmentid",
        "camera",
        "sensorid",
        "stageid",
        "collectionid",
    ]

    condition_frames = []

    for candidate in candidate_conditions:
        column = find_column(
            df,
            [
                candidate,
            ],
        )

        if column is None:
            continue

        report = condition_report(
            df=df,
            failure_mask=no_face_mask,
            column=column,
        )

        condition_frames.append(
            report
        )

        if candidate == "yaw":
            report.to_csv(
                report_dir
                / "no_face_by_yaw.csv",
                index=False,
            )

    if condition_frames:
        condition_report_all = pd.concat(
            condition_frames,
            ignore_index=True,
        )

        condition_report_all.to_csv(
            report_dir
            / "no_face_by_condition.csv",
            index=False,
        )

    # ========================================================
    # MULTIPLE-FACE AUDIT
    # ========================================================

    multiple_face_audit = (
        audit_multiple_faces(
            df=df,
            qc_config=qc_config,
            audit_config=audit_config,
            report_dir=report_dir,
            image_path_column=(
                image_path_column
            ),
            id_column=id_column,
        )
    )

    multiple_face_audit.to_csv(
        report_dir
        / "multiple_face_audit.csv",
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary[
        "failure_audit"
    ] = {
        "unreadable_images":
            int(
                unreadable_mask.sum()
            ),

        "no_face_images":
            int(
                no_face_mask.sum()
            ),

        "multiple_face_sample_size":
            int(
                len(
                    multiple_face_audit
                )
            ),
    }

    summary_path = (
        report_dir
        / "qc_final_audit.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\n========================================"
    )

    print(
        "ND-TWINS QC AUDIT COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Total images:          "
        f"{summary['total_images']:,}"
    )

    print(
        f"Usable:                "
        f"{summary['usable']:,}"
    )

    print(
        f"Unreadable:            "
        f"{summary['unreadable']:,}"
    )

    print(
        f"No face detected:      "
        f"{summary['no_face_detected']:,}"
    )

    print(
        f"Multiple-face images:  "
        f"{summary['multiple_faces']:,}"
    )

    print(
        "\nReports:"
    )

    print(
        report_dir
    )


if __name__ == "__main__":
    main()
