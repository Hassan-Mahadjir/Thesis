from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rawpy
import torch
import yaml
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


# ============================================================
# QC OUTPUT COLUMNS
# ============================================================

QC_COLUMNS = [
    "qc_processed",

    "image_readable",
    "image_width",
    "image_height",

    "face_detected",
    "face_detection_confidence",
    "number_of_faces_detected",

    "face_bbox_x1",
    "face_bbox_y1",
    "face_bbox_x2",
    "face_bbox_y2",
    "face_bbox_ratio",

    "landmarks_detected",

    "pred_left_eye_x",
    "pred_left_eye_y",

    "pred_right_eye_x",
    "pred_right_eye_y",

    "pred_nose_x",
    "pred_nose_y",

    "pred_mouth_left_x",
    "pred_mouth_left_y",

    "pred_mouth_right_x",
    "pred_mouth_right_y",

    "alignment_success",

    "aligned_112_arcface_path",
    "aligned_112_fullface_path",
    "aligned_448_fullface_path",

    "blur_score",
    "brightness_mean",
    "brightness_std",

    "quality_score",

    "usable",
    "exclusion_reason",
]


QC_BOOLEAN_COLUMNS = [
    "qc_processed",
    "image_readable",
    "face_detected",
    "landmarks_detected",
    "alignment_success",
    "usable",
]


QC_INTEGER_COLUMNS = [
    "image_width",
    "image_height",

    "number_of_faces_detected",

    "face_bbox_x1",
    "face_bbox_y1",
    "face_bbox_x2",
    "face_bbox_y2",
]


QC_FLOAT_COLUMNS = [
    "face_detection_confidence",
    "face_bbox_ratio",

    "pred_left_eye_x",
    "pred_left_eye_y",

    "pred_right_eye_x",
    "pred_right_eye_y",

    "pred_nose_x",
    "pred_nose_y",

    "pred_mouth_left_x",
    "pred_mouth_left_y",

    "pred_mouth_right_x",
    "pred_mouth_right_y",

    "blur_score",
    "brightness_mean",
    "brightness_std",

    "quality_score",
]


QC_STRING_COLUMNS = [
    "aligned_112_arcface_path",
    "aligned_112_fullface_path",
    "aligned_448_fullface_path",

    "exclusion_reason",
]


# ============================================================
# ARCFACE LANDMARK TEMPLATE
#
# Standard five-point ArcFace template for a 112x112 image.
# ============================================================

ARCFACE_REFERENCE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


RAW_EXTENSIONS = {
    ".nef",
    ".cr2",
    ".cr3",
    ".arw",
    ".dng",
    ".rw2",
}


# ============================================================
# PROJECT / CONFIG
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return project_root() / path


def stored_path(
    path: Path,
) -> str:
    root = project_root().resolve()
    path = path.resolve()

    try:
        return path.relative_to(
            root
        ).as_posix()

    except ValueError:
        return str(path)


def resolve_stored_path(
    value,
) -> Path | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value:
        return None

    return resolve_project_path(
        value
    )


def load_config(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            f"Invalid YAML configuration: {path}"
        )

    return config


# ============================================================
# BOOLEAN / PANDAS TYPE HELPERS
# ============================================================

def as_bool(
    value,
) -> bool:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def parse_nullable_bool(
    value,
):
    if value is None:
        return pd.NA

    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    normalized = (
        str(value)
        .strip()
        .lower()
    )

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot interpret '{value}' as boolean."
    )


def apply_qc_schema(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    # --------------------------------------------------------
    # Nullable booleans
    # --------------------------------------------------------

    for column in QC_BOOLEAN_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="boolean",
            )
        else:
            df[column] = (
                df[column]
                .map(
                    parse_nullable_bool
                )
                .astype(
                    "boolean"
                )
            )

    # --------------------------------------------------------
    # Nullable integers
    # --------------------------------------------------------

    for column in QC_INTEGER_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="Int64",
            )
        else:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .astype(
                    "Int64"
                )
            )

    # --------------------------------------------------------
    # Nullable floats
    # --------------------------------------------------------

    for column in QC_FLOAT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="Float64",
            )
        else:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .astype(
                    "Float64"
                )
            )

    # --------------------------------------------------------
    # Nullable strings
    # --------------------------------------------------------

    for column in QC_STRING_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="string",
            )
        else:
            df[column] = (
                df[column]
                .astype(
                    "string"
                )
            )

    return df


# ============================================================
# DEVICE
# ============================================================

def resolve_device(
    requested: str,
) -> torch.device:
    requested = (
        str(requested)
        .strip()
        .lower()
    )

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot "
            "access a CUDA device."
        )

    return device


# ============================================================
# IMAGE LOADING
# ============================================================

def read_image_rgb(
    path: Path,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {path}"
        )

    suffix = (
        path.suffix
        .lower()
    )

    if suffix in RAW_EXTENSIONS:
        with rawpy.imread(
            str(path)
        ) as raw:

            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=False,
            )

        rgb = np.asarray(
            rgb,
            dtype=np.uint8,
        )

    else:
        image_bgr = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise ValueError(
                "OpenCV could not decode the image."
            )

        rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
    ):
        raise ValueError(
            f"Unexpected image shape: {rgb.shape}"
        )

    return rgb


# ============================================================
# FACE DETECTOR
# ============================================================

def create_detector(
    config: dict,
    device: torch.device,
) -> MTCNN:
    detector_config = config[
        "detector"
    ]

    return MTCNN(
        image_size=160,
        margin=0,

        min_face_size=int(
            detector_config[
                "min_face_size"
            ]
        ),

        thresholds=[
            float(value)
            for value in detector_config[
                "thresholds"
            ]
        ],

        factor=float(
            detector_config[
                "factor"
            ]
        ),

        keep_all=True,
        post_process=False,
        device=device,
    )


# ============================================================
# FACE SELECTION
# ============================================================

def bbox_area(
    box: np.ndarray,
) -> float:
    x1, y1, x2, y2 = box

    width = max(
        0.0,
        float(x2 - x1),
    )

    height = max(
        0.0,
        float(y2 - y1),
    )

    return (
        width
        * height
    )


def choose_primary_face(
    boxes,
    probabilities,
    landmarks,
    min_confidence: float,
):
    if (
        boxes is None
        or probabilities is None
        or len(boxes) == 0
    ):
        return None

    candidates = []

    for index, (
        box,
        probability,
    ) in enumerate(
        zip(
            boxes,
            probabilities,
        )
    ):
        if probability is None:
            continue

        probability = float(
            probability
        )

        if not np.isfinite(
            probability
        ):
            continue

        if probability < min_confidence:
            continue

        box = np.asarray(
            box,
            dtype=np.float32,
        )

        if (
            box.shape != (4,)
            or not np.all(
                np.isfinite(
                    box
                )
            )
        ):
            continue

        area = bbox_area(
            box
        )

        if area <= 0:
            continue

        face_landmarks = None

        if (
            landmarks is not None
            and index < len(
                landmarks
            )
        ):
            face_landmarks = np.asarray(
                landmarks[index],
                dtype=np.float32,
            )

        candidates.append(
            {
                "index":
                    index,

                "box":
                    box,

                "probability":
                    probability,

                "landmarks":
                    face_landmarks,

                "area":
                    area,
            }
        )

    if not candidates:
        return None

    # Largest confident face is considered the primary
    # subject in the ND-TWINS photograph.
    candidates.sort(
        key=lambda item: (
            item["area"],
            item["probability"],
        ),
        reverse=True,
    )

    return {
        "primary":
            candidates[0],

        "valid_detections":
            candidates,
    }


# ============================================================
# LANDMARK VALIDATION
# ============================================================

def valid_landmarks(
    landmarks: np.ndarray | None,
) -> bool:
    if landmarks is None:
        return False

    if landmarks.shape != (
        5,
        2,
    ):
        return False

    if not np.all(
        np.isfinite(
            landmarks
        )
    ):
        return False

    eye_distance = float(
        np.linalg.norm(
            landmarks[0]
            - landmarks[1]
        )
    )

    if eye_distance < 2.0:
        return False

    return True


# ============================================================
# ALIGNMENT TEMPLATES
# ============================================================

def arcface_reference(
    output_size: int,
) -> np.ndarray:
    """
    Canonical ArcFace geometry.
    """

    scale = (
        float(output_size)
        / 112.0
    )

    return (
        ARCFACE_REFERENCE_112
        * scale
    ).astype(
        np.float32
    )


def full_face_reference(
    output_size: int,
    face_scale: float,
    vertical_shift: float,
) -> np.ndarray:
    """
    Build a looser full-face template.

    face_scale < 1.0:
        Makes the face smaller inside the output canvas,
        preserving more forehead, chin, jaw and surrounding
        facial context.

    vertical_shift < 0:
        Moves facial landmarks upward and creates more room
        beneath the mouth for the chin/jaw.
    """

    if not (
        0.5
        <= face_scale
        <= 1.0
    ):
        raise ValueError(
            "face_scale must be between 0.5 and 1.0."
        )

    base = arcface_reference(
        output_size
    )

    center = np.array(
        [
            output_size / 2.0,
            output_size / 2.0,
        ],
        dtype=np.float32,
    )

    reference = (
        center
        + face_scale
        * (
            base
            - center
        )
    )

    reference[
        :,
        1,
    ] += (
        vertical_shift
        * output_size
    )

    return reference.astype(
        np.float32
    )


# ============================================================
# ALIGN FACE
# ============================================================

def align_face(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
    output_size: int,
    reference: np.ndarray,
) -> np.ndarray:
    source = landmarks.astype(
        np.float32
    )

    destination = reference.astype(
        np.float32
    )

    matrix, _ = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.LMEDS,
    )

    if matrix is None:
        raise ValueError(
            "Could not estimate facial alignment transform."
        )

    aligned = cv2.warpAffine(
        image_rgb,
        matrix,
        (
            output_size,
            output_size,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0,
        ),
    )

    if aligned is None:
        raise ValueError(
            "Face alignment produced no output."
        )

    if aligned.shape[:2] != (
        output_size,
        output_size,
    ):
        raise ValueError(
            "Unexpected aligned image dimensions: "
            f"{aligned.shape}"
        )

    return aligned


# ============================================================
# BOUNDING BOX
# ============================================================

def clip_bbox(
    box: np.ndarray,
    width: int,
    height: int,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(
            width - 1,
            int(
                round(
                    x1
                )
            ),
        ),
    )

    y1 = max(
        0,
        min(
            height - 1,
            int(
                round(
                    y1
                )
            ),
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            width,
            int(
                round(
                    x2
                )
            ),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            height,
            int(
                round(
                    y2
                )
            ),
        ),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# QUALITY METRICS
# ============================================================

def calculate_quality_metrics(
    image_rgb: np.ndarray,
    bbox: tuple[
        int,
        int,
        int,
        int,
    ],
) -> dict:
    """
    Measure quality using the detected face in the original
    decoded image.

    We intentionally do not calculate blur from the resized
    aligned images.
    """

    x1, y1, x2, y2 = bbox

    face = image_rgb[
        y1:y2,
        x1:x2,
    ]

    if face.size == 0:
        raise ValueError(
            "Detected face crop is empty."
        )

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_RGB2GRAY,
    )

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    brightness_mean = float(
        np.mean(
            gray
        )
    )

    brightness_std = float(
        np.std(
            gray
        )
    )

    return {
        "blur_score":
            blur_score,

        "brightness_mean":
            brightness_mean,

        "brightness_std":
            brightness_std,
    }


# ============================================================
# OUTPUT PATHS
# ============================================================

def aligned_paths(
    subject_id: str,
    image_id: str,

    aligned_112_arcface_dir: Path,
    aligned_112_fullface_dir: Path,
    aligned_448_fullface_dir: Path,

    extension: str,
) -> tuple[
    Path,
    Path,
    Path,
]:
    filename = (
        f"{image_id}.{extension}"
    )

    path_112_arcface = (
        aligned_112_arcface_dir
        / str(subject_id)
        / filename
    )

    path_112_fullface = (
        aligned_112_fullface_dir
        / str(subject_id)
        / filename
    )

    path_448_fullface = (
        aligned_448_fullface_dir
        / str(subject_id)
        / filename
    )

    return (
        path_112_arcface,
        path_112_fullface,
        path_448_fullface,
    )


# ============================================================
# SAVE IMAGE
# ============================================================

def save_png(
    image_rgb: np.ndarray,
    path: Path,
    compression: int,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        path.exists()
        and not overwrite
    ):
        return

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(path),
        image_bgr,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            int(
                compression
            ),
        ],
    )

    if not success:
        raise IOError(
            f"Could not save aligned image: {path}"
        )


# ============================================================
# EMPTY RESULT
# ============================================================

def empty_result() -> dict:
    return {
        column: pd.NA
        for column in QC_COLUMNS
    }


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    row: pd.Series,
    detector: MTCNN,
    config: dict,

    aligned_112_arcface_dir: Path,
    aligned_112_fullface_dir: Path,
    aligned_448_fullface_dir: Path,
) -> dict:
    result = empty_result()

    result[
        "qc_processed"
    ] = True

    image_path = Path(
        str(
            row[
                "image_path"
            ]
        )
    )

    image_id = str(
        row[
            "image_id"
        ]
    )

    subject_id = str(
        row[
            "subject_id"
        ]
    )

    # ========================================================
    # 1. READ ORIGINAL IMAGE
    # ========================================================

    try:
        image_rgb = read_image_rgb(
            image_path
        )

    except Exception as error:
        result[
            "image_readable"
        ] = False

        result[
            "usable"
        ] = False

        result[
            "exclusion_reason"
        ] = "unreadable_image"

        print(
            f"\n[READ ERROR]"
            f"\nImage: {image_path}"
            f"\n{type(error).__name__}: {error}"
        )

        return result

    result[
        "image_readable"
    ] = True

    height, width = (
        image_rgb.shape[:2]
    )

    result[
        "image_width"
    ] = int(
        width
    )

    result[
        "image_height"
    ] = int(
        height
    )

    # ========================================================
    # 2. FACE DETECTION
    # ========================================================

    try:
        pil_image = Image.fromarray(
            image_rgb
        )

        with torch.inference_mode():
            (
                boxes,
                probabilities,
                landmarks,
            ) = detector.detect(
                pil_image,
                landmarks=True,
            )

    except Exception as error:
        result[
            "face_detected"
        ] = False

        result[
            "usable"
        ] = False

        result[
            "exclusion_reason"
        ] = "face_detection_error"

        print(
            f"\n[DETECTION ERROR]"
            f"\nImage: {image_path}"
            f"\n{type(error).__name__}: {error}"
        )

        return result

    min_confidence = float(
        config[
            "detector"
        ][
            "min_detection_confidence"
        ]
    )

    selection = choose_primary_face(
        boxes=boxes,
        probabilities=probabilities,
        landmarks=landmarks,
        min_confidence=min_confidence,
    )

    if selection is None:
        result[
            "face_detected"
        ] = False

        result[
            "number_of_faces_detected"
        ] = 0

        result[
            "usable"
        ] = False

        result[
            "exclusion_reason"
        ] = "no_face_detected"

        return result

    primary = selection[
        "primary"
    ]

    valid_detections = selection[
        "valid_detections"
    ]

    result[
        "face_detected"
    ] = True

    result[
        "number_of_faces_detected"
    ] = int(
        len(
            valid_detections
        )
    )

    result[
        "face_detection_confidence"
    ] = float(
        primary[
            "probability"
        ]
    )

    # ========================================================
    # 3. BOUNDING BOX
    # ========================================================

    bbox = clip_bbox(
        primary[
            "box"
        ],
        width=width,
        height=height,
    )

    (
        x1,
        y1,
        x2,
        y2,
    ) = bbox

    result[
        "face_bbox_x1"
    ] = x1

    result[
        "face_bbox_y1"
    ] = y1

    result[
        "face_bbox_x2"
    ] = x2

    result[
        "face_bbox_y2"
    ] = y2

    face_area = (
        (x2 - x1)
        * (y2 - y1)
    )

    image_area = (
        width
        * height
    )

    result[
        "face_bbox_ratio"
    ] = float(
        face_area
        / image_area
    )

    # ========================================================
    # 4. QUALITY MEASUREMENTS
    # ========================================================

    try:
        quality = calculate_quality_metrics(
            image_rgb,
            bbox,
        )

        result.update(
            quality
        )

    except Exception as error:
        print(
            f"\n[QUALITY WARNING]"
            f"\nImage: {image_path}"
            f"\n{type(error).__name__}: {error}"
        )

    # No arbitrary combined score yet.
    result[
        "quality_score"
    ] = pd.NA

    # ========================================================
    # 5. LANDMARKS
    # ========================================================

    face_landmarks = primary[
        "landmarks"
    ]

    if not valid_landmarks(
        face_landmarks
    ):
        result[
            "landmarks_detected"
        ] = False

        result[
            "usable"
        ] = False

        result[
            "exclusion_reason"
        ] = "invalid_landmarks"

        return result

    result[
        "landmarks_detected"
    ] = True

    (
        left_eye,
        right_eye,
        nose,
        mouth_left,
        mouth_right,
    ) = face_landmarks

    result[
        "pred_left_eye_x"
    ] = float(
        left_eye[0]
    )

    result[
        "pred_left_eye_y"
    ] = float(
        left_eye[1]
    )

    result[
        "pred_right_eye_x"
    ] = float(
        right_eye[0]
    )

    result[
        "pred_right_eye_y"
    ] = float(
        right_eye[1]
    )

    result[
        "pred_nose_x"
    ] = float(
        nose[0]
    )

    result[
        "pred_nose_y"
    ] = float(
        nose[1]
    )

    result[
        "pred_mouth_left_x"
    ] = float(
        mouth_left[0]
    )

    result[
        "pred_mouth_left_y"
    ] = float(
        mouth_left[1]
    )

    result[
        "pred_mouth_right_x"
    ] = float(
        mouth_right[0]
    )

    result[
        "pred_mouth_right_y"
    ] = float(
        mouth_right[1]
    )

    # ========================================================
    # 6. CREATE THREE ALIGNED REPRESENTATIONS
    #
    # A) 112 ArcFace
    #     Canonical crop for pretrained ArcFace/AdaFace.
    #
    # B) 448 full-face
    #     Looser field of view including chin/jaw.
    #
    # C) 112 full-face
    #     Resized from B so its geometry is identical to B.
    # ========================================================

    alignment_config = config[
        "alignment"
    ]

    arcface_size = int(
        alignment_config[
            "arcface_size"
        ]
    )

    fullface_size = int(
        alignment_config[
            "fullface_size"
        ]
    )

    fullface_scale = float(
        alignment_config[
            "fullface_scale"
        ]
    )

    fullface_vertical_shift = float(
        alignment_config[
            "fullface_vertical_shift"
        ]
    )

    try:
        # ----------------------------------------------------
        # A. Canonical ArcFace 112
        # ----------------------------------------------------

        reference_arcface = arcface_reference(
            arcface_size
        )

        aligned_112_arcface = align_face(
            image_rgb=image_rgb,
            landmarks=face_landmarks,
            output_size=arcface_size,
            reference=reference_arcface,
        )

        # ----------------------------------------------------
        # B. Full-face 448
        # ----------------------------------------------------

        reference_fullface = full_face_reference(
            output_size=fullface_size,
            face_scale=fullface_scale,
            vertical_shift=fullface_vertical_shift,
        )

        aligned_448_fullface = align_face(
            image_rgb=image_rgb,
            landmarks=face_landmarks,
            output_size=fullface_size,
            reference=reference_fullface,
        )

        # ----------------------------------------------------
        # C. Full-face 112
        #
        # Same field of view as the 448 version.
        # Only resolution differs.
        # ----------------------------------------------------

        aligned_112_fullface = cv2.resize(
            aligned_448_fullface,
            (
                arcface_size,
                arcface_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

    except Exception as error:
        result[
            "alignment_success"
        ] = False

        result[
            "usable"
        ] = False

        result[
            "exclusion_reason"
        ] = "alignment_failed"

        print(
            f"\n[ALIGNMENT ERROR]"
            f"\nImage: {image_path}"
            f"\n{type(error).__name__}: {error}"
        )

        return result

    # ========================================================
    # 7. OUTPUT PATHS
    # ========================================================

    processing_config = config[
        "processing"
    ]

    extension = (
        str(
            processing_config.get(
                "image_format",
                "png",
            )
        )
        .lower()
        .lstrip(".")
    )

    if extension != "png":
        raise ValueError(
            "Aligned images must currently use PNG "
            "because the preprocessing output is intended "
            "to remain lossless."
        )

    compression = int(
        processing_config.get(
            "png_compression",
            3,
        )
    )

    (
        path_112_arcface,
        path_112_fullface,
        path_448_fullface,
    ) = aligned_paths(
        subject_id=subject_id,
        image_id=image_id,

        aligned_112_arcface_dir=
            aligned_112_arcface_dir,

        aligned_112_fullface_dir=
            aligned_112_fullface_dir,

        aligned_448_fullface_dir=
            aligned_448_fullface_dir,

        extension=extension,
    )

    # ========================================================
    # 8. SAVE OUTPUTS
    # ========================================================

    try:
        save_png(
            aligned_112_arcface,
            path_112_arcface,
            compression=compression,
        )

        save_png(
            aligned_112_fullface,
            path_112_fullface,
            compression=compression,
        )

        save_png(
            aligned_448_fullface,
            path_448_fullface,
            compression=compression,
        )

    except Exception as error:
        result[
            "alignment_success"
        ] = False

        result[
            "usable"
        ] = False

        result[
            "exclusion_reason"
        ] = "aligned_image_save_failed"

        print(
            f"\n[SAVE ERROR]"
            f"\nImage: {image_path}"
            f"\n{type(error).__name__}: {error}"
        )

        return result

    result[
        "aligned_112_arcface_path"
    ] = stored_path(
        path_112_arcface
    )

    result[
        "aligned_112_fullface_path"
    ] = stored_path(
        path_112_fullface
    )

    result[
        "aligned_448_fullface_path"
    ] = stored_path(
        path_448_fullface
    )

    result[
        "alignment_success"
    ] = True

    # ========================================================
    # 9. FINAL USABILITY
    #
    # Pose, illumination, expression and eyewear are NOT QC
    # exclusion criteria.
    # ========================================================

    result[
        "usable"
    ] = True

    result[
        "exclusion_reason"
    ] = pd.NA

    return result


# ============================================================
# ATOMIC CHECKPOINT WRITING
# ============================================================

def atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    df.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def atomic_write_json(
    data: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )

    os.replace(
        temporary,
        path,
    )


# ============================================================
# NUMERIC STATISTICS
# ============================================================

def numeric_stats(
    series: pd.Series,
) -> dict:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "count": 0,
        }

    return {
        "count":
            int(
                len(values)
            ),

        "mean":
            float(
                values.mean()
            ),

        "std":
            float(
                values.std()
            ),

        "min":
            float(
                values.min()
            ),

        "p05":
            float(
                values.quantile(
                    0.05
                )
            ),

        "median":
            float(
                values.median()
            ),

        "p95":
            float(
                values.quantile(
                    0.95
                )
            ),

        "max":
            float(
                values.max()
            ),
    }


# ============================================================
# QC SUMMARY
# ============================================================

def build_summary(
    df: pd.DataFrame,
) -> dict:
    processed = (
        df[
            "qc_processed"
        ]
        .map(
            as_bool
        )
    )

    readable = (
        df[
            "image_readable"
        ]
        .map(
            as_bool
        )
    )

    face_detected = (
        df[
            "face_detected"
        ]
        .map(
            as_bool
        )
    )

    landmarks_detected = (
        df[
            "landmarks_detected"
        ]
        .map(
            as_bool
        )
    )

    alignment_success = (
        df[
            "alignment_success"
        ]
        .map(
            as_bool
        )
    )

    usable = (
        df[
            "usable"
        ]
        .map(
            as_bool
        )
    )

    face_counts = pd.to_numeric(
        df[
            "number_of_faces_detected"
        ],
        errors="coerce",
    )

    reasons = (
        df.loc[
            processed
            & ~usable,
            "exclusion_reason",
        ]
        .fillna(
            "unknown"
        )
        .value_counts()
        .to_dict()
    )

    return {
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "total_images":
            int(
                len(df)
            ),

        "processed":
            int(
                processed.sum()
            ),

        "remaining":
            int(
                len(df)
                - processed.sum()
            ),

        "readable":
            int(
                (
                    processed
                    & readable
                ).sum()
            ),

        "unreadable":
            int(
                (
                    processed
                    & ~readable
                ).sum()
            ),

        "face_detected":
            int(
                (
                    processed
                    & face_detected
                ).sum()
            ),

        "face_not_detected":
            int(
                (
                    processed
                    & readable
                    & ~face_detected
                ).sum()
            ),

        "landmarks_detected":
            int(
                (
                    processed
                    & landmarks_detected
                ).sum()
            ),

        "alignment_success":
            int(
                (
                    processed
                    & alignment_success
                ).sum()
            ),

        "multiple_faces":
            int(
                (
                    processed
                    & (
                        face_counts
                        > 1
                    )
                ).sum()
            ),

        "usable":
            int(
                (
                    processed
                    & usable
                ).sum()
            ),

        "unusable":
            int(
                (
                    processed
                    & ~usable
                ).sum()
            ),

        "exclusion_reasons": {
            str(key):
                int(value)

            for key, value
            in reasons.items()
        },

        "face_detection_confidence":
            numeric_stats(
                df.loc[
                    processed,
                    "face_detection_confidence",
                ]
            ),

        "face_bbox_ratio":
            numeric_stats(
                df.loc[
                    processed,
                    "face_bbox_ratio",
                ]
            ),

        "blur_score":
            numeric_stats(
                df.loc[
                    processed,
                    "blur_score",
                ]
            ),

        "brightness_mean":
            numeric_stats(
                df.loc[
                    processed,
                    "brightness_mean",
                ]
            ),

        "brightness_std":
            numeric_stats(
                df.loc[
                    processed,
                    "brightness_std",
                ]
            ),
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    df: pd.DataFrame,
    metadata_path: Path,
    report_path: Path,
) -> None:
    atomic_write_csv(
        df,
        metadata_path,
    )

    summary = build_summary(
        df
    )

    atomic_write_json(
        summary,
        report_path,
    )


# ============================================================
# PREPARE / RESUME DATAFRAME
# ============================================================

def prepare_dataframe(
    master: pd.DataFrame,
    output_path: Path,
    resume: bool,
) -> pd.DataFrame:
    required_columns = {
        "image_id",
        "subject_id",
        "image_path",
    }

    missing_columns = (
        required_columns
        - set(
            master.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Master metadata is missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if master[
        "image_id"
    ].duplicated().any():
        raise ValueError(
            "Master metadata contains duplicate image IDs."
        )

    # ========================================================
    # RESUME
    # ========================================================

    if (
        resume
        and output_path.exists()
    ):
        print(
            "\nResuming from existing QC metadata:"
        )

        print(
            output_path
        )

        previous = pd.read_csv(
            output_path,
            low_memory=False,
        )

        if (
            "image_id"
            not in previous.columns
        ):
            raise ValueError(
                "Existing QC checkpoint does not contain "
                "image_id."
            )

        if previous[
            "image_id"
        ].duplicated().any():
            raise ValueError(
                "Existing QC checkpoint contains duplicate "
                "image IDs."
            )

        master_ids = set(
            master[
                "image_id"
            ]
            .astype(
                str
            )
        )

        previous_ids = set(
            previous[
                "image_id"
            ]
            .astype(
                str
            )
        )

        if (
            master_ids
            != previous_ids
        ):
            raise ValueError(
                "Existing QC checkpoint does not match "
                "the current master metadata."
            )

        previous = (
            previous
            .set_index(
                "image_id"
            )
        )

        df = master.copy()

        # Remove placeholder/stale QC fields from the current
        # master before copying checkpoint-generated values.
        for column in QC_COLUMNS:
            if column in df.columns:
                df = df.drop(
                    columns=[
                        column
                    ]
                )

        for column in QC_COLUMNS:
            if (
                column
                not in previous.columns
            ):
                continue

            df[
                column
            ] = (
                df[
                    "image_id"
                ]
                .map(
                    previous[
                        column
                    ]
                )
            )

    # ========================================================
    # NEW RUN
    # ========================================================

    else:
        df = master.copy()

        # build_metadata.py may contain reserved QC columns.
        # Recreate them here with proper pandas nullable types.
        for column in QC_COLUMNS:
            if column in df.columns:
                df = df.drop(
                    columns=[
                        column
                    ]
                )

    # ========================================================
    # ENFORCE QC DTYPES
    # ========================================================

    df = apply_qc_schema(
        df
    )

    if not (
        resume
        and output_path.exists()
    ):
        df[
            "qc_processed"
        ] = pd.Series(
            False,
            index=df.index,
            dtype="boolean",
        )

    return df


# ============================================================
# COMPLETION CHECK
# ============================================================

def row_is_complete(
    row,
) -> bool:
    if not as_bool(
        row.qc_processed
    ):
        return False

    # Controlled failures are considered complete.
    if not as_bool(
        row.alignment_success
    ):
        return True

    path_arcface = resolve_stored_path(
        row.aligned_112_arcface_path
    )

    path_112_fullface = resolve_stored_path(
        row.aligned_112_fullface_path
    )

    path_448_fullface = resolve_stored_path(
        row.aligned_448_fullface_path
    )

    if (
        path_arcface is None
        or path_112_fullface is None
        or path_448_fullface is None
    ):
        return False

    return (
        path_arcface.exists()
        and path_112_fullface.exists()
        and path_448_fullface.exists()
    )


# ============================================================
# CLEAR ROW BEFORE REPROCESSING
# ============================================================

def clear_qc_row(
    df: pd.DataFrame,
    index,
) -> None:
    for column in QC_COLUMNS:
        df.at[
            index,
            column,
        ] = pd.NA

    df.at[
        index,
        "qc_processed",
    ] = False


# ============================================================
# MAIN
# ============================================================

def main(
    config_path: Path,
) -> None:
    config = load_config(
        config_path
    )

    # ========================================================
    # PATHS
    # ========================================================

    input_metadata = resolve_project_path(
        config[
            "input"
        ][
            "metadata_csv"
        ]
    )

    output_metadata = resolve_project_path(
        config[
            "output"
        ][
            "metadata_csv"
        ]
    )

    aligned_112_arcface_dir = resolve_project_path(
        config[
            "output"
        ][
            "aligned_112_arcface_dir"
        ]
    )

    aligned_112_fullface_dir = resolve_project_path(
        config[
            "output"
        ][
            "aligned_112_fullface_dir"
        ]
    )

    aligned_448_fullface_dir = resolve_project_path(
        config[
            "output"
        ][
            "aligned_448_fullface_dir"
        ]
    )

    report_path = resolve_project_path(
        config[
            "output"
        ][
            "report_json"
        ]
    )

    if not input_metadata.exists():
        raise FileNotFoundError(
            "Master metadata does not exist:\n"
            f"{input_metadata}"
        )

    output_metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aligned_112_arcface_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aligned_112_fullface_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aligned_448_fullface_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # ALIGNMENT CONFIG
    # ========================================================

    alignment_config = config[
        "alignment"
    ]

    arcface_size = int(
        alignment_config[
            "arcface_size"
        ]
    )

    fullface_size = int(
        alignment_config[
            "fullface_size"
        ]
    )

    fullface_scale = float(
        alignment_config[
            "fullface_scale"
        ]
    )

    fullface_vertical_shift = float(
        alignment_config[
            "fullface_vertical_shift"
        ]
    )

    if arcface_size != 112:
        print(
            "\nWARNING:"
            f" arcface_size is {arcface_size}. "
            "Most pretrained ArcFace models expect 112x112."
        )

    if fullface_size < arcface_size:
        raise ValueError(
            "fullface_size must be greater than or equal "
            "to arcface_size."
        )

    if not (
        0.5
        <= fullface_scale
        <= 1.0
    ):
        raise ValueError(
            "fullface_scale must be between 0.5 and 1.0."
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = resolve_device(
        config[
            "processing"
        ].get(
            "device",
            "auto",
        )
    )

    print(
        "========================================"
    )

    print(
        "ND-TWINS IMAGE QUALITY CONTROL"
    )

    print(
        "========================================"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    print(
        f"Canonical ArcFace: "
        f"{arcface_size}x{arcface_size}"
    )

    print(
        f"Full-face master: "
        f"{fullface_size}x{fullface_size}"
    )

    print(
        f"Full-face scale: "
        f"{fullface_scale}"
    )

    print(
        f"Full-face vertical shift: "
        f"{fullface_vertical_shift}"
    )

    # ========================================================
    # LOAD MASTER
    # ========================================================

    print(
        f"\nLoading metadata:\n"
        f"{input_metadata}"
    )

    master = pd.read_csv(
        input_metadata,
        low_memory=False,
    )

    print(
        f"Images: "
        f"{len(master):,}"
    )

    print(
        f"Subjects: "
        f"{master['subject_id'].nunique():,}"
    )

    # ========================================================
    # PREPARE / RESUME
    # ========================================================

    resume = bool(
        config[
            "processing"
        ].get(
            "resume",
            True,
        )
    )

    df = prepare_dataframe(
        master=master,
        output_path=output_metadata,
        resume=resume,
    )

    # ========================================================
    # MTCNN
    # ========================================================

    print(
        "\nInitializing MTCNN..."
    )

    detector = create_detector(
        config,
        device,
    )

    # ========================================================
    # PROCESSING OPTIONS
    # ========================================================

    checkpoint_every = int(
        config[
            "processing"
        ].get(
            "checkpoint_every",
            25,
        )
    )

    if checkpoint_every <= 0:
        raise ValueError(
            "checkpoint_every must be greater than zero."
        )

    limit = (
        config[
            "processing"
        ].get(
            "limit"
        )
    )

    if limit is not None:
        limit = int(
            limit
        )

        if limit <= 0:
            raise ValueError(
                "processing.limit must be positive "
                "or null."
            )

    # ========================================================
    # DETERMINE REMAINING WORK
    # ========================================================

    complete_mask = np.array(
        [
            row_is_complete(
                row
            )
            for row
            in df.itertuples(
                index=False
            )
        ],
        dtype=bool,
    )

    already_complete = int(
        complete_mask.sum()
    )

    rows_to_process = (
        df.index[
            ~complete_mask
        ]
        .tolist()
    )

    total_remaining = len(
        rows_to_process
    )

    if limit is not None:
        rows_to_process = (
            rows_to_process[
                :limit
            ]
        )

    print(
        f"\nAlready complete: "
        f"{already_complete:,}"
    )

    print(
        f"Remaining overall: "
        f"{total_remaining:,}"
    )

    print(
        f"Processing this run: "
        f"{len(rows_to_process):,}"
    )

    if not rows_to_process:
        save_checkpoint(
            df,
            output_metadata,
            report_path,
        )

        print(
            "\nNothing left to process."
        )

        return

    # ========================================================
    # RUN QC
    # ========================================================

    processed_this_run = 0

    try:
        for index in tqdm(
            rows_to_process,
            desc="QC",
            unit="image",
        ):
            row = df.loc[
                index
            ]

            clear_qc_row(
                df,
                index,
            )

            result = process_image(
                row=row,
                detector=detector,
                config=config,

                aligned_112_arcface_dir=
                    aligned_112_arcface_dir,

                aligned_112_fullface_dir=
                    aligned_112_fullface_dir,

                aligned_448_fullface_dir=
                    aligned_448_fullface_dir,
            )

            for (
                column,
                value,
            ) in result.items():
                df.at[
                    index,
                    column,
                ] = value

            processed_this_run += 1

            if (
                processed_this_run
                % checkpoint_every
                == 0
            ):
                save_checkpoint(
                    df,
                    output_metadata,
                    report_path,
                )

    except KeyboardInterrupt:
        print(
            "\n\nInterrupted by user."
        )

        print(
            "Saving checkpoint..."
        )

        save_checkpoint(
            df,
            output_metadata,
            report_path,
        )

        print(
            "Checkpoint saved."
        )

        return

    except Exception:
        print(
            "\nUnexpected error."
        )

        print(
            "Saving completed work before raising error..."
        )

        save_checkpoint(
            df,
            output_metadata,
            report_path,
        )

        raise

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    save_checkpoint(
        df,
        output_metadata,
        report_path,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = build_summary(
        df
    )

    print(
        "\n========================================"
    )

    print(
        "ND-TWINS IMAGE QC RUN COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Total images:       "
        f"{summary['total_images']:,}"
    )

    print(
        f"Processed:          "
        f"{summary['processed']:,}"
    )

    print(
        f"Remaining:          "
        f"{summary['remaining']:,}"
    )

    print(
        f"Readable:           "
        f"{summary['readable']:,}"
    )

    print(
        f"Unreadable:         "
        f"{summary['unreadable']:,}"
    )

    print(
        f"Faces detected:     "
        f"{summary['face_detected']:,}"
    )

    print(
        f"No face detected:   "
        f"{summary['face_not_detected']:,}"
    )

    print(
        f"Landmarks detected: "
        f"{summary['landmarks_detected']:,}"
    )

    print(
        f"Alignment success:  "
        f"{summary['alignment_success']:,}"
    )

    print(
        f"Multiple faces:     "
        f"{summary['multiple_faces']:,}"
    )

    print(
        f"Usable:             "
        f"{summary['usable']:,}"
    )

    print(
        f"Unusable:           "
        f"{summary['unusable']:,}"
    )

    if summary[
        "exclusion_reasons"
    ]:
        print(
            "\nExclusion reasons:"
        )

        for (
            reason,
            count,
        ) in summary[
            "exclusion_reasons"
        ].items():

            print(
                f"  {reason}: "
                f"{count:,}"
            )

    print(
        f"\nQC metadata:\n"
        f"{output_metadata}"
    )

    print(
        f"\nQC report:\n"
        f"{report_path}"
    )

    print(
        f"\nCanonical ArcFace 112:\n"
        f"{aligned_112_arcface_dir}"
    )

    print(
        f"\nFull-face 112:\n"
        f"{aligned_112_fullface_dir}"
    )

    print(
        f"\nFull-face 448:\n"
        f"{aligned_448_fullface_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run ND-TWINS image QC and generate canonical "
            "ArcFace and full-face aligned images."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/qc.yaml"
        ),
        help=(
            "Path to QC YAML configuration."
        ),
    )

    args = parser.parse_args()

    main(
        args.config
    )