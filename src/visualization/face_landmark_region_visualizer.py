from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.image_qc import (
    create_detector,
    load_config,
    read_image_rgb,
    resolve_device,
    resolve_project_path,
)


# ============================================================
# MEDIAPIPE LANDMARK GROUPS
#
# These are used only for visualization / region definition.
# They do not replace the MTCNN alignment landmarks.
# ============================================================

FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


LEFT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]


RIGHT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]


LEFT_EYEBROW = [
    70,
    63,
    105,
    66,
    107,
    55,
    65,
    52,
    53,
    46,
]


RIGHT_EYEBROW = [
    300,
    293,
    334,
    296,
    336,
    285,
    295,
    282,
    283,
    276,
]


NOSE_REGION = [
    168,
    6,
    197,
    195,
    5,
    4,
    1,
    19,
    94,
    2,
    98,
    97,
    326,
    327,
    129,
    358,
]


MOUTH_REGION = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
]


# Lower part of the face oval:
# left cheek -> jaw -> chin -> right cheek
JAW_CONTOUR = [
    234,
    93,
    132,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    361,
    323,
    454,
]


# ============================================================
# DISPLAY COLORS
#
# OpenCV uses BGR.
# ============================================================

COLOR_FACE = (255, 255, 255)

COLOR_LEFT_EYE = (0, 255, 0)

COLOR_RIGHT_EYE = (0, 255, 255)

COLOR_NOSE = (255, 0, 0)

COLOR_MOUTH = (0, 0, 255)

COLOR_JAW = (255, 0, 255)

COLOR_MTCNN = (0, 165, 255)


# ============================================================
# PROJECT HELPERS
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def report_directory(
    image_id: str,
) -> Path:
    output = (
        project_root()
        / "reports"
        / "figures"
        / "landmark_demo"
        / image_id
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output


# ============================================================
# IMAGE SELECTION
# ============================================================

def find_image_from_metadata(
    metadata_path: Path,
    image_id: str,
) -> Path:
    metadata = pd.read_csv(
        metadata_path,
        low_memory=False,
    )

    if "image_id" not in metadata.columns:
        raise ValueError(
            "Metadata does not contain image_id."
        )

    matches = metadata[
        metadata["image_id"].astype(str)
        == str(image_id)
    ]

    if matches.empty:
        raise ValueError(
            f"Image ID '{image_id}' was not found."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Image ID '{image_id}' is not unique."
        )

    image_path = Path(
        str(
            matches.iloc[0][
                "image_path"
            ]
        )
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist:\n{image_path}"
        )

    return image_path


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_point(
    image: np.ndarray,
    point: tuple[int, int],
    color,
    radius: int = 8,
) -> None:
    cv2.circle(
        image,
        point,
        radius,
        color,
        -1,
        lineType=cv2.LINE_AA,
    )


def draw_label(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def landmark_points(
    landmarks: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    return np.asarray(
        [
            landmarks[index]
            for index in indices
        ],
        dtype=np.int32,
    )


def draw_convex_region(
    image: np.ndarray,
    landmarks: np.ndarray,
    indices: list[int],
    color,
    alpha: float = 0.18,
) -> None:
    points = landmark_points(
        landmarks,
        indices,
    )

    hull = cv2.convexHull(
        points
    )

    overlay = image.copy()

    cv2.fillConvexPoly(
        overlay,
        hull,
        color,
        lineType=cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
        image,
    )

    cv2.polylines(
        image,
        [hull],
        True,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_periocular_region(
    image: np.ndarray,
    landmarks: np.ndarray,
    indices: list[int],
    color,
    horizontal_padding: float = 0.20,
    top_padding: float = 0.15,
    bottom_padding: float = 0.45,
    alpha: float = 0.18,
    thickness: int = 2,
) -> None:
    """
    Draw an expanded periocular region.

    The region includes the eyebrow, upper eyelid, eye, lower
    eyelid, and infraorbital / under-eye area. Extra padding is
    intentionally larger below the eye because this area may
    contain useful twin-discriminative information.
    """
    points = landmark_points(
        landmarks,
        indices,
    )

    min_x = int(points[:, 0].min())
    max_x = int(points[:, 0].max())
    min_y = int(points[:, 1].min())
    max_y = int(points[:, 1].max())

    region_width = max_x - min_x
    region_height = max_y - min_y

    x_padding = int(round(region_width * horizontal_padding))
    y_top_padding = int(round(region_height * top_padding))
    y_bottom_padding = int(round(region_height * bottom_padding))

    image_height, image_width = image.shape[:2]

    x1 = max(0, min_x - x_padding)
    y1 = max(0, min_y - y_top_padding)
    x2 = min(image_width - 1, max_x + x_padding)
    y2 = min(image_height - 1, max_y + y_bottom_padding)

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1,
        lineType=cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
        image,
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_scaled_region(
    image: np.ndarray,
    landmarks: np.ndarray,
    indices: list[int],
    color,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shift_y: float = 0.0,
    alpha: float = 0.18,
    thickness: int = 2,
) -> None:
    """
    Draw a scaled version of a landmark-defined region.

    This is useful for regions such as the mouth, where the exact
    landmark hull is too tight and a larger semantic ROI is needed.
    ``shift_y`` is expressed as a fraction of the original region
    height.
    """
    points = landmark_points(
        landmarks,
        indices,
    ).astype(np.float32)

    center = points.mean(axis=0)
    height = points[:, 1].max() - points[:, 1].min()

    scaled_points = points.copy()
    scaled_points[:, 0] = (
        center[0]
        + (points[:, 0] - center[0]) * scale_x
    )
    scaled_points[:, 1] = (
        center[1]
        + (points[:, 1] - center[1]) * scale_y
        + height * shift_y
    )

    image_height, image_width = image.shape[:2]
    scaled_points[:, 0] = np.clip(
        scaled_points[:, 0],
        0,
        image_width - 1,
    )
    scaled_points[:, 1] = np.clip(
        scaled_points[:, 1],
        0,
        image_height - 1,
    )

    hull = cv2.convexHull(
        np.rint(scaled_points).astype(np.int32)
    )

    overlay = image.copy()

    cv2.fillConvexPoly(
        overlay,
        hull,
        color,
        lineType=cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
        image,
    )

    cv2.polylines(
        image,
        [hull],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_polyline(
    image: np.ndarray,
    landmarks: np.ndarray,
    indices: list[int],
    color,
    thickness: int = 4,
    closed: bool = False,
) -> None:
    points = landmark_points(
        landmarks,
        indices,
    )

    cv2.polylines(
        image,
        [points],
        closed,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# MTCNN 5-POINT LANDMARKS
# ============================================================

def detect_mtcnn_landmarks(
    image_rgb: np.ndarray,
    detector,
    min_confidence: float,
):
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

    if boxes is None:
        raise RuntimeError(
            "MTCNN detected no face."
        )

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

        if float(
            probability
        ) < min_confidence:
            continue

        x1, y1, x2, y2 = box

        area = (
            max(
                0,
                x2 - x1,
            )
            *
            max(
                0,
                y2 - y1,
            )
        )

        candidates.append(
            (
                area,
                float(
                    probability
                ),
                box,
                landmarks[index],
            )
        )

    if not candidates:
        raise RuntimeError(
            "No MTCNN face passed the confidence threshold."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    _, confidence, box, points = (
        candidates[0]
    )

    return (
        np.asarray(
            box,
            dtype=np.float32,
        ),
        np.asarray(
            points,
            dtype=np.float32,
        ),
        confidence,
    )


def create_mtcnn_figure(
    image_rgb: np.ndarray,
    detector,
    min_confidence: float,
) -> np.ndarray:
    box, landmarks, confidence = (
        detect_mtcnn_landmarks(
            image_rgb,
            detector,
            min_confidence,
        )
    )

    output = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    x1, y1, x2, y2 = (
        box.astype(int)
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        COLOR_MTCNN,
        4,
    )

    names = [
        "Left eye",
        "Right eye",
        "Nose",
        "Mouth L",
        "Mouth R",
    ]

    for point, name in zip(
        landmarks,
        names,
    ):
        x, y = point.astype(int)

        draw_point(
            output,
            (x, y),
            COLOR_MTCNN,
            radius=5,
        )

        draw_label(
            output,
            name,
            (
                x + 8,
                y - 6,
            ),
            COLOR_MTCNN,
            font_scale=0.42,
            thickness=1,
        )

    draw_label(
        output,
        f"MTCNN confidence: {confidence:.4f}",
        (
            20,
            30,
        ),
        COLOR_MTCNN,
        font_scale=0.48,
        thickness=1,
    )

    return output


# ============================================================
# MEDIAPIPE DENSE LANDMARKS
# ============================================================

def detect_dense_landmarks(
    image_rgb: np.ndarray,
    model_path: Path,
    detection_confidence: float = 0.35,
    presence_confidence: float = 0.35,
    max_detection_dimension: int = 2000,
) -> np.ndarray:
    """
    Detect dense facial landmarks using the modern
    MediaPipe Face Landmarker Tasks API.

    Returns landmark coordinates in the coordinate system
    of the ORIGINAL image.

    MediaPipe Face Landmarker currently provides 478
    normalized 3D facial landmarks.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            "MediaPipe Face Landmarker model was not found:\n"
            f"{model_path}\n\n"
            "Expected model file:\n"
            "models/mediapipe/face_landmarker.task"
        )

    original_height, original_width = (
        image_rgb.shape[:2]
    )

    # --------------------------------------------------------
    # MediaPipe does not need the original 2868x4310 image.
    #
    # Downscale temporarily for landmark inference.
    # Because the aspect ratio is unchanged, normalized
    # landmark coordinates can be mapped directly back to
    # the original image.
    # --------------------------------------------------------
    max_dimension = max_detection_dimension
    
    scale = min(
        1.0,
        max_dimension
        / float(
            max(
                original_width,
                original_height,
            )
        ),
    )

    if scale < 1.0:
        detection_width = int(
            round(
                original_width
                * scale
            )
        )

        detection_height = int(
            round(
                original_height
                * scale
            )
        )

        detection_image = cv2.resize(
            image_rgb,
            (
                detection_width,
                detection_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    else:
        detection_image = image_rgb

    # MediaPipe expects a contiguous uint8 RGB array.
    detection_image = np.ascontiguousarray(
        detection_image,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Modern MediaPipe Tasks API
    # --------------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions

    FaceLandmarker = (
        mp.tasks.vision.FaceLandmarker
    )

    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(
                model_path
            )
        ),

        running_mode=RunningMode.IMAGE,

        num_faces=1,

        min_face_detection_confidence=detection_confidence,

        min_face_presence_confidence=presence_confidence,

        min_tracking_confidence=0.5,

        output_face_blendshapes=False,

        output_facial_transformation_matrixes=False,
    )



    # --------------------------------------------------------
    # Convert NumPy RGB image to MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=detection_image,
    )

    # --------------------------------------------------------
    # Detect landmarks
    # --------------------------------------------------------

    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        result = landmarker.detect(
            mp_image
        )

    if (
        result.face_landmarks is None
        or len(
            result.face_landmarks
        ) == 0
    ):
        raise RuntimeError(
            "MediaPipe Face Landmarker did not "
            "detect a face."
        )

    face_landmarks = (
        result.face_landmarks[0]
    )

    # --------------------------------------------------------
    # Convert normalized MediaPipe coordinates into ORIGINAL
    # image pixel coordinates.
    #
    # MediaPipe:
    #
    #       x ∈ approximately [0, 1]
    #       y ∈ approximately [0, 1]
    #
    # Original pixel coordinates:
    #
    #       X = x * original_width
    #       Y = y * original_height
    # --------------------------------------------------------

    points = []

    for landmark in face_landmarks:
        x = int(
            round(
                landmark.x
                * original_width
            )
        )

        y = int(
            round(
                landmark.y
                * original_height
            )
        )

        # Protect visualization code against occasional
        # coordinates slightly outside [0, 1].
        x = max(
            0,
            min(
                original_width - 1,
                x,
            ),
        )

        y = max(
            0,
            min(
                original_height - 1,
                y,
            ),
        )

        points.append(
            (
                x,
                y,
            )
        )

    landmarks_array = np.asarray(
        points,
        dtype=np.int32,
    )

    if (
        landmarks_array.ndim != 2
        or landmarks_array.shape[1] != 2
    ):
        raise RuntimeError(
            "Unexpected MediaPipe landmark output "
            f"shape: {landmarks_array.shape}"
        )

    return landmarks_array

# ============================================================
# DENSE LANDMARK FIGURE
# ============================================================

def create_dense_landmark_figure(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    output = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    # Draw every MediaPipe landmark.
    for x, y in landmarks:
        cv2.circle(
            output,
            (
                int(x),
                int(y),
            ),
            2,
            (
                0,
                255,
                255,
            ),
            -1,
            cv2.LINE_AA,
        )

    # Highlight complete face outline.
    draw_polyline(
        output,
        landmarks,
        FACE_OVAL,
        COLOR_FACE,
        thickness=4,
        closed=True,
    )

    return output


# ============================================================
# REGION OVERLAY
# ============================================================

def create_region_figure(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    output = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    # --------------------------------------------------------
    # FACE OUTLINE
    # --------------------------------------------------------

    draw_polyline(
        output,
        landmarks,
        FACE_OVAL,
        COLOR_FACE,
        thickness=3,
        closed=True,
    )

    # --------------------------------------------------------
    # PERIOCULAR REGIONS
    #
    # Eye + eyebrow points are combined into a local region.
    # --------------------------------------------------------

    left_periocular = (
        LEFT_EYE
        + LEFT_EYEBROW
    )

    right_periocular = (
        RIGHT_EYE
        + RIGHT_EYEBROW
    )

    draw_periocular_region(
        output,
        landmarks,
        left_periocular,
        COLOR_LEFT_EYE,
        horizontal_padding=0.14,
        top_padding=0.10,
        bottom_padding=0.32,
    )

    draw_periocular_region(
        output,
        landmarks,
        right_periocular,
        COLOR_RIGHT_EYE,
        horizontal_padding=0.14,
        top_padding=0.10,
        bottom_padding=0.32,
    )

    # --------------------------------------------------------
    # NOSE
    # --------------------------------------------------------

    draw_convex_region(
        output,
        landmarks,
        NOSE_REGION,
        COLOR_NOSE,
    )

    # --------------------------------------------------------
    # MOUTH
    # --------------------------------------------------------

    draw_scaled_region(
        output,
        landmarks,
        MOUTH_REGION,
        COLOR_MOUTH,
        scale_x=1.22,
        scale_y=1.30,
        shift_y=0.05,
        alpha=0.18,
        thickness=2,
    )

    # --------------------------------------------------------
    # JAW / FACE CONTOUR
    # --------------------------------------------------------

    draw_polyline(
        output,
        landmarks,
        JAW_CONTOUR,
        COLOR_JAW,
        thickness=3,
        closed=False,
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    draw_label(
        output,
        "Left periocular",
        (
            15,
            25,
        ),
        COLOR_LEFT_EYE,
        font_scale=0.45,
        thickness=1,
    )

    draw_label(
        output,
        "Right periocular",
        (
            15,
            48,
        ),
        COLOR_RIGHT_EYE,
        font_scale=0.45,
        thickness=1,
    )

    draw_label(
        output,
        "Nose",
        (
            15,
            71,
        ),
        COLOR_NOSE,
        font_scale=0.45,
        thickness=1,
    )

    draw_label(
        output,
        "Mouth",
        (
            15,
            94,
        ),
        COLOR_MOUTH,
        font_scale=0.45,
        thickness=1,
    )

    draw_label(
        output,
        "Jaw / contour",
        (
            15,
            117,
        ),
        COLOR_JAW,
        font_scale=0.45,
        thickness=1,
    )

    return output


# ============================================================
# REGION CROP BOUNDING BOX
# ============================================================

def region_bbox(
    landmarks: np.ndarray,
    indices: list[int],
    image_width: int,
    image_height: int,
    padding: float = 0.15,
):
    points = landmark_points(
        landmarks,
        indices,
    )

    min_x = int(
        points[:, 0].min()
    )

    max_x = int(
        points[:, 0].max()
    )

    min_y = int(
        points[:, 1].min()
    )

    max_y = int(
        points[:, 1].max()
    )

    width = (
        max_x
        - min_x
    )

    height = (
        max_y
        - min_y
    )

    pad_x = int(
        width
        * padding
    )

    pad_y = int(
        height
        * padding
    )

    x1 = max(
        0,
        min_x - pad_x,
    )

    y1 = max(
        0,
        min_y - pad_y,
    )

    x2 = min(
        image_width,
        max_x + pad_x,
    )

    y2 = min(
        image_height,
        max_y + pad_y,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# CREATE REPORT PANEL
# ============================================================

def create_report_panel(
    original_rgb: np.ndarray,
    mtcnn_bgr: np.ndarray,
    dense_bgr: np.ndarray,
    regions_bgr: np.ndarray,
    output_path: Path,
) -> None:
    figures = [
        (
            original_rgb,
            "Original image",
            True,
        ),
        (
            mtcnn_bgr,
            "MTCNN 5-point landmarks",
            False,
        ),
        (
            dense_bgr,
            "Dense facial landmarks",
            False,
        ),
        (
            regions_bgr,
            "Proposed facial regions",
            False,
        ),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            20,
            6,
        ),
    )

    for axis, (
        image,
        title,
        is_rgb,
    ) in zip(
        axes,
        figures,
    ):
        if is_rgb:
            display = image
        else:
            display = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        axis.imshow(
            display
        )

        axis.set_title(
            title,
            fontsize=14,
        )

        axis.axis(
            "off"
        )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# SAVE BGR IMAGE
# ============================================================

def save_bgr(
    path: Path,
    image: np.ndarray,
) -> None:
    success = cv2.imwrite(
        str(path),
        image,
    )

    if not success:
        raise IOError(
            f"Could not save {path}"
        )


# ============================================================
# MAIN
# ============================================================

def main(
    config_path: Path,
    image_id: str | None,
    image_path_argument: Path | None,
) -> None:
    config = load_config(
        config_path
    )

    # --------------------------------------------------------
    # Select source image
    # --------------------------------------------------------

    if image_path_argument is not None:
        image_path = (
            image_path_argument
            .expanduser()
            .resolve()
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist:\n"
                f"{image_path}"
            )

        selected_id = (
            image_path.stem
        )

    else:
        metadata_path = (
            resolve_project_path(
                config[
                    "input"
                ][
                    "metadata_csv"
                ]
            )
        )

        image_path = (
            find_image_from_metadata(
                metadata_path,
                image_id,
            )
        )

        selected_id = (
            image_id
        )

    print(
        "Selected image:"
    )

    print(
        image_path
    )

    # --------------------------------------------------------
    # Load original
    # --------------------------------------------------------

    image_rgb = read_image_rgb(
        image_path
    )

    height, width = (
        image_rgb.shape[:2]
    )

    print(
        f"Original resolution: "
        f"{width} x {height}"
    )

    # --------------------------------------------------------
    # MTCNN
    # --------------------------------------------------------

    device = resolve_device(
        config[
            "processing"
        ].get(
            "device",
            "auto",
        )
    )

    print(
        f"Device: {device}"
    )

    detector = create_detector(
        config,
        device,
    )

    visualization_config = config.get(
        "visualization",
        {}
    )

    min_confidence = float(
        visualization_config.get(
            "mtcnn_min_confidence",
            config[
                "detector"
            ][
                "min_detection_confidence"
            ],
        )
    )

    print(
        "Detecting MTCNN landmarks..."
    )

    mtcnn_figure = (
        create_mtcnn_figure(
            image_rgb,
            detector,
            min_confidence,
        )
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    print(
        "Detecting dense facial landmarks..."
    )

    model_path = resolve_project_path(
        config[
            "visualization"
        ][
            "face_landmarker_model"
        ]
    )

    print(
        "Face Landmarker model:"
    )

    print(
        model_path
    )

    visualization_config = config.get(
    "visualization",
    {}
)

    dense_landmarks = detect_dense_landmarks(
        image_rgb=image_rgb,

        model_path=model_path,

        detection_confidence=float(
            visualization_config.get(
                "mediapipe_detection_confidence",
                0.35,
            )
        ),

        presence_confidence=float(
            visualization_config.get(
                "mediapipe_presence_confidence",
                0.35,
            )
        ),

        max_detection_dimension=int(
            visualization_config.get(
                "max_detection_dimension",
                2000,
            )
        ),
    )

    print(
        "Dense landmarks detected:",
        len(
            dense_landmarks
        ),
    )

    dense_figure = (
        create_dense_landmark_figure(
            image_rgb,
            dense_landmarks,
        )
    )

    region_figure = (
        create_region_figure(
            image_rgb,
            dense_landmarks,
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = report_directory(
        selected_id
    )

    mtcnn_path = (
        output_dir
        / "01_mtcnn_5point_landmarks.png"
    )

    dense_path = (
        output_dir
        / "02_dense_landmarks.png"
    )

    regions_path = (
        output_dir
        / "03_facial_regions.png"
    )

    panel_path = (
        output_dir
        / "04_landmark_pipeline_figure.png"
    )

    save_bgr(
        mtcnn_path,
        mtcnn_figure,
    )

    save_bgr(
        dense_path,
        dense_figure,
    )

    save_bgr(
        regions_path,
        region_figure,
    )

    create_report_panel(
        original_rgb=image_rgb,
        mtcnn_bgr=mtcnn_figure,
        dense_bgr=dense_figure,
        regions_bgr=region_figure,
        output_path=panel_path,
    )

    print(
        "\nSaved report figures:"
    )

    print(
        mtcnn_path
    )

    print(
        dense_path
    )

    print(
        regions_path
    )

    print(
        panel_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Visualize MTCNN and dense facial landmarks "
            "for an ND-TWINS image."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/qc.yaml"
        ),
    )

    selection = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    selection.add_argument(
        "--image-id",
        type=str,
        help=(
            "Image ID from nd_twins_metadata.csv, "
            "for example 90003d20."
        ),
    )

    selection.add_argument(
        "--image",
        type=Path,
        help=(
            "Direct path to an image/NEF file."
        ),
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        image_id=args.image_id,
        image_path_argument=args.image,
    )
