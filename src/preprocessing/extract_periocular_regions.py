from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from src.global_local.common import (
    normalize_identifier,
    resolve_path,
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("._") or "image"


def crop_periocular(
    image: Image.Image,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    output_width: int,
    output_height: int,
) -> Image.Image:
    width, height = image.size

    left = int(round(width * x_min))
    top = int(round(height * y_min))
    right = int(round(width * x_max))
    bottom = int(round(height * y_max))

    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))

    crop = image.crop((left, top, right, bottom))

    return crop.resize(
        (output_width, output_height),
        Image.Resampling.LANCZOS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small test run. Omit for the full dataset.",
    )

    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))

    assignments_path = resolve_path(
        config["input"]["assignments_csv"]
    )

    manifest_path = resolve_path(
        config["input"]["periocular_manifest_csv"]
    )

    output_dir = resolve_path(
        config["output"]["periocular_dir"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    columns = config["columns"]
    region_config = config["periocular"]

    image_id_column = columns["image_id"]
    source_column = region_config["source_image_column"]

    frame = pd.read_csv(assignments_path, low_memory=False)
    frame[image_id_column] = normalize_identifier(frame[image_id_column])

    if source_column not in frame.columns:
        raise ValueError(
            f"Assignments CSV does not contain '{source_column}'."
        )

    if args.limit is not None:
        frame = frame.head(args.limit).copy()

    rows = []

    for _, row in tqdm(
        frame.iterrows(),
        total=len(frame),
        desc="Extracting periocular regions",
        unit="image",
    ):
        image_id = str(row[image_id_column])
        source_value = row[source_column]

        result = {
            image_id_column: image_id,
            "source_image_path": str(source_value),
            "periocular_path": "",
            "status": "failed",
            "error": "",
        }

        if pd.isna(source_value) or not str(source_value).strip():
            result["error"] = "missing_source_path"
            rows.append(result)
            continue

        source_path = resolve_path(str(source_value))

        suffix = "." + str(region_config["image_format"]).lower().lstrip(".")
        output_path = output_dir / f"{safe_stem(image_id)}{suffix}"

        try:
            if not output_path.exists() or args.overwrite:
                with Image.open(source_path) as source_image:
                    source_image = source_image.convert("RGB")

                    crop = crop_periocular(
                        source_image,
                        x_min=float(region_config["x_min"]),
                        y_min=float(region_config["y_min"]),
                        x_max=float(region_config["x_max"]),
                        y_max=float(region_config["y_max"]),
                        output_width=int(region_config["output_width"]),
                        output_height=int(region_config["output_height"]),
                    )

                    temp_path = output_path.with_name(
                        output_path.stem + ".tmp" + output_path.suffix
                    )

                    crop.save(temp_path)
                    os.replace(temp_path, output_path)

            result["periocular_path"] = str(
                output_path.relative_to(
                    Path(__file__).resolve().parents[2]
                )
            ).replace("\\", "/")
            result["status"] = "ok"

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(result)

    manifest = pd.DataFrame(rows)

    # If this was a limited test, do not overwrite a complete manifest.
    if args.limit is not None:
        test_path = manifest_path.with_name(
            manifest_path.stem + "_test.csv"
        )
        manifest.to_csv(test_path, index=False)
        print(f"\nTest manifest written to:\n{test_path}")
    else:
        manifest.to_csv(manifest_path, index=False)
        print(f"\nManifest written to:\n{manifest_path}")

    ok = int((manifest["status"] == "ok").sum())
    failed = int((manifest["status"] != "ok").sum())

    print(f"Successful: {ok}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
