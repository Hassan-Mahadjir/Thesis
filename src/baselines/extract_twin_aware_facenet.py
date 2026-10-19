from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


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


class ImageDataset(
    Dataset
):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_column: str,
        image_size: int,
    ):
        self.frame = (
            frame
            .reset_index(
                drop=True
            )
        )

        self.image_column = (
            image_column
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        image_size,
                        image_size,
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=(
                        0.5,
                        0.5,
                        0.5,
                    ),
                    std=(
                        0.5,
                        0.5,
                        0.5,
                    ),
                ),
            ]
        )

    def __len__(
        self,
    ):
        return len(
            self.frame
        )

    def __getitem__(
        self,
        index: int,
    ):
        path = Path(
            str(
                self.frame.loc[
                    index,
                    self.image_column,
                ]
            )
        )

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        return (
            index,
            self.transform(
                image
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional explicit checkpoint path. "
            "Otherwise the configured best checkpoint is used."
        ),
    )

    parser.add_argument(
        "--feature-name",
        default="twin_aware_facenet",
    )

    args = parser.parse_args()

    config = load_yaml(
        resolve_path(
            args.config
        )
    )

    model_config = config[
        "twin_aware_facenet"
    ]

    assignments_path = resolve_path(
        config[
            "input"
        ][
            "assignments_csv"
        ]
    )

    frame = pd.read_csv(
        assignments_path,
        low_memory=False,
    )

    image_column = (
        model_config[
            "image_column"
        ]
    )

    image_id_column = (
        config[
            "columns"
        ][
            "image_id"
        ]
    )

    checkpoints_dir = resolve_path(
        config[
            "output"
        ][
            "checkpoints_dir"
        ]
    )

    checkpoint_path = (
        resolve_path(
            args.checkpoint
        )
        if args.checkpoint
        else (
            checkpoints_dir
            / model_config[
                "checkpoint_name"
            ]
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = InceptionResnetV1(
        pretrained=None,
        classify=False,
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    dataset = ImageDataset(
        frame=frame,
        image_column=image_column,
        image_size=int(
            model_config[
                "image_size"
            ]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=int(
            model_config[
                "num_workers"
            ]
        ),
        pin_memory=True,
        persistent_workers=(
            int(
                model_config[
                    "num_workers"
                ]
            )
            > 0
        ),
    )

    features_root = resolve_path(
        config[
            "output"
        ][
            "features_dir"
        ]
    )

    output_dir = (
        features_root
        / args.feature_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = np.lib.format.open_memmap(
        output_dir
        / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(
            len(frame),
            512,
        ),
    )

    with torch.no_grad():
        for indices, images in tqdm(
            loader,
            desc="Extracting twin-aware FaceNet",
            unit="batch",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            embeddings = model(
                images
            )

            embeddings = F.normalize(
                embeddings,
                dim=1,
            )

            embeddings = (
                embeddings
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            features[
                indices.numpy()
            ] = embeddings

    features.flush()

    index = pd.DataFrame(
        {
            "row_index":
                np.arange(
                    len(frame)
                ),

            "image_id":
                frame[
                    image_id_column
                ].astype(
                    str
                ),

            "image_path":
                frame[
                    image_column
                ].astype(
                    str
                ),
        }
    )

    index.to_csv(
        output_dir
        / "index.csv",
        index=False,
    )

    metadata = {
        "feature_name":
            args.feature_name,

        "method":
            "Twin-aware fine-tuned FaceNet",

        "base_model":
            "InceptionResnetV1 / VGGFace2",

        "checkpoint":
            str(
                checkpoint_path
            ),

        "embedding_dimension":
            512,

        "matching_metric":
            "cosine",
    }

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nSaved features to:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
