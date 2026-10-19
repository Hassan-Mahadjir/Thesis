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


class ImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_column: str,
        image_size: int,
    ):
        self.frame = (
            frame.reset_index(
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

    def __len__(self):
        return len(self.frame)

    def __getitem__(
        self,
        index: int,
    ):
        raw_path = str(
            self.frame.loc[
                index,
                self.image_column,
            ]
        )

        path = Path(raw_path)

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            ).resolve()

        image = Image.open(
            path
        ).convert("RGB")

        return (
            index,
            self.transform(image),
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--feature-name",
        default="facenet_pytorch_vggface2",
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

    frame = pd.read_csv(
        resolve_path(
            config[
                "input"
            ][
                "assignments_csv"
            ]
        ),
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = InceptionResnetV1(
        pretrained=str(
            model_config[
                "pretrained"
            ]
        ),
        classify=False,
    ).to(device)

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

    num_workers = int(
        model_config[
            "num_workers"
        ]
    )

    loader = DataLoader(
        dataset,
        batch_size=int(
            model_config[
                "evaluation_batch_size"
            ]
        ),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            num_workers > 0
        ),
    )

    output_dir = (
        resolve_path(
            config[
                "output"
            ][
                "features_dir"
            ]
        )
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
            desc="Extracting untouched FaceNet-VGGFace2",
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

            features[
                indices.numpy()
            ] = (
                embeddings
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

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
                ].astype(str),

            "image_path":
                frame[
                    image_column
                ].astype(str),
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
            "Untouched facenet-pytorch InceptionResnetV1",

        "pretrained":
            model_config[
                "pretrained"
            ],

        "image_column":
            image_column,

        "image_size":
            int(
                model_config[
                    "image_size"
                ]
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
