from __future__ import annotations

import argparse
import copy
import random
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


def normalize_identifier(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )


def seed_everything(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


class TwinTripletDataset(
    Dataset
):
    """
    Samples:

        anchor:
            subject A

        positive:
            another image of subject A

        co_twin:
            image of monozygotic co-twin B

        unrelated:
            subject from another family

    The dataset uses only the supplied dataframe. Therefore,
    passing DEVELOPMENT rows preserves your frozen final test.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_column: str,
        subject_column: str,
        family_column: str,
        monozygotic_group_column: str,
        transform,
        samples_per_epoch: int | None = None,
    ):
        self.frame = (
            frame
            .reset_index(
                drop=True
            )
            .copy()
        )

        self.image_column = (
            image_column
        )

        self.subject_column = (
            subject_column
        )

        self.family_column = (
            family_column
        )

        self.monozygotic_group_column = (
            monozygotic_group_column
        )

        self.transform = transform

        self.subject_rows = {
            str(
                subject
            ):
                group.index.to_list()

            for subject, group
            in self.frame.groupby(
                subject_column
            )
        }

        self.subject_family = {
            str(
                subject
            ):
                str(
                    group[
                        family_column
                    ].iloc[
                        0
                    ]
                )

            for subject, group
            in self.frame.groupby(
                subject_column
            )
        }

        self.subject_mz_group = {
            str(
                subject
            ):
                (
                    None
                    if pd.isna(
                        group[
                            monozygotic_group_column
                        ].iloc[
                            0
                        ]
                    )
                    else str(
                        group[
                            monozygotic_group_column
                        ].iloc[
                            0
                        ]
                    )
                )

            for subject, group
            in self.frame.groupby(
                subject_column
            )
        }

        mz_to_subjects = {}

        for (
            subject,
            mz_group,
        ) in self.subject_mz_group.items():

            if (
                mz_group is None
                or mz_group == ""
                or mz_group == "<NA>"
            ):
                continue

            mz_to_subjects.setdefault(
                mz_group,
                []
            ).append(
                subject
            )

        eligible = []

        for subject, rows in (
            self.subject_rows.items()
        ):
            if len(
                rows
            ) < 2:
                continue

            mz_group = self.subject_mz_group[
                subject
            ]

            if (
                mz_group is None
                or mz_group not in mz_to_subjects
            ):
                continue

            twin_subjects = [
                other
                for other
                in mz_to_subjects[
                    mz_group
                ]
                if (
                    other != subject
                    and other
                    in self.subject_rows
                )
            ]

            if not twin_subjects:
                continue

            eligible.append(
                subject
            )

        if not eligible:
            raise ValueError(
                "No eligible twin subjects were found."
            )

        self.eligible_subjects = (
            eligible
        )

        self.mz_to_subjects = (
            mz_to_subjects
        )

        self.all_subjects = list(
            self.subject_rows.keys()
        )

        self.samples_per_epoch = (
            int(
                samples_per_epoch
            )
            if samples_per_epoch
            is not None
            else max(
                len(
                    self.frame
                ),
                len(
                    eligible
                )
                * 20,
            )
        )

    def __len__(
        self,
    ) -> int:
        return (
            self.samples_per_epoch
        )

    def _read(
        self,
        row_index: int,
    ):
        path = Path(
            str(
                self.frame.loc[
                    row_index,
                    self.image_column,
                ]
            )
        )

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        return self.transform(
            image
        )

    def __getitem__(
        self,
        index: int,
    ):
        subject = random.choice(
            self.eligible_subjects
        )

        subject_rows = (
            self.subject_rows[
                subject
            ]
        )

        anchor_row, positive_row = (
            random.sample(
                subject_rows,
                2,
            )
        )

        mz_group = (
            self.subject_mz_group[
                subject
            ]
        )

        twin_subjects = [
            other
            for other
            in self.mz_to_subjects[
                mz_group
            ]
            if (
                other != subject
                and other
                in self.subject_rows
            )
        ]

        twin_subject = random.choice(
            twin_subjects
        )

        twin_row = random.choice(
            self.subject_rows[
                twin_subject
            ]
        )

        family = (
            self.subject_family[
                subject
            ]
        )

        unrelated_candidates = [
            other
            for other
            in self.all_subjects
            if (
                self.subject_family[
                    other
                ]
                != family
            )
        ]

        unrelated_subject = (
            random.choice(
                unrelated_candidates
            )
        )

        unrelated_row = random.choice(
            self.subject_rows[
                unrelated_subject
            ]
        )

        return (
            self._read(
                anchor_row
            ),
            self._read(
                positive_row
            ),
            self._read(
                twin_row
            ),
            self._read(
                unrelated_row
            ),
        )


def build_transform(
    image_size: int,
    training: bool,
):
    operations = [
        transforms.Resize(
            (
                image_size,
                image_size,
            )
        )
    ]

    if training:
        operations.extend(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.12,
                            contrast=0.12,
                            saturation=0.06,
                        )
                    ],
                    p=0.35,
                ),

                transforms.RandomAffine(
                    degrees=3,
                    translate=(
                        0.015,
                        0.015,
                    ),
                    scale=(
                        0.98,
                        1.02,
                    ),
                ),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),

            # FaceNet-style [-1, 1] scaling.
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

    return transforms.Compose(
        operations
    )


def configure_trainable_layers(
    model: torch.nn.Module,
    module_names: list[str],
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = (
            False
        )

    matched = set()

    for name, module in (
        model.named_modules()
    ):
        for target in module_names:
            if (
                name == target
                or name.startswith(
                    f"{target}."
                )
            ):
                matched.add(
                    target
                )

                for parameter in (
                    module.parameters()
                ):
                    parameter.requires_grad = (
                        True
                    )

    missing = (
        set(
            module_names
        )
        - matched
    )

    if missing:
        print(
            "[WARNING] Could not match "
            f"trainable modules: "
            f"{sorted(missing)}"
        )


def cosine_distance(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    return (
        1.0
        - F.cosine_similarity(
            first,
            second,
            dim=1,
        )
    )


def calculate_loss(
    student_embeddings: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    teacher_embeddings: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    config: dict,
):
    (
        anchor,
        positive,
        twin,
        unrelated,
    ) = student_embeddings

    (
        teacher_anchor,
        teacher_positive,
        teacher_twin,
        teacher_unrelated,
    ) = teacher_embeddings

    d_ap = cosine_distance(
        anchor,
        positive,
    )

    d_at = cosine_distance(
        anchor,
        twin,
    )

    d_an = cosine_distance(
        anchor,
        unrelated,
    )

    twin_loss = F.relu(
        d_ap
        - d_at
        + float(
            config[
                "twin_margin"
            ]
        )
    ).mean()

    unrelated_loss = F.relu(
        d_ap
        - d_an
        + float(
            config[
                "unrelated_margin"
            ]
        )
    ).mean()

    positive_compactness = (
        d_ap.mean()
    )

    preserve_losses = []

    for (
        student,
        teacher,
    ) in zip(
        student_embeddings,
        teacher_embeddings,
    ):
        preserve_losses.append(
            (
                1.0
                - F.cosine_similarity(
                    student,
                    teacher,
                    dim=1,
                )
            ).mean()
        )

    preserve_loss = torch.stack(
        preserve_losses
    ).mean()

    total = (
        float(
            config[
                "twin_loss_weight"
            ]
        )
        * twin_loss

        + float(
            config[
                "unrelated_loss_weight"
            ]
        )
        * unrelated_loss

        + float(
            config[
                "positive_compactness_weight"
            ]
        )
        * positive_compactness

        + float(
            config[
                "preserve_pretrained_weight"
            ]
        )
        * preserve_loss
    )

    return {
        "total":
            total,

        "twin":
            twin_loss,

        "unrelated":
            unrelated_loss,

        "positive":
            positive_compactness,

        "preserve":
            preserve_loss,
    }


def forward_four(
    model,
    anchor,
    positive,
    twin,
    unrelated,
):
    combined = torch.cat(
        [
            anchor,
            positive,
            twin,
            unrelated,
        ],
        dim=0,
    )

    embeddings = model(
        combined
    )

    embeddings = F.normalize(
        embeddings,
        dim=1,
    )

    return torch.chunk(
        embeddings,
        chunks=4,
        dim=0,
    )


@torch.no_grad()
def evaluate_loss(
    model,
    teacher,
    loader,
    device,
    loss_config,
):
    model.eval()
    teacher.eval()

    totals = []

    for (
        anchor,
        positive,
        twin,
        unrelated,
    ) in loader:

        anchor = anchor.to(
            device,
            non_blocking=True,
        )

        positive = positive.to(
            device,
            non_blocking=True,
        )

        twin = twin.to(
            device,
            non_blocking=True,
        )

        unrelated = unrelated.to(
            device,
            non_blocking=True,
        )

        student_embeddings = (
            forward_four(
                model,
                anchor,
                positive,
                twin,
                unrelated,
            )
        )

        teacher_embeddings = (
            forward_four(
                teacher,
                anchor,
                positive,
                twin,
                unrelated,
            )
        )

        losses = calculate_loss(
            student_embeddings,
            teacher_embeddings,
            loss_config,
        )

        totals.append(
            float(
                losses[
                    "total"
                ].item()
            )
        )

    return float(
        np.mean(
            totals
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Train on all DEVELOPMENT using the configured "
            "number of epochs. Use only after hyperparameters "
            "and epoch count are frozen."
        ),
    )

    args = parser.parse_args()

    config = load_yaml(
        resolve_path(
            args.config
        )
    )

    training_config = config[
        "twin_aware_facenet"
    ]

    seed = int(
        training_config[
            "seed"
        ]
    )

    seed_everything(
        seed
    )

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

    columns = config[
        "columns"
    ]

    image_column = training_config[
        "image_column"
    ]

    subject_column = columns[
        "subject_id"
    ]

    family_column = columns[
        "family_id"
    ]

    monozygotic_group_column = columns[
        "monozygotic_group_id"
    ]

    split_column = columns[
        "split"
    ]

    fold_column = columns[
        "fold"
    ]

    for column in [
        subject_column,
        family_column,
        monozygotic_group_column,
    ]:
        if column in frame.columns:
            frame[
                column
            ] = normalize_identifier(
                frame[
                    column
                ]
            )

    development = frame.loc[
        frame[
            split_column
        ]
        == "development"
    ].copy()

    validation_fold = (
        training_config.get(
            "validation_fold",
            None,
        )
    )

    if (
        args.final
        or validation_fold is None
    ):
        train_frame = development
        validation_frame = None

    else:
        fold_values = pd.to_numeric(
            development[
                fold_column
            ],
            errors="coerce",
        )

        validation_fold = int(
            validation_fold
        )

        train_frame = development.loc[
            fold_values
            != validation_fold
        ].copy()

        validation_frame = (
            development.loc[
                fold_values
                == validation_fold
            ].copy()
        )

    image_size = int(
        training_config[
            "image_size"
        ]
    )

    train_dataset = TwinTripletDataset(
        frame=train_frame,
        image_column=image_column,
        subject_column=subject_column,
        family_column=family_column,
        monozygotic_group_column=(
            monozygotic_group_column
        ),
        transform=build_transform(
            image_size,
            training=True,
        ),
    )

    validation_dataset = None

    if validation_frame is not None:
        validation_dataset = TwinTripletDataset(
            frame=validation_frame,
            image_column=image_column,
            subject_column=subject_column,
            family_column=family_column,
            monozygotic_group_column=(
                monozygotic_group_column
            ),
            transform=build_transform(
                image_size,
                training=False,
            ),
            samples_per_epoch=max(
                len(
                    validation_frame
                ),
                2000,
            ),
        )

    batch_size = int(
        training_config[
            "batch_size"
        ]
    )

    num_workers = int(
        training_config[
            "num_workers"
        ]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
        drop_last=True,
    )

    validation_loader = None

    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(
                num_workers > 0
            ),
            drop_last=False,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    student = InceptionResnetV1(
        pretrained=str(
            training_config[
                "pretrained"
            ]
        ),
        classify=False,
    ).to(
        device
    )

    teacher = copy.deepcopy(
        student
    ).to(
        device
    )

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = (
            False
        )

    configure_trainable_layers(
        student,
        list(
            training_config[
                "unfreeze_modules"
            ]
        ),
    )

    trainable_parameters = [
        parameter
        for parameter
        in student.parameters()
        if parameter.requires_grad
    ]

    print(
        f"Trainable parameter tensors: "
        f"{len(trainable_parameters)}"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(
            training_config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            training_config[
                "weight_decay"
            ]
        ),
    )

    epochs = int(
        training_config[
            "epochs"
        ]
    )

    checkpoints_dir = resolve_path(
        config[
            "output"
        ][
            "checkpoints_dir"
        ]
    )

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_name = (
        training_config[
            "final_checkpoint_name"
        ]
        if args.final
        else training_config[
            "checkpoint_name"
        ]
    )

    checkpoint_path = (
        checkpoints_dir
        / checkpoint_name
    )

    best_validation = float(
        "inf"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        student.train()

        running = []

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch}/{epochs}"
            ),
            unit="batch",
        )

        for (
            anchor,
            positive,
            twin,
            unrelated,
        ) in progress:

            anchor = anchor.to(
                device,
                non_blocking=True,
            )

            positive = positive.to(
                device,
                non_blocking=True,
            )

            twin = twin.to(
                device,
                non_blocking=True,
            )

            unrelated = unrelated.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            student_embeddings = (
                forward_four(
                    student,
                    anchor,
                    positive,
                    twin,
                    unrelated,
                )
            )

            with torch.no_grad():
                teacher_embeddings = (
                    forward_four(
                        teacher,
                        anchor,
                        positive,
                        twin,
                        unrelated,
                    )
                )

            losses = calculate_loss(
                student_embeddings,
                teacher_embeddings,
                training_config,
            )

            losses[
                "total"
            ].backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=5.0,
            )

            optimizer.step()

            running.append(
                float(
                    losses[
                        "total"
                    ].item()
                )
            )

            progress.set_postfix(
                loss=(
                    f"{np.mean(running[-50:]):.4f}"
                ),
                twin=(
                    f"{losses['twin'].item():.4f}"
                ),
            )

        train_loss = float(
            np.mean(
                running
            )
        )

        if validation_loader is not None:
            validation_loss = (
                evaluate_loss(
                    model=student,
                    teacher=teacher,
                    loader=validation_loader,
                    device=device,
                    loss_config=training_config,
                )
            )

            print(
                f"Epoch {epoch}: "
                f"train={train_loss:.4f} | "
                f"validation={validation_loss:.4f}"
            )

            if validation_loss < best_validation:
                best_validation = (
                    validation_loss
                )

                torch.save(
                    {
                        "state_dict":
                            student.state_dict(),

                        "epoch":
                            epoch,

                        "validation_loss":
                            validation_loss,

                        "config":
                            training_config,
                    },
                    checkpoint_path,
                )

                print(
                    f"Saved best checkpoint: "
                    f"{checkpoint_path}"
                )

        else:
            print(
                f"Epoch {epoch}: "
                f"train={train_loss:.4f}"
            )

            torch.save(
                {
                    "state_dict":
                        student.state_dict(),

                    "epoch":
                        epoch,

                    "config":
                        training_config,
                },
                checkpoint_path,
            )

    print(
        f"\nTraining complete.\n"
        f"Checkpoint:\n{checkpoint_path}"
    )


if __name__ == "__main__":
    main()
