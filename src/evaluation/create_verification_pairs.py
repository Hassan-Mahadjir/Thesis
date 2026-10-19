from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def sample_pairs(candidates: list[tuple[str, str]], maximum: int | None, rng: np.random.Generator) -> list[tuple[str, str]]:
    if maximum is None or len(candidates) <= maximum:
        return candidates
    selected = rng.choice(len(candidates), size=maximum, replace=False)
    return [candidates[int(index)] for index in selected]


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


def create_genuine_pairs(df: pd.DataFrame, image_column: str, subject_column: str, family_column: str,
                         maximum_per_subject: int, rng: np.random.Generator) -> list[dict]:
    records = []
    for subject_id, group in df.groupby(subject_column):
        image_ids = group[image_column].astype(str).tolist()
        if len(image_ids) < 2:
            continue
        candidates = [canonical_pair(a, b) for a, b in itertools.combinations(image_ids, 2)]
        candidates = sample_pairs(candidates, maximum_per_subject, rng)
        family_id = str(group[family_column].iloc[0])
        for first, second in candidates:
            records.append({
                "image_id_1": first,
                "image_id_2": second,
                "subject_id_1": str(subject_id),
                "subject_id_2": str(subject_id),
                "family_id_1": family_id,
                "family_id_2": family_id,
                "pair_type": "genuine",
                "label": 1,
            })
    return records


def create_cotwin_pairs(df: pd.DataFrame, image_column: str, subject_column: str, family_column: str,
                        monozygotic_group_column: str, maximum_per_subject_pair: int,
                        rng: np.random.Generator) -> list[dict]:
    records = []
    working = df.loc[df[monozygotic_group_column].notna()].copy()
    working[monozygotic_group_column] = normalize_identifier(working[monozygotic_group_column])
    working = working.loc[working[monozygotic_group_column].notna() & (working[monozygotic_group_column] != "")]

    for _, group in working.groupby(monozygotic_group_column):
        subjects = sorted(group[subject_column].astype(str).unique().tolist())
        if len(subjects) < 2:
            continue

        for first_subject, second_subject in itertools.combinations(subjects, 2):
            first_images = group.loc[group[subject_column].astype(str) == first_subject, image_column].astype(str).tolist()
            second_images = group.loc[group[subject_column].astype(str) == second_subject, image_column].astype(str).tolist()
            candidates = list(dict.fromkeys(
                canonical_pair(first, second)
                for first in first_images
                for second in second_images
            ))
            candidates = sample_pairs(candidates, maximum_per_subject_pair, rng)
            first_family = str(group.loc[group[subject_column].astype(str) == first_subject, family_column].iloc[0])
            second_family = str(group.loc[group[subject_column].astype(str) == second_subject, family_column].iloc[0])

            for first, second in candidates:
                records.append({
                    "image_id_1": first,
                    "image_id_2": second,
                    "subject_id_1": first_subject,
                    "subject_id_2": second_subject,
                    "family_id_1": first_family,
                    "family_id_2": second_family,
                    "pair_type": "co_twin",
                    "label": 0,
                })
    return records


def create_unrelated_pairs(df: pd.DataFrame, image_column: str, subject_column: str, family_column: str,
                           number_of_pairs: int, rng: np.random.Generator) -> list[dict]:
    if number_of_pairs <= 0:
        return []

    working = df[[image_column, subject_column, family_column]].drop_duplicates(subset=[image_column]).reset_index(drop=True)
    records, seen = [], set()
    maximum_attempts = max(number_of_pairs * 100, 10_000)
    attempts = 0

    while len(records) < number_of_pairs and attempts < maximum_attempts:
        attempts += 1
        first_index = int(rng.integers(0, len(working)))
        second_index = int(rng.integers(0, len(working)))
        if first_index == second_index:
            continue

        first = working.iloc[first_index]
        second = working.iloc[second_index]
        first_subject, second_subject = str(first[subject_column]), str(second[subject_column])
        first_family, second_family = str(first[family_column]), str(second[family_column])
        if first_subject == second_subject or first_family == second_family:
            continue

        pair = canonical_pair(str(first[image_column]), str(second[image_column]))
        if pair in seen:
            continue
        seen.add(pair)
        records.append({
            "image_id_1": pair[0],
            "image_id_2": pair[1],
            "subject_id_1": first_subject,
            "subject_id_2": second_subject,
            "family_id_1": first_family,
            "family_id_2": second_family,
            "pair_type": "unrelated",
            "label": 0,
        })

    if len(records) < number_of_pairs:
        raise RuntimeError(f"Could not create requested unrelated pairs: {len(records):,}/{number_of_pairs:,}")
    return records


def create_partition_pairs(partition_df: pd.DataFrame, config: dict, protocol_name: str,
                           partition_name: str, seed: int) -> pd.DataFrame:
    columns = config["columns"]
    pair_config = config["pair_generation"]
    image_column = columns["image_id"]
    subject_column = columns["subject_id"]
    family_column = columns["family_id"]
    monozygotic_group_column = columns["monozygotic_group_id"]
    rng = np.random.default_rng(seed)

    genuine = create_genuine_pairs(
        partition_df, image_column, subject_column, family_column,
        int(pair_config["max_genuine_pairs_per_subject"]), rng,
    )
    cotwin = create_cotwin_pairs(
        partition_df, image_column, subject_column, family_column,
        monozygotic_group_column, int(pair_config["max_cotwin_pairs_per_subject_pair"]), rng,
    )
    unrelated_target = int(round(float(pair_config.get("unrelated_pairs_multiplier", 1.0)) * max(len(genuine), len(cotwin), 1)))
    unrelated = create_unrelated_pairs(
        partition_df, image_column, subject_column, family_column,
        unrelated_target, rng,
    )

    result = pd.DataFrame(genuine + cotwin + unrelated)
    result.insert(0, "partition", partition_name)
    result.insert(0, "protocol", protocol_name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    protocol_dir = resolve_path(config["input"]["protocol_dir"])
    pairs_dir = resolve_path(config["output"]["pairs_dir"])
    pairs_dir.mkdir(parents=True, exist_ok=True)
    columns = config["columns"]
    required = [columns["image_id"], columns["subject_id"], columns["family_id"], columns["monozygotic_group_id"], columns["split"]]
    base_seed = int(config["pair_generation"]["seed"])

    for protocol_index, protocol_name in enumerate(config["protocols"]):
        path = protocol_dir / f"{protocol_name}.csv"
        df = pd.read_csv(path, low_memory=False)
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{protocol_name} missing columns: {missing}")

        for column in [columns["image_id"], columns["subject_id"], columns["family_id"]]:
            df[column] = normalize_identifier(df[column])

        out_dir = pairs_dir / protocol_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{protocol_name}")
        for partition_index, partition_name in enumerate(["development", "final_test"]):
            partition_df = df.loc[df[columns["split"]] == partition_name].copy()
            seed = base_seed + protocol_index * 1000 + partition_index * 100
            pairs = create_partition_pairs(partition_df, config, protocol_name, partition_name, seed)
            pairs.to_csv(out_dir / f"{partition_name}.csv", index=False)
            print(f"  {partition_name}: {len(pairs):,} | {pairs['pair_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
