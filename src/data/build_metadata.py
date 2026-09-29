from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyreadr
import yaml


# ============================================================
# RELATIONSHIP DEFINITIONS
# ============================================================

STANDARD_IDENTICAL_TYPES = {
    "Identical",
}

MONOZYGOTIC_TYPES = {
    "Identical",
    "IdenticalMirror",
    "IdenticalTriplet",
}

TWIN_TYPES = {
    "Identical",
    "IdenticalMirror",
    "IdenticalTriplet",
    "Fraternal",
    "UnknownTwinType",
}

PRIMARY_RELATIONSHIP_PRIORITY = [
    "IdenticalTriplet",
    "IdenticalMirror",
    "Identical",
    "Fraternal",
    "UnknownTwinType",
    "Sibling",
]


# ============================================================
# EXTENDED METADATA FIELDS
#
# We intentionally do NOT import duplicate semantic fields
# such as gender, yaw, date, glasses, etc.
#
# Those already exist in image-metadata-2.0.csv.
# ============================================================

XML_FIELD_MAP = {
    "id": "xml_record_id",
    "recordingid": "recording_id",
    "shotid": "shot_id",
    "fileid": "xml_file_id",
    "format": "source_format",
    "stageid": "stage_id",
    "collectionid": "collection_id",
    "environmentid": "environment_id",
    "sensorid": "sensor_id",
    "illuminantid": "illuminant_id",
    "illuminantid1": "illuminant_id_1",
    "state": "xml_state",
    "tag_list": "tag_list",
}


# ============================================================
# CONFIG
# ============================================================

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_path(path: str) -> Path:
    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    return project_root() / candidate


def dataset_path(dataset_root: Path, path: str) -> Path:
    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    return dataset_root / candidate


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return pd.NA

    return str(value).strip()


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            df[column] = df[column].map(clean_text)

    return df


def normalize_subject_id(value) -> str | None:
    """
    Examples
    --------
    nd1S90008 -> 90008
    90008     -> 90008
    """
    if pd.isna(value):
        return None

    match = re.search(
        r"(\d{5})$",
        str(value).strip(),
    )

    if match is None:
        return None

    return match.group(1)


def subject_from_filename(filename) -> str | None:
    """
    Example
    -------
    90003d17.nef -> 90003
    """
    if pd.isna(filename):
        return None

    match = re.match(
        r"^(\d+)d",
        str(filename).strip(),
    )

    if match is None:
        return None

    return match.group(1)


def semicolon_join(values: Iterable[str]) -> str:
    cleaned = sorted(
        {
            str(value)
            for value in values
            if value is not None
            and not pd.isna(value)
        }
    )

    return ";".join(cleaned)


# ============================================================
# UNION-FIND
#
# Used to construct family/twin components from the
# relationship graph.
# ============================================================

class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value

        if self.parent[value] != value:
            self.parent[value] = self.find(
                self.parent[value]
            )

        return self.parent[value]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


def connected_components(
    edges: pd.DataFrame,
    allowed_types: set[str] | None = None,
) -> list[list[str]]:
    if allowed_types is None:
        subset = edges
    else:
        subset = edges[
            edges["relationship_type"].isin(
                allowed_types
            )
        ]

    uf = UnionFind()
    subjects: set[str] = set()

    for row in subset.itertuples(index=False):
        subject_a = row.subject_id1
        subject_b = row.subject_id2

        uf.union(subject_a, subject_b)

        subjects.add(subject_a)
        subjects.add(subject_b)

    groups: dict[str, list[str]] = defaultdict(list)

    for subject in subjects:
        groups[uf.find(subject)].append(subject)

    components = [
        sorted(members)
        for members in groups.values()
    ]

    return sorted(
        components,
        key=lambda members: members[0],
    )


def make_group_lookup(
    components: list[list[str]],
    prefix: str,
    id_column: str,
    size_column: str,
    members_column: str,
    label_column: str,
) -> pd.DataFrame:
    records = []

    for group_index, members in enumerate(
        components,
        start=1,
    ):
        group_id = f"{prefix}_{group_index:03d}"

        for member_index, subject_id in enumerate(members):
            records.append(
                {
                    "subject_id": subject_id,
                    id_column: group_id,
                    size_column: len(members),
                    members_column: ";".join(members),
                    label_column: chr(
                        ord("A") + member_index
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# IMAGE METADATA
# ============================================================

def load_image_metadata(
    csv_path: Path,
    expected_rows: int | None,
    expected_subjects: int | None,
    strict: bool,
) -> pd.DataFrame:
    print(f"Loading image metadata: {csv_path}")

    images = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    images.columns = (
        images.columns
        .astype(str)
        .str.strip()
    )

    # Remove empty exported columns such as "Unnamed: 9".
    images = images.loc[
        :,
        ~images.columns.str.startswith("Unnamed:")
    ].copy()

    images = clean_text_columns(images)

    required_columns = {
        "filename",
        "date",
        "gender",
        "race",
        "yearofbirth",
        "weather",
        "glasses",
        "expression",
        "yaw",
    }

    missing = required_columns - set(images.columns)

    if missing:
        raise ValueError(
            "image-metadata CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    images = images.rename(
        columns={
            "date": "capture_date",
            "yearofbirth": "year_of_birth",
        }
    )

    images["subject_id"] = (
        images["filename"]
        .apply(subject_from_filename)
    )

    images["image_id"] = (
        images["filename"]
        .map(lambda value: Path(str(value)).stem)
    )

    images["capture_date"] = pd.to_datetime(
        images["capture_date"],
        errors="coerce",
    )

    images["capture_year"] = (
        images["capture_date"].dt.year
    )

    images["year_of_birth"] = pd.to_numeric(
        images["year_of_birth"],
        errors="coerce",
    )

    images["age_at_capture"] = (
        images["capture_year"]
        - images["year_of_birth"]
    )

    images["age_group_40"] = (
        images["age_at_capture"]
        .map(
            lambda age:
                pd.NA
                if pd.isna(age)
                else "<=40"
                if age <= 40
                else ">40"
        )
    )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    if images["subject_id"].isna().any():
        bad = images.loc[
            images["subject_id"].isna(),
            "filename",
        ].tolist()

        raise ValueError(
            "Could not derive subject_id from filenames: "
            f"{bad[:10]}"
        )

    if images["image_id"].duplicated().any():
        duplicate_ids = (
            images.loc[
                images["image_id"].duplicated(
                    keep=False
                ),
                "image_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate image IDs found: "
            f"{duplicate_ids[:20]}"
        )

    if strict and expected_rows is not None:
        if len(images) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} image rows, "
                f"found {len(images)}."
            )

    if strict and expected_subjects is not None:
        actual_subjects = images[
            "subject_id"
        ].nunique()

        if actual_subjects != expected_subjects:
            raise ValueError(
                f"Expected {expected_subjects} subjects, "
                f"found {actual_subjects}."
            )

    print(
        f"Loaded {len(images):,} images from "
        f"{images['subject_id'].nunique():,} subjects."
    )

    return images


# ============================================================
# RELATIONSHIPS
# ============================================================

def load_relationship_edges(
    rdata_path: Path,
    object_name: str,
    dataset_subjects: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading relationship metadata: {rdata_path}")

    objects = pyreadr.read_r(rdata_path)

    if object_name not in objects:
        raise KeyError(
            f"RData object '{object_name}' not found. "
            f"Available objects: {list(objects.keys())}"
        )

    pairs = objects[object_name].copy()

    pairs = clean_text_columns(pairs)

    required = {
        "subject_id1",
        "subject_id2",
        "relationship_type",
    }

    missing = required - set(pairs.columns)

    if missing:
        raise ValueError(
            "TwinsPairTable is missing columns: "
            + ", ".join(sorted(missing))
        )

    pairs["subject_id1"] = (
        pairs["subject_id1"]
        .apply(normalize_subject_id)
        .astype("string")
    )

    pairs["subject_id2"] = (
        pairs["subject_id2"]
        .apply(normalize_subject_id)
        .astype("string")
    )

    pairs["relationship_type"] = (
        pairs["relationship_type"]
        .map(clean_text)
    )

    pairs = pairs[
        pairs["subject_id1"].notna()
        & pairs["subject_id2"].notna()
        & pairs["relationship_type"].notna()
    ].copy()

    pairs = pairs[
        pairs["subject_id1"]
        != pairs["subject_id2"]
    ].copy()

    # --------------------------------------------------------
    # Make relationships undirected.
    #
    # A->B and B->A represent the same relationship.
    # --------------------------------------------------------

    pairs["pair_low"] = pairs[
        ["subject_id1", "subject_id2"]
    ].min(axis=1)

    pairs["pair_high"] = pairs[
        ["subject_id1", "subject_id2"]
    ].max(axis=1)

    unique_edges = (
        pairs[
            [
                "pair_low",
                "pair_high",
                "relationship_type",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "pair_low": "subject_id1",
                "pair_high": "subject_id2",
            }
        )
        .reset_index(drop=True)
    )

    # Relationships referring to subjects outside the
    # current physical image release are retained in the
    # report but not used for grouping.
    external_edges = unique_edges[
        ~unique_edges["subject_id1"].isin(
            dataset_subjects
        )
        | ~unique_edges["subject_id2"].isin(
            dataset_subjects
        )
    ].copy()

    current_edges = unique_edges[
        unique_edges["subject_id1"].isin(
            dataset_subjects
        )
        & unique_edges["subject_id2"].isin(
            dataset_subjects
        )
    ].copy()

    print(
        f"Relationship edges usable in current release: "
        f"{len(current_edges):,}"
    )

    print(
        f"Relationship edges outside current release: "
        f"{len(external_edges):,}"
    )

    return current_edges, external_edges


def build_relationship_lookup(
    edges: pd.DataFrame,
    dataset_subjects: set[str],
) -> pd.DataFrame:
    family_components = connected_components(
        edges
    )

    twin_components = connected_components(
        edges,
        allowed_types=TWIN_TYPES,
    )

    monozygotic_components = connected_components(
        edges,
        allowed_types=MONOZYGOTIC_TYPES,
    )

    family_lookup = make_group_lookup(
        family_components,
        prefix="FAMILY",
        id_column="family_group_id",
        size_column="family_group_size",
        members_column="family_member_ids",
        label_column="family_member_label",
    )

    twin_lookup = make_group_lookup(
        twin_components,
        prefix="TWIN",
        id_column="twin_group_id",
        size_column="twin_group_size",
        members_column="twin_member_ids",
        label_column="twin_member_label",
    )

    mono_lookup = make_group_lookup(
        monozygotic_components,
        prefix="MONOZYGOTIC",
        id_column="monozygotic_group_id",
        size_column="monozygotic_group_size",
        members_column="monozygotic_member_ids",
        label_column="monozygotic_member_label",
    )

    # --------------------------------------------------------
    # Direct neighbours by relationship type
    # --------------------------------------------------------

    neighbours: dict[
        str,
        dict[str, set[str]]
    ] = defaultdict(
        lambda: defaultdict(set)
    )

    for row in edges.itertuples(index=False):
        a = row.subject_id1
        b = row.subject_id2
        relation = row.relationship_type

        neighbours[a][relation].add(b)
        neighbours[b][relation].add(a)

    records = []

    for subject_id in sorted(dataset_subjects):
        relation_map = neighbours.get(
            subject_id,
            {},
        )

        relationship_types = sorted(
            relation_map.keys()
        )

        primary = "NoRelationshipRecord"

        for candidate in PRIMARY_RELATIONSHIP_PRIORITY:
            if candidate in relationship_types:
                primary = candidate
                break

        standard_identical = set()

        for relation in STANDARD_IDENTICAL_TYPES:
            standard_identical.update(
                relation_map.get(
                    relation,
                    set(),
                )
            )

        monozygotic = set()

        for relation in MONOZYGOTIC_TYPES:
            monozygotic.update(
                relation_map.get(
                    relation,
                    set(),
                )
            )

        fraternal = set(
            relation_map.get(
                "Fraternal",
                set(),
            )
        )

        unknown_twin = set(
            relation_map.get(
                "UnknownTwinType",
                set(),
            )
        )

        siblings = set(
            relation_map.get(
                "Sibling",
                set(),
            )
        )

        co_twins = (
            monozygotic
            | fraternal
            | unknown_twin
        )

        all_related = set()

        for values in relation_map.values():
            all_related.update(values)

        records.append(
            {
                "subject_id": subject_id,

                "has_relationship_record":
                    bool(relationship_types),

                "relationship_types_raw":
                    ";".join(relationship_types),

                "primary_relationship_type":
                    primary,

                "related_subject_ids":
                    semicolon_join(all_related),

                "co_twin_subject_ids":
                    semicolon_join(co_twins),

                "standard_identical_twin_subject_ids":
                    semicolon_join(
                        standard_identical
                    ),

                "monozygotic_twin_subject_ids":
                    semicolon_join(
                        monozygotic
                    ),

                "fraternal_twin_subject_ids":
                    semicolon_join(
                        fraternal
                    ),

                "unknown_twin_subject_ids":
                    semicolon_join(
                        unknown_twin
                    ),

                "sibling_subject_ids":
                    semicolon_join(
                        siblings
                    ),

                "is_standard_identical_twin":
                    bool(standard_identical),

                "is_monozygotic_twin":
                    bool(monozygotic),

                "is_identical_mirror":
                    "IdenticalMirror"
                    in relationship_types,

                "is_identical_triplet":
                    "IdenticalTriplet"
                    in relationship_types,

                "is_fraternal_twin":
                    bool(fraternal),

                "has_unknown_twin_type":
                    bool(unknown_twin),

                "has_sibling_relationship":
                    bool(siblings),

                "is_unpaired_in_pair_table":
                    not bool(relationship_types),
            }
        )

    lookup = pd.DataFrame(records)

    # --------------------------------------------------------
    # Add group information
    # --------------------------------------------------------

    for group_lookup in (
        family_lookup,
        twin_lookup,
        mono_lookup,
    ):
        if group_lookup.empty:
            continue

        lookup = lookup.merge(
            group_lookup,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # Every subject receives a FAMILY group, even subjects
    # without an entry in the pair table.
    #
    # This is critical for family-disjoint splitting.
    # --------------------------------------------------------

    no_family = lookup[
        "family_group_id"
    ].isna()

    lookup.loc[
        no_family,
        "family_group_id",
    ] = (
        lookup.loc[
            no_family,
            "subject_id",
        ]
        .map(
            lambda value:
                f"UNPAIRED_{value}"
        )
    )

    lookup.loc[
        no_family,
        "family_group_size",
    ] = 1

    lookup.loc[
        no_family,
        "family_member_ids",
    ] = lookup.loc[
        no_family,
        "subject_id",
    ]

    lookup.loc[
        no_family,
        "family_member_label",
    ] = "A"

    if not lookup["subject_id"].is_unique:
        raise ValueError(
            "Relationship lookup contains duplicate "
            "subjects."
        )

    return lookup


# ============================================================
# IMAGE CONDITION LABELS
# ============================================================

def add_condition_labels(
    master: pd.DataFrame,
) -> pd.DataFrame:
    master = master.copy()

    # Expression
    expression_map = {
        "blankstare": "neutral",
        "happiness": "smiling",
    }

    master["expression_group"] = (
        master["expression"]
        .map(
            lambda value:
                expression_map.get(
                    str(value).strip().casefold(),
                    "unknown",
                )
                if not pd.isna(value)
                else pd.NA
        )
    )

    # Illumination / environment
    def illumination(value):
        if pd.isna(value):
            return pd.NA

        value = str(value).strip().casefold()

        if value == "inside":
            return "controlled"

        if value in {
            "rain",
            "sunny",
        }:
            return "uncontrolled"

        return "unknown"

    master["illumination_group"] = (
        master["weather"].map(
            illumination
        )
    )

    # Pose
    master["yaw"] = pd.to_numeric(
        master["yaw"],
        errors="coerce",
    )

    def pose_group(yaw):
        if pd.isna(yaw):
            return pd.NA

        if yaw == 0:
            return "frontal"

        if abs(yaw) == 45:
            return "semi_profile"

        if abs(yaw) == 90:
            return "full_profile"

        return "other"

    def profile_side(yaw):
        if pd.isna(yaw):
            return pd.NA

        if yaw < 0:
            return "left"

        if yaw > 0:
            return "right"

        return "frontal"

    master["pose_group"] = (
        master["yaw"].map(
            pose_group
        )
    )

    master["profile_side"] = (
        master["yaw"].map(
            profile_side
        )
    )

    master["is_frontal"] = (
        master["yaw"].eq(0)
    )

    # Eyewear
    master["glasses_normalized"] = (
        master["glasses"]
        .map(
            lambda value:
                str(value).strip().casefold()
                if not pd.isna(value)
                else pd.NA
        )
    )

    master["has_glasses"] = (
        master["glasses_normalized"]
        .isin(
            {
                "glasses",
                "prism",
            }
        )
    )

    master["has_prism"] = (
        master["glasses_normalized"]
        .eq("prism")
        .fillna(False)
    )

    return master


# ============================================================
# PHYSICAL IMAGE INDEX
# ============================================================

def add_image_paths(
    master: pd.DataFrame,
    images_root: Path,
    valid_extensions: set[str],
) -> tuple[pd.DataFrame, int, int]:
    print(f"Indexing physical images: {images_root}")

    image_index: dict[
        tuple[str, str],
        Path
    ] = {}

    duplicate_keys = []

    for subject_folder in images_root.iterdir():
        if not subject_folder.is_dir():
            continue

        if not subject_folder.name.isdigit():
            continue

        subject_id = subject_folder.name

        for image_path in subject_folder.iterdir():
            if not image_path.is_file():
                continue

            if (
                image_path.suffix.lower()
                not in valid_extensions
            ):
                continue

            key = (
                subject_id,
                image_path.stem.lower(),
            )

            if key in image_index:
                duplicate_keys.append(key)
                continue

            image_index[key] = image_path

    if duplicate_keys:
        raise ValueError(
            "Duplicate physical image keys found: "
            f"{duplicate_keys[:20]}"
        )

    def resolve(row):
        key = (
            str(row.subject_id),
            Path(
                str(row.filename)
            ).stem.lower(),
        )

        path = image_index.get(key)

        return (
            str(path.resolve())
            if path is not None
            else pd.NA
        )

    master = master.copy()

    master["image_path"] = [
        resolve(row)
        for row in master.itertuples(
            index=False
        )
    ]

    master["file_exists"] = (
        master["image_path"].notna()
    )

    metadata_keys = {
        (
            str(row.subject_id),
            Path(
                str(row.filename)
            ).stem.lower(),
        )
        for row in master.itertuples(
            index=False
        )
    }

    physical_keys = set(
        image_index.keys()
    )

    missing = metadata_keys - physical_keys
    extra = physical_keys - metadata_keys

    print(
        f"Indexed physical images: "
        f"{len(image_index):,}"
    )

    print(
        f"Metadata images missing from disk: "
        f"{len(missing):,}"
    )

    print(
        f"Extra physical images without metadata: "
        f"{len(extra):,}"
    )

    return master, len(missing), len(extra)


# ============================================================
# EXPERIMENT FLAGS
#
# These will become useful immediately when we start the
# frontal-first experiments.
# ============================================================

def add_experiment_flags(
    master: pd.DataFrame,
) -> pd.DataFrame:
    master = master.copy()

    master[
        "benchmark_identical_subject"
    ] = master[
        "is_standard_identical_twin"
    ].fillna(False)

    master[
        "benchmark_clean_frontal"
    ] = (
        master[
            "benchmark_identical_subject"
        ]
        & master["is_frontal"]
        & master[
            "illumination_group"
        ].eq("controlled")
        & master[
            "glasses_normalized"
        ].eq("none")
    ).fillna(False)

    master[
        "controlled_neutral_frontal"
    ] = (
        master[
            "benchmark_clean_frontal"
        ]
        & master[
            "expression_group"
        ].eq("neutral")
    ).fillna(False)

    master[
        "controlled_smiling_frontal"
    ] = (
        master[
            "benchmark_clean_frontal"
        ]
        & master[
            "expression_group"
        ].eq("smiling")
    ).fillna(False)

    master[
        "uncontrolled_frontal"
    ] = (
        master[
            "benchmark_identical_subject"
        ]
        & master["is_frontal"]
        & master[
            "illumination_group"
        ].eq("uncontrolled")
        & master[
            "glasses_normalized"
        ].eq("none")
    ).fillna(False)

    master[
        "extended_monozygotic_subject"
    ] = master[
        "is_monozygotic_twin"
    ].fillna(False)

    return master


# ============================================================
# EXTENDED R METADATA
#
# Key:
#
# master.image_id <-> xmlTable.sequenceid
#
# Never merge this table using subject_id.
# ============================================================

def enrich_from_extended_metadata(
    master: pd.DataFrame,
    rdata_path: Path,
    object_name: str,
) -> tuple[pd.DataFrame, int]:
    print(
        f"Loading extended metadata: "
        f"{rdata_path}"
    )

    objects = pyreadr.read_r(
        rdata_path
    )

    if object_name not in objects:
        raise KeyError(
            f"RData object '{object_name}' not found. "
            f"Available objects: {list(objects.keys())}"
        )

    xml = objects[
        object_name
    ].copy()

    xml.columns = (
        xml.columns
        .astype(str)
        .str.strip()
    )

    xml = clean_text_columns(xml)

    if "sequenceid" not in xml.columns:
        raise ValueError(
            "xmlTable does not contain sequenceid."
        )

    xml["sequenceid"] = (
        xml["sequenceid"]
        .astype("string")
        .str.strip()
    )

    master_ids = set(
        master["image_id"]
        .astype(str)
    )

    xml_current = xml[
        xml["sequenceid"].isin(
            master_ids
        )
    ].copy()

    extra_xml_rows = (
        len(xml)
        - len(xml_current)
    )

    if xml_current[
        "sequenceid"
    ].duplicated().any():
        duplicates = (
            xml_current.loc[
                xml_current[
                    "sequenceid"
                ].duplicated(
                    keep=False
                ),
                "sequenceid",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate sequenceid values found "
            "within current dataset overlap: "
            f"{duplicates[:20]}"
        )

    if len(xml_current) != len(master):
        missing = (
            master_ids
            - set(
                xml_current[
                    "sequenceid"
                ].astype(str)
            )
        )

        raise ValueError(
            "Extended metadata does not contain "
            "one row for every master image. "
            f"Missing sequence IDs: "
            f"{list(sorted(missing))[:20]}"
        )

    # --------------------------------------------------------
    # Validate identity before merge
    # --------------------------------------------------------

    if "subjectid" in xml_current.columns:
        xml_current[
            "_xml_subject_id"
        ] = (
            xml_current["subjectid"]
            .apply(
                normalize_subject_id
            )
        )

        check = master[
            [
                "image_id",
                "subject_id",
            ]
        ].merge(
            xml_current[
                [
                    "sequenceid",
                    "_xml_subject_id",
                ]
            ],
            left_on="image_id",
            right_on="sequenceid",
            how="left",
            validate="one_to_one",
        )

        mismatch = check[
            check["subject_id"]
            != check["_xml_subject_id"]
        ]

        if not mismatch.empty:
            raise ValueError(
                "Subject mismatch between master "
                "metadata and xmlTable:\n"
                + mismatch[
                    [
                        "image_id",
                        "subject_id",
                        "_xml_subject_id",
                    ]
                ]
                .head(20)
                .to_string(index=False)
            )

    available_fields = {
        source: target
        for source, target
        in XML_FIELD_MAP.items()
        if source in xml_current.columns
    }

    enrichment = (
        xml_current[
            [
                "sequenceid",
                *available_fields.keys(),
            ]
        ]
        .rename(
            columns={
                "sequenceid": "image_id",
                **available_fields,
            }
        )
        .copy()
    )

    master = master.merge(
        enrichment,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Extended metadata rows: "
        f"{len(xml):,}"
    )

    print(
        f"Rows matching current release: "
        f"{len(xml_current):,}"
    )

    print(
        f"Extra rows outside current release: "
        f"{extra_xml_rows:,}"
    )

    return master, extra_xml_rows


# ============================================================
# RESERVED QC COLUMNS
#
# No values are invented here.
#
# These fields remain empty until the NEXT stage actually
# reads and processes each image.
# ============================================================

def add_empty_qc_columns(
    master: pd.DataFrame,
) -> pd.DataFrame:
    master = master.copy()

    qc_columns = [
        "image_readable",
        "face_detected",
        "alignment_success",
        "blur_score",
        "quality_score",
        "usable",
        "exclusion_reason",
    ]

    for column in qc_columns:
        master[column] = pd.NA

    return master


# ============================================================
# COLUMN ORDER
# ============================================================

def reorder_columns(
    master: pd.DataFrame,
) -> pd.DataFrame:
    first = [
        "image_id",
        "filename",
        "image_path",
        "file_exists",

        "subject_id",

        "family_group_id",
        "family_group_size",
        "family_member_ids",
        "family_member_label",

        "twin_group_id",
        "twin_group_size",
        "twin_member_ids",
        "twin_member_label",

        "monozygotic_group_id",
        "monozygotic_group_size",
        "monozygotic_member_ids",
        "monozygotic_member_label",

        "has_relationship_record",
        "relationship_types_raw",
        "primary_relationship_type",
        "related_subject_ids",
        "co_twin_subject_ids",
        "standard_identical_twin_subject_ids",
        "monozygotic_twin_subject_ids",
        "fraternal_twin_subject_ids",
        "unknown_twin_subject_ids",
        "sibling_subject_ids",

        "is_standard_identical_twin",
        "is_monozygotic_twin",
        "is_identical_mirror",
        "is_identical_triplet",
        "is_fraternal_twin",
        "has_unknown_twin_type",
        "has_sibling_relationship",
        "is_unpaired_in_pair_table",

        "capture_date",
        "capture_year",
        "gender",
        "race",
        "year_of_birth",
        "age_at_capture",
        "age_group_40",

        "weather",
        "illumination_group",

        "glasses",
        "glasses_normalized",
        "has_glasses",
        "has_prism",

        "expression",
        "expression_group",

        "yaw",
        "pose_group",
        "profile_side",
        "is_frontal",

        "benchmark_identical_subject",
        "benchmark_clean_frontal",
        "controlled_neutral_frontal",
        "controlled_smiling_frontal",
        "uncontrolled_frontal",
        "extended_monozygotic_subject",

        "xml_record_id",
        "recording_id",
        "shot_id",
        "xml_file_id",
        "source_format",
        "stage_id",
        "collection_id",
        "environment_id",
        "sensor_id",
        "illuminant_id",
        "illuminant_id_1",
        "xml_state",
        "tag_list",

        "image_readable",
        "face_detected",
        "alignment_success",
        "blur_score",
        "quality_score",
        "usable",
        "exclusion_reason",
    ]

    first = [
        column
        for column in first
        if column in master.columns
    ]

    remaining = [
        column
        for column in master.columns
        if column not in first
    ]

    return master[
        first + remaining
    ].copy()


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_master(
    master: pd.DataFrame,
    expected_rows: int | None,
    expected_subjects: int | None,
    strict: bool,
) -> None:
    if not master.columns.is_unique:
        raise ValueError(
            "Duplicate column names exist in master "
            "metadata."
        )

    if master[
        "image_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate image IDs remain after merge."
        )

    if master[
        "subject_id"
    ].isna().any():
        raise ValueError(
            "Master metadata contains missing "
            "subject IDs."
        )

    if master[
        "family_group_id"
    ].isna().any():
        raise ValueError(
            "Some subjects do not have a family "
            "group assignment."
        )

    if strict and expected_rows is not None:
        if len(master) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} final rows, "
                f"found {len(master)}."
            )

    if strict and expected_subjects is not None:
        subjects = master[
            "subject_id"
        ].nunique()

        if subjects != expected_subjects:
            raise ValueError(
                f"Expected {expected_subjects} "
                f"subjects, found {subjects}."
            )


# ============================================================
# REPORT
# ============================================================

def build_report(
    master: pd.DataFrame,
    relationships: pd.DataFrame,
    external_edges: pd.DataFrame,
    missing_physical: int,
    extra_physical: int,
    extra_xml_rows: int,
) -> dict:
    standard_identical_edges = relationships[
        relationships[
            "relationship_type"
        ].eq("Identical")
    ]

    report = {
        "images": int(len(master)),
        "unique_image_ids": int(
            master["image_id"].nunique()
        ),
        "subjects": int(
            master["subject_id"].nunique()
        ),
        "family_groups": int(
            master[
                "family_group_id"
            ].nunique()
        ),
        "twin_groups": int(
            master[
                "twin_group_id"
            ].dropna().nunique()
        ),
        "monozygotic_groups": int(
            master[
                "monozygotic_group_id"
            ].dropna().nunique()
        ),
        "standard_identical_edges": int(
            len(standard_identical_edges)
        ),
        "standard_identical_subjects": int(
            master.loc[
                master[
                    "is_standard_identical_twin"
                ],
                "subject_id",
            ].nunique()
        ),
        "monozygotic_subjects": int(
            master.loc[
                master[
                    "is_monozygotic_twin"
                ],
                "subject_id",
            ].nunique()
        ),
        "fraternal_subjects": int(
            master.loc[
                master[
                    "is_fraternal_twin"
                ],
                "subject_id",
            ].nunique()
        ),
        "unpaired_subjects": int(
            master.loc[
                master[
                    "is_unpaired_in_pair_table"
                ],
                "subject_id",
            ].nunique()
        ),
        "external_relationship_edges": int(
            len(external_edges)
        ),
        "missing_physical_images": int(
            missing_physical
        ),
        "extra_physical_images": int(
            extra_physical
        ),
        "extra_xml_rows": int(
            extra_xml_rows
        ),
        "yaw_distribution": {
            str(key): int(value)
            for key, value
            in master[
                "yaw"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .items()
        },
        "expression_distribution": {
            str(key): int(value)
            for key, value
            in master[
                "expression_group"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        },
        "illumination_distribution": {
            str(key): int(value)
            for key, value
            in master[
                "illumination_group"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        },
    }

    return report


# ============================================================
# MAIN
# ============================================================

def main(config_path: Path) -> None:
    config = load_config(
        config_path
    )

    dataset_config = config[
        "dataset"
    ]

    output_config = config[
        "output"
    ]

    validation_config = config.get(
        "validation",
        {},
    )

    dataset_root = Path(
        dataset_config["root"]
    )

    images_root = dataset_path(
        dataset_root,
        dataset_config["images_dir"],
    )

    image_metadata_csv = dataset_path(
        dataset_root,
        dataset_config[
            "image_metadata_csv"
        ],
    )

    twin_pair_rdata = dataset_path(
        dataset_root,
        dataset_config[
            "twin_pair_rdata"
        ],
    )

    extended_metadata_rdata = dataset_path(
        dataset_root,
        dataset_config[
            "extended_metadata_rdata"
        ],
    )

    output_csv = project_path(
        output_config[
            "metadata_csv"
        ]
    )

    report_json = project_path(
        output_config[
            "report_json"
        ]
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_rows = validation_config.get(
        "expected_image_rows"
    )

    expected_subjects = (
        validation_config.get(
            "expected_subjects"
        )
    )

    strict = validation_config.get(
        "strict",
        True,
    )

    # --------------------------------------------------------
    # 1. Image-level metadata
    # --------------------------------------------------------

    images = load_image_metadata(
        csv_path=image_metadata_csv,
        expected_rows=expected_rows,
        expected_subjects=expected_subjects,
        strict=strict,
    )

    dataset_subjects = set(
        images[
            "subject_id"
        ]
        .astype(str)
        .unique()
    )

    # --------------------------------------------------------
    # 2. Relationship metadata
    # --------------------------------------------------------

    relationships, external_edges = (
        load_relationship_edges(
            rdata_path=twin_pair_rdata,
            object_name=dataset_config[
                "twin_pair_object"
            ],
            dataset_subjects=dataset_subjects,
        )
    )

    relationship_lookup = (
        build_relationship_lookup(
            relationships,
            dataset_subjects,
        )
    )

    # --------------------------------------------------------
    # 3. Merge image -> subject metadata
    #
    # many image rows -> exactly one subject row.
    # --------------------------------------------------------

    master = images.merge(
        relationship_lookup,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 4. Derived labels
    # --------------------------------------------------------

    master = add_condition_labels(
        master
    )

    # --------------------------------------------------------
    # 5. Physical image paths
    # --------------------------------------------------------

    valid_extensions = {
        extension.lower()
        for extension
        in dataset_config[
            "valid_image_extensions"
        ]
    }

    (
        master,
        missing_physical,
        extra_physical,
    ) = add_image_paths(
        master,
        images_root,
        valid_extensions,
    )

    # --------------------------------------------------------
    # 6. Experiment eligibility flags
    # --------------------------------------------------------

    master = add_experiment_flags(
        master
    )

    # --------------------------------------------------------
    # 7. Extended xmlTable metadata
    # --------------------------------------------------------

    (
        master,
        extra_xml_rows,
    ) = enrich_from_extended_metadata(
        master,
        rdata_path=extended_metadata_rdata,
        object_name=dataset_config[
            "extended_metadata_object"
        ],
    )

    # --------------------------------------------------------
    # 8. Reserve QC columns
    # --------------------------------------------------------

    master = add_empty_qc_columns(
        master
    )

    # --------------------------------------------------------
    # 9. Final integrity checks
    # --------------------------------------------------------

    master = reorder_columns(
        master
    )

    validate_master(
        master,
        expected_rows=expected_rows,
        expected_subjects=expected_subjects,
        strict=strict,
    )

    # --------------------------------------------------------
    # 10. Save metadata
    # --------------------------------------------------------

    master.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # 11. Report
    # --------------------------------------------------------

    report = build_report(
        master=master,
        relationships=relationships,
        external_edges=external_edges,
        missing_physical=missing_physical,
        extra_physical=extra_physical,
        extra_xml_rows=extra_xml_rows,
    )

    with report_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\n========================================")
    print("ND-TWINS METADATA BUILD COMPLETE")
    print("========================================")

    print(
        f"Images:             "
        f"{report['images']:,}"
    )

    print(
        f"Subjects:           "
        f"{report['subjects']:,}"
    )

    print(
        f"Family groups:      "
        f"{report['family_groups']:,}"
    )

    print(
        f"Twin groups:        "
        f"{report['twin_groups']:,}"
    )

    print(
        f"Monozygotic groups: "
        f"{report['monozygotic_groups']:,}"
    )

    print(
        f"Missing files:      "
        f"{report['missing_physical_images']:,}"
    )

    print(
        f"\nMetadata saved to:\n"
        f"{output_csv}"
    )

    print(
        f"\nBuild report saved to:\n"
        f"{report_json}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build ND-TWINS master metadata."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/data.yaml"
        ),
    )

    args = parser.parse_args()

    main(
        args.config
    )