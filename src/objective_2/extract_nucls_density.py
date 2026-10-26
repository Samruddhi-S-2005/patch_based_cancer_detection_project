from pathlib import Path
import random
import shutil
import math

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("data/ANOMALIES/nucls")

IMAGE_DIR = ROOT / "images"
ANNOTATION_DIR = ROOT / "annotations"

OUT_DIR = Path("result/objective_2/anomaly_data/nucls_density")

PATCH_SIZE = 224
STRIDE = 112

TISSUE_THRESHOLD = 0.50
STD_THRESHOLD = 8.0

RANDOM_SEED = 42

# Maximum selected patches per split
MAX_TRAIN_POS = 3000
MAX_TRAIN_NEG = 3000

MAX_VAL_POS = 750
MAX_VAL_NEG = 750

MAX_TEST_POS = 750
MAX_TEST_NEG = 750


# ============================================================
# HELPERS
# ============================================================

def get_wsi_id(filename):
    """
    NuCLS FOV filenames contain:
        ..._id-<FOV_ID>_left-...

    The parent WSI is the portion before '_id-'.
    """
    stem = Path(filename).stem

    if "_id-" in stem:
        return stem.split("_id-")[0]

    return stem


def parse_polygon_center(x_string, y_string):
    """
    NuCLS coords_x / coords_y contain comma-separated
    polygon boundary coordinates.

    Example:
        coords_x = '72,76,79,...,141'
        coords_y = '62,43,33,...,62'

    Return the centroid based on the polygon vertices.
    """

    try:
        xs = np.array(
            [float(v) for v in str(x_string).split(",") if str(v).strip() != ""],
            dtype=np.float64,
        )

        ys = np.array(
            [float(v) for v in str(y_string).split(",") if str(v).strip() != ""],
            dtype=np.float64,
        )

        if len(xs) == 0 or len(ys) == 0:
            return None

        if len(xs) != len(ys):
            return None

        return float(xs.mean()), float(ys.mean())

    except Exception:
        return None


def load_tumor_centers(annotation_path):
    """
    Read NuCLS annotations and return tumor nucleus centers.
    """

    df = pd.read_csv(annotation_path)

    if len(df) == 0:
        return []

    tumor = df[
        df["raw_classification"]
        .astype(str)
        .str.lower()
        .eq("tumor")
    ]

    centers = []

    for _, row in tumor.iterrows():

        center = parse_polygon_center(
            row["coords_x"],
            row["coords_y"],
        )

        if center is not None:
            centers.append(center)

    return centers


def tissue_fraction(patch):
    """
    Estimate tissue fraction using non-near-white pixels.
    """

    arr = np.asarray(patch)

    tissue = np.any(arr < 240, axis=2)

    return float(tissue.mean())


def patch_std(patch):
    """
    Standard deviation across RGB pixels.
    """

    arr = np.asarray(patch).astype(np.float32)

    return float(arr.std())


def nuclei_in_patch(centers, x, y):
    """
    Count tumor nuclei whose centers fall inside
    the 224x224 patch.
    """

    x2 = x + PATCH_SIZE
    y2 = y + PATCH_SIZE

    count = 0

    for cx, cy in centers:

        if x <= cx < x2 and y <= cy < y2:
            count += 1

    return count


def make_candidate_records(image_path, annotation_path, split, wsi_id):
    """
    Generate metadata only.

    IMPORTANT:
    No PNG files are written during this stage.
    """

    records = []

    centers = load_tumor_centers(annotation_path)

    with Image.open(image_path) as img:

        img = img.convert("RGB")

        width, height = img.size

        for y in range(0, height - PATCH_SIZE + 1, STRIDE):

            for x in range(0, width - PATCH_SIZE + 1, STRIDE):

                patch = img.crop(
                    (
                        x,
                        y,
                        x + PATCH_SIZE,
                        y + PATCH_SIZE,
                    )
                )

                tissue = tissue_fraction(patch)

                if tissue < TISSUE_THRESHOLD:
                    continue

                std = patch_std(patch)

                if std < STD_THRESHOLD:
                    continue

                tumor_nuclei = nuclei_in_patch(
                    centers,
                    x,
                    y,
                )

                tumor_density = (
                    tumor_nuclei /
                    float(PATCH_SIZE * PATCH_SIZE)
                )

                total_density = (
                    len(centers) /
                    float(PATCH_SIZE * PATCH_SIZE)
                )

                records.append(
                    {
                        "split": split,
                        "wsi_id": wsi_id,
                        "source_image": image_path.name,
                        "source_annotation": annotation_path.name,
                        "patch_x": x,
                        "patch_y": y,
                        "patch_size": PATCH_SIZE,
                        "tissue_fraction": tissue,
                        "patch_std": std,
                        "total_tumor_nuclei_in_fov": len(centers),
                        "tumor_nuclei": tumor_nuclei,
                        "tumor_density": tumor_density,
                        "total_density": total_density,
                    }
                )

    return records


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("NUCLS IMAGE / ANNOTATION MATCH")
print("=" * 70)

image_files = sorted(IMAGE_DIR.glob("*.png"))
annotation_files = sorted(ANNOTATION_DIR.glob("*.csv"))

# Ignore inventory CSV if present
annotation_files = [
    p for p in annotation_files
    if "inventory" not in p.name.lower()
]

annotation_map = {
    p.stem: p
    for p in annotation_files
}

matched = []

for image_path in image_files:

    annotation_path = annotation_map.get(image_path.stem)

    if annotation_path is not None:
        matched.append(
            (
                image_path,
                annotation_path,
            )
        )

print(f"RGB images:   {len(image_files)}")
print(f"Annotations:  {len(annotation_files)}")
print(f"Matched FOVs: {len(matched)}")


# ============================================================
# WSI-LEVEL SPLIT
# ============================================================

wsi_ids = sorted(
    {
        get_wsi_id(image_path.name)
        for image_path, _ in matched
    }
)

print()
print(f"Parent WSIs: {len(wsi_ids)}")

random.seed(RANDOM_SEED)

random.shuffle(wsi_ids)

n = len(wsi_ids)

n_train = int(round(n * 0.70))
n_val = int(round(n * 0.15))

train_wsis = set(wsi_ids[:n_train])
val_wsis = set(
    wsi_ids[n_train:n_train + n_val]
)
test_wsis = set(
    wsi_ids[n_train + n_val:]
)

def assign_split(wsi_id):

    if wsi_id in train_wsis:
        return "train"

    if wsi_id in val_wsis:
        return "validation"

    return "test"


split_records = []

for wsi_id in wsi_ids:

    split_records.append(
        {
            "wsi_id": wsi_id,
            "split": assign_split(wsi_id),
        }
    )

split_df = pd.DataFrame(split_records)

print()
print("WSI-level split:")
print(split_df["split"].value_counts())


# ============================================================
# SAVE SPLIT
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

split_path = OUT_DIR / "nucls_wsi_split.csv"

split_df.to_csv(
    split_path,
    index=False,
)


# ============================================================
# METADATA SCAN
# ============================================================

print()

all_records = []

for i, (image_path, annotation_path) in enumerate(
    matched,
    start=1,
):

    wsi_id = get_wsi_id(image_path.name)

    split = assign_split(wsi_id)

    records = make_candidate_records(
        image_path,
        annotation_path,
        split,
        wsi_id,
    )

    all_records.extend(records)

    if i % 100 == 0:
        print(
            f"Scanned {i}/{len(matched)} FOVs"
        )


metadata_df = pd.DataFrame(all_records)

metadata_path = (
    OUT_DIR /
    "nucls_density_candidate_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False,
)

print()
print("=" * 70)
print("METADATA SCAN COMPLETE")
print("=" * 70)

print(
    f"Candidate patches: {len(metadata_df):,}"
)

print(
    f"Metadata file: {metadata_path}"
)


# ============================================================
# SANITY CHECK — TUMOR NUCLEI
# ============================================================

print()
print("=" * 70)
print("NUCLS TUMOR-NUCLEUS SANITY CHECK")
print("=" * 70)

print(
    "Candidate patches containing >=1 tumor nucleus:",
    int(
        (metadata_df["tumor_nuclei"] > 0).sum()
    ),
)

print(
    "Total tumor nuclei counted across candidates:",
    int(
        metadata_df["tumor_nuclei"].sum()
    ),
)

print(
    "Maximum tumor nuclei in one patch:",
    int(
        metadata_df["tumor_nuclei"].max()
    ),
)

print(
    "Maximum tumor density:",
    float(
        metadata_df["tumor_density"].max()
    ),
)


# ============================================================
# TRAINING-ONLY THRESHOLD
# ============================================================

train_df = metadata_df[
    metadata_df["split"] == "train"
].copy()

positive_density_values = train_df.loc[
    train_df["tumor_nuclei"] > 0,
    "tumor_density",
]

if len(positive_density_values) == 0:

    raise RuntimeError(
        "\nERROR: No training candidate contains a tumor nucleus.\n"
        "The NuCLS coordinate parsing is still incorrect.\n"
        "DO NOT continue to training."
    )

threshold = float(
    positive_density_values.quantile(0.75)
)

print()
print("=" * 70)
print("TRAINING-ONLY DENSITY THRESHOLD")
print("=" * 70)

print(
    f"75th percentile tumor density among "
    f"tumor-containing TRAIN patches: "
    f"{threshold:.8f}"
)

print(
    "Equivalent tumor nuclei / patch:",
    threshold * PATCH_SIZE * PATCH_SIZE,
)


# ============================================================
# LABEL CANDIDATES
# ============================================================

def assign_label(row):

    density = row["tumor_density"]
    nuclei = row["tumor_nuclei"]

    # Positive: high tumor-nucleus density
    if density >= threshold and nuclei > 0:
        return 1

    # Negative: no annotated tumor nuclei
    if nuclei == 0:
        return 0

    # Intermediate-density patches excluded
    return -1


metadata_df["label"] = metadata_df.apply(
    assign_label,
    axis=1,
)

metadata_df["label_name"] = metadata_df[
    "label"
].map(
    {
        1: "high_tumor_density",
        0: "no_tumor_nuclei",
        -1: "intermediate_density",
    }
)


print()
print("Candidate label distribution:")
print(
    metadata_df["label_name"].value_counts()
)


# ============================================================
# PATCH SELECTION
# ============================================================

limits = {
    "train": {
        1: MAX_TRAIN_POS,
        0: MAX_TRAIN_NEG,
    },
    "validation": {
        1: MAX_VAL_POS,
        0: MAX_VAL_NEG,
    },
    "test": {
        1: MAX_TEST_POS,
        0: MAX_TEST_NEG,
    },
}

selected_parts = []

for split in [
    "train",
    "validation",
    "test",
]:

    split_df = metadata_df[
        metadata_df["split"] == split
    ].copy()

    for label in [1, 0]:

        subset = split_df[
            split_df["label"] == label
        ].copy()

        # Highest density first for positive patches.
        if label == 1:

            subset = subset.sort_values(
                [
                    "tumor_density",
                    "tissue_fraction",
                    "patch_std",
                ],
                ascending=False,
            )

        else:

            # For negatives, prefer good tissue patches.
            subset = subset.sort_values(
                [
                    "tissue_fraction",
                    "patch_std",
                ],
                ascending=False,
            )

        limit = limits[split][label]

        subset = subset.head(limit)

        selected_parts.append(subset)

        print(
            f"{split}: "
            f"{'positive' if label == 1 else 'negative'} "
            f"{len(subset)}"
        )


selected_df = pd.concat(
    selected_parts,
    ignore_index=True,
)

print()
print(
    f"Selected patches: {len(selected_df):,}"
)


# ============================================================
# SAVE SELECTED METADATA
# ============================================================

selected_metadata_path = (
    OUT_DIR /
    "nucls_density_selected_metadata.csv"
)

selected_df.to_csv(
    selected_metadata_path,
    index=False,
)


# ============================================================
# SAVE PATCHES
# ============================================================

print()

for split in [
    "train",
    "validation",
    "test",
]:

    split_selected = selected_df[
        selected_df["split"] == split
    ]

    for label, label_name in [
        (1, "high_tumor_density"),
        (0, "no_tumor_nuclei"),
    ]:

        subset = split_selected[
            split_selected["label"] == label
        ]

        output_dir = (
            OUT_DIR /
            split /
            "patches" /
            label_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved = 0

        for idx, row in subset.iterrows():

            source_path = (
                IMAGE_DIR /
                row["source_image"]
            )

            try:

                with Image.open(source_path) as img:

                    img = img.convert("RGB")

                    x = int(row["patch_x"])
                    y = int(row["patch_y"])

                    patch = img.crop(
                        (
                            x,
                            y,
                            x + PATCH_SIZE,
                            y + PATCH_SIZE,
                        )
                    )

                    output_name = (
                        f"{row['wsi_id']}"
                        f"__{Path(row['source_image']).stem}"
                        f"__x{x}_y{y}.png"
                    )

                    patch.save(
                        output_dir /
                        output_name
                    )

                    saved += 1

            except Exception as exc:

                print(
                    "WARNING: failed to save",
                    row["source_image"],
                    x,
                    y,
                    exc,
                )

        print(
            f"Saved {split}: "
            f"{saved} patches"
        )


# ============================================================
# FINAL MANIFEST
# ============================================================

manifest_records = []

for split in [
    "train",
    "validation",
    "test",
]:

    split_selected = selected_df[
        selected_df["split"] == split
    ]

    for _, row in split_selected.iterrows():

        label_name = row["label_name"]

        filename = (
            f"{row['wsi_id']}"
            f"__{Path(row['source_image']).stem}"
            f"__x{int(row['patch_x'])}"
            f"_y{int(row['patch_y'])}.png"
        )

        patch_path = (
            OUT_DIR /
            split /
            "patches" /
            label_name /
            filename
        )

        if patch_path.exists():

            manifest_records.append(
                {
                    "split": split,
                    "wsi_id": row["wsi_id"],
                    "source_image": row["source_image"],
                    "patch_path": str(
                        patch_path
                    ),
                    "x": int(row["patch_x"]),
                    "y": int(row["patch_y"]),
                    "patch_size": PATCH_SIZE,
                    "tumor_nuclei": int(
                        row["tumor_nuclei"]
                    ),
                    "tumor_density": float(
                        row["tumor_density"]
                    ),
                    "label": int(row["label"]),
                    "label_name": label_name,
                    "label_source":
                        "NuCLS_nucleus_coordinates_train_threshold",
                }
            )


manifest_df = pd.DataFrame(
    manifest_records
)

manifest_path = (
    OUT_DIR /
    "nucls_density_patch_manifest.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("NUCLS DENSITY EXTRACTION COMPLETE")
print("=" * 70)

if len(manifest_df) > 0:

    print(
        manifest_df.groupby(
            ["split", "label_name"]
        ).size()
    )

print()
print(
    f"Final patches: {len(manifest_df):,}"
)

print(
    f"Final manifest: {manifest_path}"
)

print(
    f"WSI split: {split_path}"
)

print(
    f"Training threshold: {threshold:.8f}"
)

print()
print("IMPORTANT:")
print(
    "Positive = high tumor-nucleus density."
)
print(
    "Negative = no annotated tumor nuclei."
)
print(
    "Intermediate-density patches were excluded."
)
print(
    "Threshold was calculated from TRAINING data only."
)