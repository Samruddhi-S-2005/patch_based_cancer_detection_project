from pathlib import Path
import random
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path("data/ANOMALIES/bach_iciar2018")
IMAGE_ROOT = ROOT / "images"
INVENTORY = ROOT / "bach_dataset_inventory.csv"

OUT_ROOT = Path("result/objective_2/anomaly_data/bach_ductal_proxy")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 224
STRIDE = 112

# Keep the dataset manageable and prevent one image from dominating.
MAX_PATCHES_PER_IMAGE = 50

# Meaningful tissue requirements.
MIN_TISSUE_FRACTION = 0.50
MIN_PATCH_STD = 8.0

# Reproducible selection.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

df = pd.read_csv(INVENTORY)

# Only labeled training images relevant to this anomaly head.
df = df[
    (df["split"] == "train") &
    (df["class_label"].isin(["InSitu", "Normal", "Benign"]))
].copy()

print("Images selected for extraction:")
print(df["class_label"].value_counts().sort_index())
print()

# ---------------------------------------------------------------------
# Create image-level train/validation/test split.
# IMPORTANT: split images BEFORE extracting patches.
# ---------------------------------------------------------------------

split_records = []

for cls in ["InSitu", "Normal", "Benign"]:
    sub = df[df["class_label"] == cls].copy()

    ids = sub["image_id"].tolist()
    rng = random.Random(RANDOM_SEED)

    # Deterministic shuffle at IMAGE level.
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    for _, row in sub.iterrows():
        image_id = row["image_id"]

        if image_id in train_ids:
            split = "train"
        elif image_id in val_ids:
            split = "validation"
        elif image_id in test_ids:
            split = "test"
        else:
            raise RuntimeError(f"Image {image_id} was not assigned a split.")

        split_records.append({
            "image_id": image_id,
            "filename": row["filename"],
            "class_label": row["class_label"],
            "split": split,
        })

split_df = pd.DataFrame(split_records)

split_path = OUT_ROOT / "bach_image_split.csv"
split_df.to_csv(split_path, index=False)

print("Image-level split:")
print(
    split_df.groupby(["split", "class_label"])
    .size()
    .reset_index(name="images")
    .to_string(index=False)
)
print()

# ---------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------

manifest_rows = []

for _, row in split_df.iterrows():

    image_id = row["image_id"]
    filename = row["filename"]
    class_label = row["class_label"]
    split = row["split"]

    image_path = IMAGE_ROOT / class_label / filename

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)

    height, width = arr.shape[:2]

    candidates = []

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            patch = arr[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # Tissue-like pixels:
            # white/background pixels have all channels near 255.
            tissue_mask = np.any(patch < 240, axis=2)
            tissue_fraction = float(tissue_mask.mean())

            if tissue_fraction < MIN_TISSUE_FRACTION:
                continue

            patch_std = float(patch.std())

            if patch_std < MIN_PATCH_STD:
                continue

            # Rank by tissue fraction first, then variation.
            candidates.append(
                (
                    tissue_fraction,
                    patch_std,
                    x,
                    y,
                    patch.copy()
                )
            )

    # Deterministic ranking.
    candidates.sort(
        key=lambda z: (z[0], z[1]),
        reverse=True
    )

    selected = candidates[:MAX_PATCHES_PER_IMAGE]

    label = 1 if class_label == "InSitu" else 0

    class_dir = "positive" if label == 1 else "negative"
    out_dir = OUT_ROOT / split / class_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (
        tissue_fraction,
        patch_std,
        x,
        y,
        patch
    ) in enumerate(selected):

        patch_name = (
            f"{image_id}"
            f"_x{x}_y{y}"
            f"_ductalproxy{label}.png"
        )

        patch_path = out_dir / patch_name

        Image.fromarray(patch).save(
            patch_path,
            format="PNG"
        )

        manifest_rows.append({
            "split": split,
            "image_id": image_id,
            "source_image": filename,
            "source_class": class_label,
            "patch_path": str(patch_path),
            "x": x,
            "y": y,
            "patch_size": PATCH_SIZE,
            "tissue_fraction": tissue_fraction,
            "patch_std": patch_std,
            "label": label,
            "label_name": (
                "ductal_architecture_proxy"
                if label == 1
                else "non_ductal_proxy"
            ),
            "label_source":
                "official_image_level_diagnosis_weak_label",
        })

    print(
        f"{split:10s} "
        f"{image_id:6s} "
        f"{class_label:8s} "
        f"candidates={len(candidates):3d} "
        f"selected={len(selected):2d}"
    )

manifest = pd.DataFrame(manifest_rows)

manifest_path = OUT_ROOT / "bach_ductal_proxy_patch_manifest.csv"
manifest.to_csv(manifest_path, index=False)

print()
print("=" * 70)
print("BACH EXTRACTION COMPLETE")
print("=" * 70)

print(f"Images: {len(split_df)}")
print(f"Patches: {len(manifest)}")
print()

print("Split × label:")
print(
    manifest.groupby(["split", "label_name"])
    .size()
    .reset_index(name="patches")
    .to_string(index=False)
)

print()
print("Manifest:")
print(manifest_path)

print()
print("Image split:")
print(split_path)
