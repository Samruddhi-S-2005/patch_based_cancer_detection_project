from pathlib import Path
from PIL import Image, ImageStat
import cv2
import numpy as np
import csv
import shutil


# ============================================================
# PATHS
# ============================================================

INPUT_ROOT = Path("data/PATCHES/REDESIGNED/train")
OUTPUT_ROOT = Path("data/PATCHES/PROCESSED/train")

MANIFEST_PATH = Path("data/PATCHES/PROCESSED/processed_patch_manifest.csv")


# ============================================================
# SETTINGS
# ============================================================

# Very low tissue content -> likely background/empty patch
MIN_TISSUE_RATIO = 0.20

# Very high white/background ratio
MAX_WHITE_RATIO = 0.80

# Blur threshold
# Lower values indicate stronger blur.
MIN_LAPLACIAN_VARIANCE = 20.0


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for label in ["normal", "tumor"]:
    (OUTPUT_ROOT / label).mkdir(parents=True, exist_ok=True)


# ============================================================
# QUALITY CHECK FUNCTIONS
# ============================================================

def calculate_tissue_ratio(image):
    """
    Estimate the amount of tissue in the patch.

    Histology tissue is generally more saturated/darker than
    white slide background.
    """

    img = np.array(image.convert("RGB"))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Pixels that are not close to white
    tissue_mask = ~(
        (saturation < 25) &
        (value > 210)
    )

    tissue_ratio = np.mean(tissue_mask)

    return float(tissue_ratio)


def calculate_white_ratio(image):
    """
    Calculate approximate white/background proportion.
    """

    img = np.array(image.convert("RGB"))

    white_mask = (
        (img[:, :, 0] > 220) &
        (img[:, :, 1] > 220) &
        (img[:, :, 2] > 220)
    )

    return float(np.mean(white_mask))


def calculate_blur_score(image):
    """
    Variance of Laplacian is a simple focus/blur measure.

    Lower value = blurrier image.
    """

    img = np.array(image.convert("L"))

    score = cv2.Laplacian(img, cv2.CV_64F).var()

    return float(score)


def inspect_patch(path):
    """
    Calculate QC measurements and determine whether the
    patch should be retained.
    """

    try:
        image = Image.open(path).convert("RGB")

        tissue_ratio = calculate_tissue_ratio(image)
        white_ratio = calculate_white_ratio(image)
        blur_score = calculate_blur_score(image)

        reasons = []

        # Background/empty patch
        if tissue_ratio < MIN_TISSUE_RATIO:
            reasons.append("low_tissue")

        # Extremely white patch
        if white_ratio > MAX_WHITE_RATIO:
            reasons.append("mostly_white")

        # Severe blur
        if blur_score < MIN_LAPLACIAN_VARIANCE:
            reasons.append("blur")

        keep = len(reasons) == 0

        return {
            "keep": keep,
            "tissue_ratio": tissue_ratio,
            "white_ratio": white_ratio,
            "blur_score": blur_score,
            "reason": ";".join(reasons) if reasons else ""
        }

    except Exception as e:

        return {
            "keep": False,
            "tissue_ratio": 0,
            "white_ratio": 1,
            "blur_score": 0,
            "reason": f"read_error:{e}"
        }


# ============================================================
# MAIN PROCESS
# ============================================================

rows = []

total = 0
kept = 0
removed = 0


for label in ["normal", "tumor"]:

    input_dir = INPUT_ROOT / label
    output_dir = OUTPUT_ROOT / label

    print()
    print("=" * 60)
    print(f"PROCESSING: {label.upper()}")
    print("=" * 60)

    patch_files = sorted(input_dir.glob("*.png"))

    print(f"Input patches: {len(patch_files)}")

    for i, patch_path in enumerate(patch_files, start=1):

        total += 1

        result = inspect_patch(patch_path)

        destination = output_dir / patch_path.name

        if result["keep"]:

            # Copy original patch without modifying the source.
            shutil.copy2(patch_path, destination)

            kept += 1

        else:

            removed += 1

        rows.append({
            "patch_name": patch_path.name,
            "source_path": str(patch_path),
            "processed_path": str(destination) if result["keep"] else "",
            "label": label,
            "tissue_ratio": round(result["tissue_ratio"], 4),
            "white_ratio": round(result["white_ratio"], 4),
            "blur_score": round(result["blur_score"], 2),
            "kept": result["keep"],
            "removal_reason": result["reason"]
        })

        if i % 500 == 0:
            print(f"Processed {i}/{len(patch_files)}")


# ============================================================
# SAVE MANIFEST
# ============================================================

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "patch_name",
    "source_path",
    "processed_path",
    "label",
    "tissue_ratio",
    "white_ratio",
    "blur_score",
    "kept",
    "removal_reason"
]

with open(
    MANIFEST_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("QC SUMMARY")
print("=" * 60)

print(f"Total input patches : {total}")
print(f"Patches retained    : {kept}")
print(f"Patches rejected    : {removed}")

if total > 0:
    print(
        f"Retention rate      : "
        f"{kept / total * 100:.2f}%"
    )

print()
print(f"Processed patches:")
print(OUTPUT_ROOT)

print()
print(f"Manifest:")
print(MANIFEST_PATH)

print()
print("Original extracted patches were NOT modified.")