"""
Second-Stage Quantitative QC Audit
----------------------------------
Analyzes all currently retained patches without modifying them.

For every patch, calculates:
- tissue_ratio
- white_ratio
- blur_score

The script only creates an audit CSV and summary.
It DOES NOT delete, move, or modify any patch.
"""

from pathlib import Path
import csv
import cv2
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TISSUE_THRESHOLD = 0.10
WHITE_THRESHOLD = 0.85
BLUR_THRESHOLD = 50.0


# ---------------------------------------------------------
# Patch metrics
# ---------------------------------------------------------

def calculate_metrics(image_path):
    """Calculate tissue, white-background and blur metrics."""

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # White/background detection
    # -----------------------------------------------------
    #
    # Pixels close to white are treated as background.
    #
    white_mask = np.all(rgb >= 235, axis=2)

    white_ratio = float(np.mean(white_mask))

    # -----------------------------------------------------
    # Tissue estimation
    # -----------------------------------------------------
    #
    # Tissue generally contains stronger color variation
    # than the white background.
    #
    # HSV saturation provides a simple stain/tissue signal.
    #
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]

    tissue_mask = saturation > 20

    tissue_ratio = float(np.mean(tissue_mask))

    # -----------------------------------------------------
    # Blur estimation
    # -----------------------------------------------------
    #
    # Variance of Laplacian is a common focus/blur metric.
    # Higher value generally means sharper detail.
    #
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # -----------------------------------------------------
    # Flags
    # -----------------------------------------------------

    flags = []

    if tissue_ratio < TISSUE_THRESHOLD:
        flags.append("low_tissue")

    if white_ratio > WHITE_THRESHOLD:
        flags.append("mostly_white")

    if blur_score < BLUR_THRESHOLD:
        flags.append("blur")

    if flags:
        qc_flag = "REVIEW"
        removal_reason = ";".join(flags)
    else:
        qc_flag = "KEEP"
        removal_reason = ""

    return {
        "tissue_ratio": tissue_ratio,
        "white_ratio": white_ratio,
        "blur_score": blur_score,
        "qc_flag": qc_flag,
        "removal_reason": removal_reason,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    base_dir = Path(__file__).resolve().parent.parent

    processed_dir = base_dir / "data" / "PATCHES" / "PROCESSED"

    output_dir = base_dir / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "second_stage_qc_audit.csv"

    # Current retained patches
    patch_dirs = {
        "normal": processed_dir / "train" / "normal",
        "tumor": processed_dir / "train" / "tumor",
    }

    print("=" * 70)
    print("SECOND-STAGE PATCH QC AUDIT")
    print("=" * 70)

    all_rows = []

    for label_name, patch_dir in patch_dirs.items():

        if not patch_dir.exists():
            print(f"[WARNING] Directory not found: {patch_dir}")
            continue

        patch_files = sorted(patch_dir.glob("*.png"))

        print(f"\nProcessing {label_name}: {len(patch_files)} patches")

        for index, patch_path in enumerate(patch_files, start=1):

            metrics = calculate_metrics(patch_path)

            if metrics is None:
                print(f"[WARNING] Could not read: {patch_path}")
                continue

            row = {
                "patch_name": patch_path.name,
                "label": 0 if label_name == "normal" else 1,
                "class": label_name,
                "patch_path": str(patch_path),
                **metrics,
            }

            all_rows.append(row)

            if index % 1000 == 0:
                print(f"  Processed {index}/{len(patch_files)}")

    # -----------------------------------------------------
    # Write audit CSV
    # -----------------------------------------------------

    fieldnames = [
        "patch_name",
        "label",
        "class",
        "patch_path",
        "tissue_ratio",
        "white_ratio",
        "blur_score",
        "qc_flag",
        "removal_reason",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(all_rows)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total = len(all_rows)

    keep_count = sum(
        1 for row in all_rows
        if row["qc_flag"] == "KEEP"
    )

    review_count = sum(
        1 for row in all_rows
        if row["qc_flag"] == "REVIEW"
    )

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(f"Total patches analyzed : {total}")
    print(f"KEEP                   : {keep_count}")
    print(f"REVIEW                 : {review_count}")

    print("\nReview reasons:")

    reasons = {}

    for row in all_rows:

        reason = row["removal_reason"]

        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1

    if reasons:

        for reason, count in sorted(
            reasons.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {reason:<35} {count}")

    else:
        print("  None")

    print("\nThresholds used:")
    print(f"  Tissue ratio < {TISSUE_THRESHOLD}")
    print(f"  White ratio  > {WHITE_THRESHOLD}")
    print(f"  Blur score   < {BLUR_THRESHOLD}")

    print("\n[SUCCESS]")
    print(f"Audit saved to:\n{output_csv.resolve()}")

    print("\nIMPORTANT:")
    print("No patches were deleted, moved, renamed, or modified.")


if __name__ == "__main__":
    main()