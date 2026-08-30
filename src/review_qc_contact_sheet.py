"""
Visual audit of REVIEW patches from second-stage QC.

Does NOT modify, delete, move, or rename any dataset patches.
Creates a contact sheet containing representative REVIEW patches.
"""

import csv
import random
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


def load_review_rows(csv_path):
    """Load only REVIEW rows from the QC audit CSV."""
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    return [row for row in rows if row["qc_flag"].strip().upper() == "REVIEW"]


def sample_rows(rows, count, seed=42):
    """Reproducible random sample."""
    if len(rows) <= count:
        return rows

    rng = random.Random(seed)
    return rng.sample(rows, count)


def main():
    base_dir = Path(__file__).resolve().parent.parent

    audit_csv = base_dir / "result" / "second_stage_qc_audit.csv"
    output_file = base_dir / "result" / "review_qc_contact_sheet.png"

    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")

    review_rows = load_review_rows(audit_csv)

    normal_review = [
        r for r in review_rows
        if r["class"].strip().lower() == "normal"
    ]

    tumor_review = [
        r for r in review_rows
        if r["class"].strip().lower() == "tumor"
    ]

    # -----------------------------
    # Select NORMAL REVIEW patches
    # -----------------------------
    selected_normal = sample_rows(normal_review, 30, seed=42)

    # -----------------------------
    # Select TUMOR REVIEW patches
    #
    # Make sure all 3 low-tissue tumor
    # patches are included.
    # -----------------------------
    tumor_low_tissue = [
        r for r in tumor_review
        if "low_tissue" in r["removal_reason"].lower()
    ]

    tumor_blur = [
        r for r in tumor_review
        if "blur" in r["removal_reason"].lower()
    ]

    selected_tumor = tumor_low_tissue.copy()

    remaining_needed = 30 - len(selected_tumor)

    selected_tumor.extend(
        sample_rows(tumor_blur, remaining_needed, seed=43)
    )

    # If anything unexpected happens, cap at 30.
    selected_normal = selected_normal[:30]
    selected_tumor = selected_tumor[:30]

    print("=" * 70)
    print("SECOND-STAGE QC REVIEW CONTACT SHEET")
    print("=" * 70)
    print(f"Total REVIEW patches : {len(review_rows)}")
    print(f"Normal REVIEW        : {len(normal_review)}")
    print(f"Tumor REVIEW         : {len(tumor_review)}")
    print()
    print(f"Selected normal      : {len(selected_normal)}")
    print(f"Selected tumor       : {len(selected_tumor)}")
    print("=" * 70)

    # 6 rows x 10 columns:
    # 3 rows normal + 3 rows tumor
    fig, axes = plt.subplots(
        6,
        10,
        figsize=(22, 15)
    )

    fig.suptitle(
        "CAMELYON16 — SECOND-STAGE QC REVIEW PATCH AUDIT",
        fontsize=18,
        fontweight="bold"
    )

    all_rows = selected_normal + selected_tumor

    for idx, ax in enumerate(axes.flat):

        if idx >= len(all_rows):
            ax.axis("off")
            continue

        row = all_rows[idx]

        patch_path = Path(row["patch_path"])

        try:
            img = Image.open(patch_path)
            ax.imshow(img)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                "IMAGE ERROR",
                ha="center",
                va="center"
            )

        ax.set_xticks([])
        ax.set_yticks([])

        reason = row["removal_reason"]
        blur = row["blur_score"]
        tissue = row["tissue_ratio"]
        white = row["white_ratio"]

        if idx < 30:
            title_prefix = "NORMAL"
        else:
            title_prefix = "TUMOR"

        ax.set_title(
            f"{title_prefix}\n"
            f"blur={float(blur):.2f} | "
            f"tissue={float(tissue):.3f}\n"
            f"{reason}",
            fontsize=6.5
        )

    # Add visual divider between normal and tumor sections.
    fig.text(
        0.5,
        0.505,
        "↓ TUMOR REVIEW PATCHES ↓",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.985,
        "↑ NORMAL REVIEW PATCHES ↑",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("[SUCCESS] REVIEW contact sheet generated.")
    print(f"Saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()