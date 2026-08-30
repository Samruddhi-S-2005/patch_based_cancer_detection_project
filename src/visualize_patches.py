"""
Patch Visualization Script for CAMELYON16 Dataset
Creates a contact sheet showing representative Normal and Tumor patches.
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_patch_files(patch_dir: Path):
    """Return all PNG files sorted in the given directory."""
    if not patch_dir.exists():
        return []
    return sorted(
        [f for f in patch_dir.iterdir() if f.suffix.lower() == ".png"],
        key=lambda p: p.name
    )


def select_representative_patches(patch_list, count=10):
    """
    Select representative patches evenly across different slide sources if possible.
    """
    if len(patch_list) <= count:
        return patch_list

    # Group by slide prefix (e.g. normal_083, normal_084, tumor_105, tumor_107)
    slide_groups = {}
    for p in patch_list:
        parts = p.stem.split("_")
        slide_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "default"
        slide_groups.setdefault(slide_id, []).append(p)

    selected = []
    num_groups = len(slide_groups)
    per_group = count // num_groups
    remainder = count % num_groups

    for idx, (slide_id, files) in enumerate(slide_groups.items()):
        take = per_group + (1 if idx < remainder else 0)

        if take <= 0:
            continue

        # Sample evenly across the files belonging to this slide
        if len(files) <= take:
            selected.extend(files)
        else:
            positions = [
                round(i * (len(files) - 1) / (take - 1))
                for i in range(take)
            ] if take > 1 else [len(files) // 2]

            selected.extend(files[pos] for pos in positions)

    if len(selected) != count:
        selected = patch_list[:count]

    return selected


def format_patch_label(patch_path: Path) -> str:
    """Extract a concise readable identifier from the patch filename."""
    filename = patch_path.name
    stem = patch_path.stem
    # Example format: normal_083_x1470_y49975_label0_patch7
    parts = stem.split("_")
    try:
        slide = f"{parts[0]}_{parts[1]}"
        patch_id = parts[-1].capitalize()
        coord = f"({parts[2]}, {parts[3]})"
        return f"{slide} - {patch_id}\n{coord}"
    except (IndexError, ValueError):
        return filename


def main():
    # Resolve project root and directory paths
    base_dir = Path(__file__).resolve().parent.parent
    patches_base = base_dir / "data" / "PATCHES" / "PROCESSED" / "train"
    
    # Case-insensitive resolution for folder names
    def resolve_subdir(parent: Path, name: str) -> Path:
        direct = parent / name
        if direct.exists():
            return direct
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
        return direct

    normal_dir = resolve_subdir(patches_base, "normal")
    tumor_dir = resolve_subdir(patches_base, "tumor")
    output_dir = base_dir / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "patch_contact_sheet.png"

    # Read patch files
    normal_patches = get_patch_files(normal_dir)
    tumor_patches = get_patch_files(tumor_dir)
    total_patches = len(normal_patches) + len(tumor_patches)

    # Print required dataset statistics
    print("=" * 60)
    print("CAMELYON16 PATCH DATASET SUMMARY")
    print("=" * 60)
    print(f"Number of normal patches : {len(normal_patches)}")
    print(f"Number of tumor patches  : {len(tumor_patches)}")
    print(f"Total patches            : {total_patches}")
    print("=" * 60)

    # Select representative samples (10 each)
    num_samples = 50
    selected_normal = select_representative_patches(normal_patches, count=num_samples)
    selected_tumor = select_representative_patches(tumor_patches, count=num_samples)

    print(f"Selected {len(selected_normal)} normal patches and {len(selected_tumor)} tumor patches for contact sheet.")

    # Create figure using 6-row GridSpec layout
    fig = plt.figure(figsize=(15, 14), facecolor="#FAFAFA")
    
    # Overall Title
    fig.suptitle(
        "CAMELYON16 Histopathology Dataset — Patch Contact Sheet (256 \u00d7 256 px)",
        fontsize=16,
        fontweight="bold",
        y=0.985,
        color="#1A202C"
    )

    # GridSpec: 6 rows (Header, Patches 1-5, Patches 6-10, Header, Patches 1-5, Patches 6-10)
    gs = gridspec.GridSpec(
        6, 5,
        figure=fig,
        height_ratios=[0.35, 2.0, 2.0, 0.45, 2.0, 2.0],
        top=0.94,
        bottom=0.03,
        left=0.04,
        right=0.96,
        hspace=0.38,
        wspace=0.16
    )

    # 1. Normal Header Row
    ax_norm_hdr = fig.add_subplot(gs[0, :])
    ax_norm_hdr.axis("off")
    ax_norm_hdr.text(
        0.5, 0.5,
        "NORMAL (Benign Tissue — Label 0)",
        fontsize=13,
        fontweight="bold",
        color="#1B5E20",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#81C784", linewidth=1.2)
    )

    # Normal Patches (Rows 1 and 2)
    for idx in range(10):
        r = 1 if idx < 5 else 2
        c = idx % 5
        ax = fig.add_subplot(gs[r, c])
        if idx < len(selected_normal):
            patch_p = selected_normal[idx]
            img = Image.open(patch_p)
            ax.imshow(img)
            ax.set_title(format_patch_label(patch_p), fontsize=8.5, pad=5, color="#1B5E20", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2E7D32")
            spine.set_linewidth(1.8)

    # 2. Tumor Header Row
    ax_tumor_hdr = fig.add_subplot(gs[3, :])
    ax_tumor_hdr.axis("off")
    ax_tumor_hdr.text(
        0.5, 0.5,
        "TUMOR (Metastatic Cancer Tissue — Label 1)",
        fontsize=13,
        fontweight="bold",
        color="#B71C1C",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#E57373", linewidth=1.2)
    )

    # Tumor Patches (Rows 4 and 5)
    for idx in range(10):
        r = 4 if idx < 5 else 5
        c = idx % 5
        ax = fig.add_subplot(gs[r, c])
        if idx < len(selected_tumor):
            patch_p = selected_tumor[idx]
            img = Image.open(patch_p)
            ax.imshow(img)
            ax.set_title(format_patch_label(patch_p), fontsize=8.5, pad=5, color="#B71C1C", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#C62828")
            spine.set_linewidth(1.8)

    # Save contact sheet
    plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n[SUCCESS] Contact sheet successfully generated!")
    print(f"Saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
