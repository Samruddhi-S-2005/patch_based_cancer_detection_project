"""
Deterministic, annotation-aware patch extractor for CAMELYON16-style WSIs.

FINAL METHODOLOGY
-----------------
- Level 0, 224x224 patches
- Candidate stride 112 pixels (50% overlap)
- Low-resolution tissue/background screening
- Tumor annotations are consulted BEFORE bounded candidate selection
- Tumor WSI candidates:
    0% overlap        -> normal_tissue, label 0
    0-50% overlap     -> boundary_mixed, excluded
    >=50% overlap     -> tumor_dominant, label 1
- Full-resolution quality/descriptors are computed only for an
  annotation-aware bounded candidate pool
- Final selection uses quality, descriptor diversity, spatial coverage,
  and class balance
- Preferred tumor WSI target: 250 tumor + 250 normal
- No duplicated patches
- Cumulative WSI-aware manifest
- Re-running one WSI replaces only that WSI's records
- --clean-wsi removes only that WSI's saved patches and manifest records
- --dry-run runs full analysis/selection without writing files
- Exactly ONE WSI is processed per command
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import openslide
except ImportError:
    openslide = None

try:
    from .annotation_parser import (
        AnnotationPolygon,
        parse_annotation_xml,
        patch_tumor_overlap_percentage,
    )
except ImportError:
    from annotation_parser import (
        AnnotationPolygon,
        parse_annotation_xml,
        patch_tumor_overlap_percentage,
    )


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data" / "wsi_registry.csv"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "PATCHES" / "REDESIGNED"
MANIFEST_PATH = OUTPUT_ROOT / "patch_manifest.csv"

PATCH_SIZE = 224
CANDIDATE_STRIDE = 112
OPENSLIDE_LEVEL = 0
TARGET_PATCHES = 500

TISSUE_THRESHOLD = 0.50
HSV_SATURATION_THRESHOLD = 25
HSV_VALUE_MIN = 30
HSV_VALUE_MAX = 230

MIN_GRAY_VARIANCE = 5.0

# Normal WSI: maximum expensive candidates per 8x8 tile.
CANDIDATES_PER_TILE = 32

# Tumor WSI: annotation-aware expensive candidate pool.
TUMOR_CANDIDATES_PER_TILE = 16
NORMAL_CANDIDATES_PER_TILE = 16

GRID_ROWS = 8
GRID_COLS = 8

APPLY_GAUSSIAN_BLUR = False
GAUSSIAN_SIGMA = 0.5

NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

TUMOR_TARGET = 250
NORMAL_TARGET = 250

MANIFEST_FIELDS = [
    "wsi_id",
    "source_wsi",
    "split",
    "class_name",
    "label",
    "x",
    "y",
    "patch_size",
    "candidate_stride",
    "level",
    "tissue_fraction",
    "overlap_percentage",
    "region_category",
    "quality_score",
    "selection_score",
    "spatial_tile_row",
    "spatial_tile_column",
    "annotation_region_id",
    "redundancy_status",
    "selected",
    "discarded_ambiguous",
    "discard_reason",
    "saved_path",
]


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

@dataclass
class ScreenedItem:
    x: int
    y: int
    tissue_fraction: float
    preview_variance: float
    tile_row: int
    tile_col: int

    # Filled for tumor WSIs BEFORE bounded preselection.
    overlap: float = 0.0
    region_category: str = "normal_tissue"
    label: int = 0


@dataclass
class Candidate:
    x: int
    y: int
    tissue_fraction: float
    gray_variance: float
    gradient_mean: float
    descriptor: np.ndarray
    tile_row: int
    tile_col: int

    overlap: float = 0.0
    region_category: str = "normal_tissue"
    annotation_region_id: str = ""
    label: int = 0

    quality_score: float = 0.0
    selection_score: float = 0.0
    selected: bool = False
    redundancy_status: str = "not_selected"
    discard_reason: str = ""
    saved_path: str = ""


# ============================================================
# 3. REGISTRY
# ============================================================

def load_registry() -> Dict[str, Dict[str, str]]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry not found: {REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("WSI registry is empty.")

    required = {
        "wsi_id",
        "class_name",
        "split",
        "path",
    }

    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(
            "Registry missing required columns: "
            + ", ".join(sorted(missing))
        )

    ids = [row["wsi_id"] for row in rows]

    if len(rows) != 60:
        raise ValueError(
            f"Expected 60 WSIs in registry, found {len(rows)}."
        )

    if len(set(ids)) != 60:
        raise ValueError(
            "Registry contains duplicate WSI IDs."
        )

    return {row["wsi_id"]: row for row in rows}


# ============================================================
# 4. COORDINATE / SPATIAL HELPERS
# ============================================================

def iter_coordinates(
    width: int,
    height: int,
) -> Iterable[Tuple[int, int]]:
    """224x224 level-0 windows with 112-pixel stride = 50% overlap."""
    max_x = width - PATCH_SIZE
    max_y = height - PATCH_SIZE

    if max_x < 0 or max_y < 0:
        return

    for y in range(0, max_y + 1, CANDIDATE_STRIDE):
        for x in range(0, max_x + 1, CANDIDATE_STRIDE):
            yield x, y


def tile_for(
    x: int,
    y: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    return (
        min(
            GRID_ROWS - 1,
            y * GRID_ROWS // max(height, 1),
        ),
        min(
            GRID_COLS - 1,
            x * GRID_COLS // max(width, 1),
        ),
    )


# ============================================================
# 5. LOW-RESOLUTION SCREENING
# ============================================================

def tissue_fraction(image: Image.Image) -> float:
    small = image.resize(
        (56, 56),
        Image.Resampling.BILINEAR,
    )

    hsv = np.asarray(
        small.convert("HSV"),
        dtype=np.uint8,
    )

    tissue = (
        (hsv[:, :, 1] >= HSV_SATURATION_THRESHOLD)
        & (hsv[:, :, 2] >= HSV_VALUE_MIN)
        & (hsv[:, :, 2] <= HSV_VALUE_MAX)
    )

    return float(tissue.mean())


def gray_variance(image: Image.Image) -> float:
    small = image.resize(
        (56, 56),
        Image.Resampling.BILINEAR,
    ).convert("L")

    return float(
        np.asarray(
            small,
            dtype=np.float32,
        ).var()
    )


def low_res_quality_score(
    image: Image.Image,
) -> Tuple[float, float]:
    return (
        tissue_fraction(image),
        gray_variance(image),
    )


# ============================================================
# 6. OPTIONAL BLUR / NORMALIZATION
# ============================================================

def gaussian_denoise(
    image: Image.Image,
) -> Image.Image:
    if not APPLY_GAUSSIAN_BLUR:
        return image

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=GAUSSIAN_SIGMA
        )
    )


def normalize_for_model(
    array: np.ndarray,
) -> np.ndarray:
    """For later CNN training; never baked into saved PNGs."""
    image = array.astype(np.float32) / 255.0

    mean = np.asarray(
        NORMALIZATION_MEAN,
        dtype=np.float32,
    ).reshape(1, 1, 3)

    std = np.asarray(
        NORMALIZATION_STD,
        dtype=np.float32,
    ).reshape(1, 1, 3)

    return (image - mean) / std


# ============================================================
# 7. FULL-RESOLUTION FEATURES
# ============================================================

def feature_vector(
    image: Image.Image,
) -> Tuple[float, float, np.ndarray]:
    """
    Lightweight selection descriptors.

    These are selection features, not the final CNN features.
    """
    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    gray = rgb.mean(axis=2)
    variance = float(gray.var())

    gradient_y, gradient_x = np.gradient(gray)
    gradient = float(
        np.hypot(
            gradient_x,
            gradient_y,
        ).mean()
    )

    hsv = np.asarray(
        image.convert("HSV"),
        dtype=np.float32,
    )

    hsv_mean = hsv.mean(axis=(0, 1)) / 255.0
    hsv_std = hsv.std(axis=(0, 1)) / 255.0

    thumbnail = np.asarray(
        image.convert("L").resize(
            (16, 16),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    ).ravel() / 255.0

    descriptor = np.concatenate(
        (
            hsv_mean,
            hsv_std,
            thumbnail,
        )
    )

    return variance, gradient, descriptor


# ============================================================
# 8. QUALITY + DIVERSITY SELECTION
# ============================================================

def score_candidates(
    candidates: List[Candidate],
) -> None:
    if not candidates:
        return

    max_variance = max(
        c.gray_variance for c in candidates
    )
    max_gradient = max(
        c.gradient_mean for c in candidates
    )

    for candidate in candidates:
        variance_score = min(
            candidate.gray_variance
            / max(max_variance, 1e-9),
            1.0,
        )

        gradient_score = min(
            candidate.gradient_mean
            / max(max_gradient, 1e-9),
            1.0,
        )

        candidate.quality_score = (
            0.50 * candidate.tissue_fraction
            + 0.25 * variance_score
            + 0.25 * gradient_score
        )


def select_candidates(
    candidates: List[Candidate],
    target: int,
) -> List[Candidate]:
    """
    Deterministic spatial/diversity selection:
    spatial coverage -> quality -> descriptor diversity -> separation.
    """
    if not candidates or target <= 0:
        return []

    target = min(target, len(candidates))
    score_candidates(candidates)

    by_tile: Dict[
        Tuple[int, int],
        List[Candidate],
    ] = {}

    for candidate in candidates:
        by_tile.setdefault(
            (
                candidate.tile_row,
                candidate.tile_col,
            ),
            [],
        ).append(candidate)

    selected: List[Candidate] = []

    for tile in sorted(by_tile):
        best = max(
            by_tile[tile],
            key=lambda c: (
                c.quality_score,
                c.tissue_fraction,
                c.gray_variance,
                -c.y,
                -c.x,
            ),
        )

        selected.append(best)

        if len(selected) >= target:
            break

    remaining = [
        c for c in candidates
        if c not in selected
    ]

    while remaining and len(selected) < target:

        def priority(candidate: Candidate):
            descriptor_gain = min(
                float(
                    np.linalg.norm(
                        candidate.descriptor
                        - other.descriptor
                    )
                )
                for other in selected
            )

            spatial_gain = min(
                math.hypot(
                    candidate.x - other.x,
                    candidate.y - other.y,
                )
                for other in selected
            )

            score = (
                candidate.quality_score
                + 0.35 * descriptor_gain
                + 0.000001 * spatial_gain
            )

            return (
                score,
                descriptor_gain,
                candidate.quality_score,
                spatial_gain,
                -candidate.y,
                -candidate.x,
            )

        choice = max(
            remaining,
            key=priority,
        )

        choice.selection_score = priority(choice)[0]
        selected.append(choice)
        remaining.remove(choice)

    for candidate in selected:
        candidate.selected = True
        candidate.redundancy_status = "selected"

    return selected


def select_tumor_candidates(
    candidates: List[Candidate],
    target: int = TARGET_PATCHES,
) -> List[Candidate]:
    """Preferred 250 tumor + 250 normal; no duplication."""
    tumor_candidates = [
        c for c in candidates
        if c.region_category == "tumor_dominant"
    ]

    normal_candidates = [
        c for c in candidates
        if c.region_category == "normal_tissue"
    ]

    clean_total = (
        len(tumor_candidates)
        + len(normal_candidates)
    )

    desired = min(target, clean_total)

    tumor_target = min(
        TUMOR_TARGET,
        len(tumor_candidates),
    )

    normal_target = min(
        NORMAL_TARGET,
        len(normal_candidates),
    )

    remaining = (
        desired
        - tumor_target
        - normal_target
    )

    if remaining > 0:
        extra_tumor = min(
            remaining,
            len(tumor_candidates) - tumor_target,
        )
        tumor_target += extra_tumor
        remaining -= extra_tumor

    if remaining > 0:
        extra_normal = min(
            remaining,
            len(normal_candidates) - normal_target,
        )
        normal_target += extra_normal

    selected = select_candidates(
        tumor_candidates,
        tumor_target,
    )

    selected.extend(
        select_candidates(
            normal_candidates,
            normal_target,
        )
    )

    return selected


# ============================================================
# 9. ANNOTATION-AWARE BOUNDED PRESELECTION
# ============================================================

def build_bounded_pool(
    screened: List[ScreenedItem],
    class_name: str,
) -> List[ScreenedItem]:
    """
    Build the expensive full-resolution pool.

    The important change from the previous version is that tumor
    candidates are classified by annotation BEFORE this step.

    Therefore a tumor candidate is not discarded simply because its
    preview variance is lower than a normal-tissue candidate.
    """
    by_tile: Dict[
        Tuple[int, int],
        List[ScreenedItem],
    ] = {}

    for item in screened:
        by_tile.setdefault(
            (
                item.tile_row,
                item.tile_col,
            ),
            [],
        ).append(item)

    bounded: List[ScreenedItem] = []

    for tile in sorted(by_tile):
        tile_items = by_tile[tile]

        if class_name != "tumor":
            ordered = sorted(
                tile_items,
                key=lambda item: (
                    item.preview_variance,
                    item.tissue_fraction,
                    -item.y,
                    -item.x,
                ),
                reverse=True,
            )

            bounded.extend(
                ordered[:CANDIDATES_PER_TILE]
            )
            continue

        tumor_items = [
            item for item in tile_items
            if item.region_category == "tumor_dominant"
        ]

        normal_items = [
            item for item in tile_items
            if item.region_category == "normal_tissue"
        ]

        # Boundary-mixed candidates are not sent to the expensive
        # full-resolution stage because they cannot be selected.
        tumor_items.sort(
            key=lambda item: (
                item.overlap,
                item.preview_variance,
                item.tissue_fraction,
                -item.y,
                -item.x,
            ),
            reverse=True,
        )

        normal_items.sort(
            key=lambda item: (
                item.preview_variance,
                item.tissue_fraction,
                -item.y,
                -item.x,
            ),
            reverse=True,
        )

        bounded.extend(
            tumor_items[:TUMOR_CANDIDATES_PER_TILE]
        )

        bounded.extend(
            normal_items[:NORMAL_CANDIDATES_PER_TILE]
        )

    return bounded


# ============================================================
# 10. MANIFEST
# ============================================================

def manifest_row(
    candidate: Candidate,
    row: Dict[str, str],
) -> Dict[str, object]:
    return {
        "wsi_id": row["wsi_id"],
        "source_wsi": row["path"],
        "split": row["split"],
        "class_name": (
            "ambiguous"
            if candidate.region_category == "boundary_mixed"
            else (
                "tumor"
                if candidate.label == 1
                else "normal"
            )
        ),
        "label": (
            ""
            if candidate.region_category == "boundary_mixed"
            else candidate.label
        ),
        "x": candidate.x,
        "y": candidate.y,
        "patch_size": PATCH_SIZE,
        "candidate_stride": CANDIDATE_STRIDE,
        "level": OPENSLIDE_LEVEL,
        "tissue_fraction": (
            f"{candidate.tissue_fraction:.6f}"
        ),
        "overlap_percentage": (
            f"{candidate.overlap:.6f}"
        ),
        "region_category": candidate.region_category,
        "quality_score": (
            f"{candidate.quality_score:.6f}"
        ),
        "selection_score": (
            f"{candidate.selection_score:.6f}"
        ),
        "spatial_tile_row": candidate.tile_row,
        "spatial_tile_column": candidate.tile_col,
        "annotation_region_id": (
            candidate.annotation_region_id
        ),
        "redundancy_status": (
            candidate.redundancy_status
        ),
        "selected": candidate.selected,
        "discarded_ambiguous": (
            candidate.region_category == "boundary_mixed"
        ),
        "discard_reason": candidate.discard_reason,
        "saved_path": candidate.saved_path,
    }


def read_manifest_rows() -> List[Dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []

    with MANIFEST_PATH.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def write_manifest_rows(
    rows: List[Dict[str, object]],
) -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = MANIFEST_PATH.with_suffix(".tmp")

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(MANIFEST_PATH)


def replace_manifest_for_wsi(
    wsi_id: str,
    new_rows: List[Dict[str, object]],
) -> None:
    """Replace only one WSI's records and preserve all others."""
    existing = read_manifest_rows()

    preserved = [
        row for row in existing
        if row.get("wsi_id") != wsi_id
    ]

    combined = preserved + [
        {
            field: row.get(field, "")
            for field in MANIFEST_FIELDS
        }
        for row in new_rows
    ]

    combined.sort(
        key=lambda row: (
            row.get("split", ""),
            row.get("wsi_id", ""),
            int(row.get("y", 0) or 0),
            int(row.get("x", 0) or 0),
        )
    )

    write_manifest_rows(combined)


# ============================================================
# 11. CLEAN ONE WSI
# ============================================================

def clean_wsi_outputs(
    wsi_id: str,
) -> int:
    """Delete only this WSI's PNGs and manifest records."""
    removed = 0

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        for class_name in (
            "normal",
            "tumor",
        ):
            class_dir = (
                OUTPUT_ROOT
                / split_name
                / class_name
            )

            if not class_dir.exists():
                continue

            for path in class_dir.glob(
                f"{wsi_id}_*.png"
            ):
                path.unlink()
                removed += 1

    existing = read_manifest_rows()

    preserved = [
        row for row in existing
        if row.get("wsi_id") != wsi_id
    ]

    if existing:
        write_manifest_rows(preserved)

    return removed


# ============================================================
# 12. DRY RUN
# ============================================================

def dry_run(
    wsi_id: str,
    requested_split: str,
    target: int = TARGET_PATCHES,
) -> None:
    """
    Full analysis-only dry run.

    Runs the complete candidate screening, annotation classification,
    bounded-pool construction, full-resolution quality checks, and
    final selection logic, but DOES NOT save PNGs or modify the manifest.
    """
    print("=" * 64)
    print("ANALYSIS-ONLY DRY RUN")
    print("=" * 64)
    print("No patch images will be saved.")
    print("No manifest will be modified.")
    print()

    result = extract_one_wsi(
        wsi_id,
        requested_split,
        target,
        analysis_only=True,
    )

    print()
    print("DRY RUN COMPLETE — no files were written.")
    print("=" * 64)


# ============================================================
# 13. EXTRACTION
# ============================================================

def extract_one_wsi(
    wsi_id: str,
    requested_split: str,
    target: int = TARGET_PATCHES,
    analysis_only: bool = False,
) -> Dict[str, object]:

    registry = load_registry()
    row = registry.get(wsi_id)

    if row is None:
        raise ValueError(
            f"Unknown WSI: {wsi_id}"
        )

    if row["split"] != requested_split:
        raise ValueError(
            f"Split mismatch: registry={row['split']} "
            f"requested={requested_split}"
        )

    if openslide is None:
        raise ImportError(
            "OpenSlide is required."
        )

    class_name = row["class_name"].lower()

    if class_name not in {"normal", "tumor"}:
        raise ValueError(
            f"Unsupported class_name: {row['class_name']}"
        )

    wsi_path = PROJECT_ROOT / row["path"]

    if not wsi_path.exists():
        raise FileNotFoundError(
            f"WSI not found: {wsi_path}"
        )

    polygons: List[AnnotationPolygon] = []

    if class_name == "tumor":
        annotation_value = row.get(
            "annotation_path",
            "",
        )

        if not annotation_value:
            raise ValueError(
                f"Tumor WSI {wsi_id} has no annotation_path."
            )

        annotation_path = (
            PROJECT_ROOT / annotation_value
        )

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation not found: {annotation_path}"
            )

        polygons = parse_annotation_xml(
            annotation_path
        )

        if not polygons:
            raise ValueError(
                f"No tumor annotation polygons parsed for {wsi_id}."
            )

        print(
            f"Tumor annotation polygons: {len(polygons)}"
        )

    total_grid = 0
    thumbnail_rejected = 0
    preview_quality_rejected = 0
    tissue_accepted = 0

    annotation_boundary = 0
    annotation_tumor = 0
    annotation_normal = 0

    fullres_quality_rejected = 0
    unreadable_patches = 0

    candidates: List[Candidate] = []

    with openslide.OpenSlide(
        str(wsi_path)
    ) as slide:

        width, height = slide.level_dimensions[
            OPENSLIDE_LEVEL
        ]

        preview_max = 4096

        preview_scale = min(
            preview_max / width,
            preview_max / height,
            1.0,
        )

        preview_width = max(
            1,
            int(width * preview_scale),
        )

        preview_height = max(
            1,
            int(height * preview_scale),
        )

        print(
            f"WSI dimensions: {width} x {height}"
        )

        print(
            f"Screening thumbnail: "
            f"{preview_width} x {preview_height}"
        )

        preview = slide.get_thumbnail(
            (
                preview_width,
                preview_height,
            )
        ).convert("RGB")

        # ----------------------------------------------------
        # Stage A: low-res screening
        # ----------------------------------------------------
        screened: List[ScreenedItem] = []

        for x, y in iter_coordinates(
            width,
            height,
        ):
            total_grid += 1

            px0 = int(
                x * preview_scale
            )
            py0 = int(
                y * preview_scale
            )

            preview_patch_size = max(
                8,
                int(
                    PATCH_SIZE * preview_scale
                ),
            )

            px1 = min(
                preview.width,
                px0 + preview_patch_size,
            )
            py1 = min(
                preview.height,
                py0 + preview_patch_size,
            )

            if px1 <= px0 or py1 <= py0:
                thumbnail_rejected += 1
                continue

            preview_patch = preview.crop(
                (
                    px0,
                    py0,
                    px1,
                    py1,
                )
            )

            fraction, variance = (
                low_res_quality_score(
                    preview_patch
                )
            )

            if fraction < TISSUE_THRESHOLD:
                thumbnail_rejected += 1
                continue

            if variance < MIN_GRAY_VARIANCE:
                preview_quality_rejected += 1
                continue

            tissue_accepted += 1

            tile_row, tile_col = tile_for(
                x,
                y,
                width,
                height,
            )

            screened.append(
                ScreenedItem(
                    x=x,
                    y=y,
                    tissue_fraction=fraction,
                    preview_variance=variance,
                    tile_row=tile_row,
                    tile_col=tile_col,
                )
            )

        print(
            f"Sliding-window candidates: {total_grid}"
        )
        print(
            f"Rejected by tissue threshold: {thumbnail_rejected}"
        )
        print(
            f"Rejected by low-res quality: {preview_quality_rejected}"
        )
        print(
            f"Passed low-res screening: {len(screened)}"
        )

        # ----------------------------------------------------
        # Stage B: EXACT annotation classification BEFORE
        # bounded candidate selection.
        # ----------------------------------------------------
        if class_name == "tumor":
            print(
                "Classifying screened candidates "
                "against tumor annotations..."
            )

            for index, item in enumerate(
                screened,
                start=1,
            ):
                item.overlap = (
                    patch_tumor_overlap_percentage(
                        (
                            item.x,
                            item.y,
                            PATCH_SIZE,
                            PATCH_SIZE,
                        ),
                        polygons,
                    )
                )

                if item.overlap == 0:
                    item.region_category = "normal_tissue"
                    item.label = 0
                    annotation_normal += 1

                elif item.overlap < 50:
                    item.region_category = "boundary_mixed"
                    item.label = 0
                    annotation_boundary += 1

                else:
                    item.region_category = "tumor_dominant"
                    item.label = 1
                    annotation_tumor += 1

                if index % 5000 == 0:
                    print(
                        "Annotation candidates classified: "
                        f"{index}/{len(screened)}"
                    )

            print(
                f"Annotation normal candidates: {annotation_normal}"
            )
            print(
                f"Annotation tumor-dominant candidates: "
                f"{annotation_tumor}"
            )
            print(
                f"Annotation boundary-mixed candidates: "
                f"{annotation_boundary}"
            )

        # ----------------------------------------------------
        # Stage C: annotation-aware bounded pool
        # ----------------------------------------------------
        bounded_screened = build_bounded_pool(
            screened,
            class_name,
        )

        print(
            "Bounded full-resolution candidate pool: "
            f"{len(bounded_screened)}"
        )

        if class_name == "tumor":
            print(
                "Bounded tumor candidates: "
                f"{sum(i.region_category == 'tumor_dominant' for i in bounded_screened)}"
            )
            print(
                "Bounded normal candidates: "
                f"{sum(i.region_category == 'normal_tissue' for i in bounded_screened)}"
            )

        # ----------------------------------------------------
        # Stage D: full-resolution features
        # ----------------------------------------------------
        total_bounded = len(bounded_screened)

        for index, item in enumerate(
            bounded_screened,
            start=1,
        ):
            try:
                patch = slide.read_region(
                    (
                        item.x,
                        item.y,
                    ),
                    OPENSLIDE_LEVEL,
                    (
                        PATCH_SIZE,
                        PATCH_SIZE,
                    ),
                ).convert("RGB")

                if patch.size != (
                    PATCH_SIZE,
                    PATCH_SIZE,
                ):
                    fullres_quality_rejected += 1
                    continue

                patch = gaussian_denoise(patch)

                variance, gradient, descriptor = (
                    feature_vector(patch)
                )

                if variance < MIN_GRAY_VARIANCE:
                    fullres_quality_rejected += 1
                    continue

            except Exception as exc:
                unreadable_patches += 1
                print(
                    f"Warning: unreadable patch "
                    f"({item.x}, {item.y}): {exc}"
                )
                continue

            candidate = Candidate(
                x=item.x,
                y=item.y,
                tissue_fraction=item.tissue_fraction,
                gray_variance=variance,
                gradient_mean=gradient,
                descriptor=descriptor,
                tile_row=item.tile_row,
                tile_col=item.tile_col,
                overlap=item.overlap,
                region_category=item.region_category,
                label=item.label,
            )

            if (
                class_name == "tumor"
                and candidate.region_category == "boundary_mixed"
            ):
                candidate.discard_reason = (
                    "boundary_mixed_excluded_from_clean_set"
                )

            candidates.append(candidate)

            if index % 250 == 0:
                print(
                    f"Full-resolution candidates processed: "
                    f"{index}/{total_bounded}"
                )

    # --------------------------------------------------------
    # Stage E: final selection
    # --------------------------------------------------------
    if class_name == "tumor":
        selected = select_tumor_candidates(
            candidates,
            target,
        )
    else:
        clean_candidates = [
            c for c in candidates
            if c.region_category != "boundary_mixed"
        ]

        selected = select_candidates(
            clean_candidates,
            target,
        )

    # --------------------------------------------------------
    # Analysis-only dry-run exit.
    # Everything above Stage F is the real selection pipeline.
    # Nothing has been written to disk or to the manifest.
    # --------------------------------------------------------
    if analysis_only:
        selected_tiles = {
            (
                c.tile_row,
                c.tile_col,
            )
            for c in selected
        }

        normal_selected = sum(
            c.label == 0
            for c in selected
        )

        tumor_selected = sum(
            c.label == 1
            for c in selected
        )

        boundary_selected = sum(
            c.region_category == "boundary_mixed"
            for c in selected
        )

        print("\n" + "=" * 64)
        print("ANALYSIS-ONLY SELECTION SUMMARY")
        print("=" * 64)
        print(f"WSI: {wsi_id}")
        print(f"Split: {requested_split}")
        print(f"Class: {class_name}")
        print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
        print(f"Candidate stride: {CANDIDATE_STRIDE}")
        print("Candidate overlap: 50%")
        print(f"Total sliding-window candidates: {total_grid}")
        print(f"Passed low-res tissue screening: {len(screened)}")
        print(
            f"Low-res tissue/quality rejected: "
            f"{thumbnail_rejected + preview_quality_rejected}"
        )
        print(
            f"Bounded full-resolution pool: "
            f"{len(bounded_screened)}"
        )
        print(
            f"Full-resolution candidates accepted: "
            f"{len(candidates)}"
        )
        print(
            f"Full-resolution quality rejected: "
            f"{fullres_quality_rejected}"
        )
        print(
            f"Unreadable candidates: "
            f"{unreadable_patches}"
        )

        if class_name == "tumor":
            bounded_tumor = sum(
                i.region_category == "tumor_dominant"
                for i in bounded_screened
            )
            bounded_normal = sum(
                i.region_category == "normal_tissue"
                for i in bounded_screened
            )

            print()
            print("ANNOTATION COUNTS — BEFORE BOUNDED SELECTION")
            print(
                f"Tumor-dominant candidates: "
                f"{annotation_tumor}"
            )
            print(
                f"Normal-tissue candidates: "
                f"{annotation_normal}"
            )
            print(
                f"Boundary-mixed candidates: "
                f"{annotation_boundary}"
            )

            print()
            print("BOUNDED POOL — AFTER ANNOTATION-AWARE SELECTION")
            print(
                f"Tumor candidates retained: "
                f"{bounded_tumor}"
            )
            print(
                f"Normal candidates retained: "
                f"{bounded_normal}"
            )

            print()
            print("FINAL SELECTION")
            print(
                f"Tumor selected: {tumor_selected}"
            )
            print(
                f"Normal selected: {normal_selected}"
            )
            print(
                f"Boundary selected: {boundary_selected}"
            )

        print(
            f"Final selected: {len(selected)}"
        )
        print(
            f"Target: {target}"
        )
        print(
            f"Shortfall: {max(0, target - len(selected))}"
        )
        print(
            f"Spatial tiles represented: "
            f"{len(selected_tiles)} / {GRID_ROWS * GRID_COLS}"
        )
        print("Files written: 0")
        print("Manifest modified: NO")
        print("=" * 64)

        return {
            "total_grid": total_grid,
            "screened": len(screened),
            "bounded_pool": len(bounded_screened),
            "fullres_candidates": len(candidates),
            "selected": len(selected),
            "normal_selected": normal_selected,
            "tumor_selected": tumor_selected,
            "boundary_selected": boundary_selected,
            "shortfall": max(0, target - len(selected)),
        }

    # --------------------------------------------------------
    # Stage F: save selected patches
    # --------------------------------------------------------
    output_dir = (
        OUTPUT_ROOT
        / requested_split
        / class_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_keys = {
        (c.x, c.y)
        for c in selected
    }

    manifest_rows: List[
        Dict[str, object]
    ] = []

    with openslide.OpenSlide(
        str(wsi_path)
    ) as slide:

        for candidate in candidates:

            if (
                candidate.x,
                candidate.y,
            ) in selected_keys:

                filename = (
                    f"{wsi_id}"
                    f"_x{candidate.x}"
                    f"_y{candidate.y}"
                    f"_label{candidate.label}.png"
                )

                path = output_dir / filename

                if path.exists():
                    raise FileExistsError(
                        f"Output already exists: {path}. "
                        f"Run with --clean-wsi."
                    )

                patch = slide.read_region(
                    (
                        candidate.x,
                        candidate.y,
                    ),
                    OPENSLIDE_LEVEL,
                    (
                        PATCH_SIZE,
                        PATCH_SIZE,
                    ),
                ).convert("RGB")

                patch = gaussian_denoise(patch)
                patch.save(path)

                candidate.saved_path = (
                    path.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                )

            manifest_rows.append(
                manifest_row(
                    candidate,
                    row,
                )
            )

    # --------------------------------------------------------
    # Stage G: cumulative manifest
    # --------------------------------------------------------
    replace_manifest_for_wsi(
        wsi_id,
        manifest_rows,
    )

    selected_tiles = {
        (
            c.tile_row,
            c.tile_col,
        )
        for c in selected
    }

    normal_selected = sum(
        c.label == 0
        for c in selected
    )

    tumor_selected = sum(
        c.label == 1
        for c in selected
    )

    boundary_selected = sum(
        c.region_category == "boundary_mixed"
        for c in selected
    )

    print("\n" + "=" * 64)
    print("EXTRACTION SUMMARY")
    print("=" * 64)
    print(f"WSI: {wsi_id}")
    print(f"Split: {requested_split}")
    print(f"Class: {class_name}")
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Candidate stride: {CANDIDATE_STRIDE}")
    print("Candidate overlap: 50%")
    print(f"Total sliding-window candidates: {total_grid}")
    print(f"Passed low-res tissue screening: {len(screened)}")
    print(f"Bounded full-resolution pool: {len(bounded_screened)}")
    print(
        f"Full-resolution candidates accepted: "
        f"{len(candidates)}"
    )
    print(
        f"Full-resolution quality rejected: "
        f"{fullres_quality_rejected}"
    )
    print(
        f"Unreadable candidates: {unreadable_patches}"
    )

    if class_name == "tumor":
        print(
            f"Tumor-dominant after annotation: "
            f"{annotation_tumor}"
        )
        print(
            f"Normal-tissue after annotation: "
            f"{annotation_normal}"
        )
        print(
            f"Boundary-mixed after annotation: "
            f"{annotation_boundary}"
        )
        print(
            f"Final selected tumor: {tumor_selected}"
        )
        print(
            f"Final selected normal: {normal_selected}"
        )
        print(
            f"Boundary selected: {boundary_selected}"
        )

    print(
        f"Final selected: {len(selected)}"
    )
    print(
        f"Target: {target}"
    )
    print(
        f"Shortfall: {max(0, target - len(selected))}"
    )
    print(
        f"Spatial tiles represented: "
        f"{len(selected_tiles)} / {GRID_ROWS * GRID_COLS}"
    )
    print(
        f"Output directory: {output_dir}"
    )
    print(
        f"Cumulative manifest: {MANIFEST_PATH}"
    )
    print("=" * 64)

    return {
        "total_grid": total_grid,
        "screened": len(screened),
        "bounded_pool": len(bounded_screened),
        "fullres_candidates": len(candidates),
        "selected": len(selected),
        "normal_selected": normal_selected,
        "tumor_selected": tumor_selected,
        "boundary_selected": boundary_selected,
        "shortfall": max(0, target - len(selected)),
    }


# ============================================================
# 14. COMMAND-LINE ENTRY POINT
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--wsi",
        required=True,
        help="Process exactly one registry WSI.",
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "validation", "test"),
        help="Expected registry split.",
    )

    parser.add_argument(
        "--target-patches",
        type=int,
        default=TARGET_PATCHES,
        help=(
            "Maximum final clean patches "
            f"(default: {TARGET_PATCHES})."
        ),
    )

    parser.add_argument(
        "--clean-wsi",
        action="store_true",
        help=(
            "Remove ONLY this WSI's existing patch files "
            "and manifest records before extraction."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the complete candidate-selection pipeline "
            "without saving patches or modifying the manifest."
        ),
    )

    args = parser.parse_args()

    if args.target_patches <= 0:
        parser.error(
            "--target-patches must be positive."
        )

    if args.dry_run:
        dry_run(
            args.wsi,
            args.split,
            args.target_patches,
        )
        return 0

    if args.clean_wsi:
        removed = clean_wsi_outputs(
            args.wsi
        )
        print(
            f"Cleaned {removed} existing patch file(s) "
            f"for {args.wsi}."
        )

    extract_one_wsi(
        args.wsi,
        args.split,
        args.target_patches,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())