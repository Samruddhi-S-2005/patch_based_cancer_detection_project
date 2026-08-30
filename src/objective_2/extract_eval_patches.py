"""
Objective 2 Phase 2B: validation/test patch extraction.

Isolated Objective-1-parity extractor. Does not write to
data/PATCHES/REDESIGNED, PROCESSED, CLEANED, or result/.

Does not import or invoke extract_patches_optimized.main.
Uses read-only imports from annotation_parser only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import openslide
except ImportError:
    openslide = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotation_parser import (  # noqa: E402
    AnnotationPolygon,
    parse_annotation_xml,
    patch_tumor_overlap_percentage,
)


# ============================================================
# CONFIGURATION (Objective-1 parity)
# ============================================================

REGISTRY_PATH = PROJECT_ROOT / "data" / "wsi_registry.csv"

PATCH_SIZE = 224
CANDIDATE_STRIDE = 112
OPENSLIDE_LEVEL = 0
TARGET_PATCHES = 500

TISSUE_THRESHOLD = 0.50
HSV_SATURATION_THRESHOLD = 25
HSV_VALUE_MIN = 30
HSV_VALUE_MAX = 230

MIN_GRAY_VARIANCE = 5.0

CANDIDATES_PER_TILE = 32
TUMOR_CANDIDATES_PER_TILE = 16
NORMAL_CANDIDATES_PER_TILE = 16

GRID_ROWS = 8
GRID_COLS = 8

APPLY_GAUSSIAN_BLUR = False
GAUSSIAN_SIGMA = 0.5

TUMOR_TARGET = 250
NORMAL_TARGET = 250

# Stage-2-equivalent QC (qc_and_preprocess.py)
QC_MIN_TISSUE_RATIO = 0.20
QC_MAX_WHITE_RATIO = 0.80
QC_MIN_LAPLACIAN_VARIANCE = 20.0

# CLEANED extra rule: normal-folder patches only
QC_NORMAL_FOLDER_STAGE3_BLUR_MIN = 30.0

ALLOWED_SPLITS = ("validation", "test")

FROZEN_PATHS = [
    PROJECT_ROOT / "data" / "PATCHES" / "CLEANED",
    PROJECT_ROOT / "data" / "PATCHES" / "PROCESSED",
    PROJECT_ROOT / "data" / "PATCHES" / "REDESIGNED",
    PROJECT_ROOT / "result",
    PROJECT_ROOT / "src" / "extract_patches_optimized.py",
    PROJECT_ROOT / "src" / "qc_and_preprocess.py",
    PROJECT_ROOT / "src" / "second_stage_qc.py",
    PROJECT_ROOT / "src" / "annotation_parser.py",
    PROJECT_ROOT / "src" / "wsi_manager.py",
]

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
    "tissue_ratio",
    "white_ratio",
    "blur_score",
    "stage3_tissue_ratio",
    "stage3_white_ratio",
    "stage3_blur_score",
    "final_decision",
    "saved_path",
]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ScreenedItem:
    x: int
    y: int
    tissue_fraction: float
    preview_variance: float
    tile_row: int
    tile_col: int
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
    tissue_ratio: float = 0.0
    white_ratio: float = 0.0
    blur_score: float = 0.0
    stage3_tissue_ratio: float = 0.0
    stage3_white_ratio: float = 0.0
    stage3_blur_score: float = 0.0
    final_decision: str = ""


# ============================================================
# PATH HELPERS
# ============================================================

def output_root_for_split(split: str) -> Path:
    return PROJECT_ROOT / "data" / "PATCHES" / split.upper()


def manifest_path_for_split(split: str) -> Path:
    return output_root_for_split(split) / f"{split}_patch_manifest.csv"


# ============================================================
# REGISTRY / SPLIT SAFETY
# ============================================================

def load_registry() -> Dict[str, Dict[str, str]]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    with REGISTRY_PATH.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("WSI registry is empty.")

    required = {"wsi_id", "class_name", "split", "path", "status"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(
            "Registry missing required columns: " + ", ".join(sorted(missing))
        )

    ids = [row["wsi_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Registry contains duplicate WSI IDs.")

    return {row["wsi_id"]: row for row in rows}


def validate_eval_wsi(wsi_id: str, requested_split: str) -> Dict[str, str]:
    if requested_split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Requested split must be validation or test, got {requested_split!r}."
        )

    registry = load_registry()
    row = registry.get(wsi_id)
    if row is None:
        raise ValueError(f"Unknown WSI: {wsi_id}")

    status = (row.get("status") or "").strip().upper()
    if status != "VALID":
        raise ValueError(
            f"WSI {wsi_id} status is {status!r}, expected VALID."
        )

    registry_split = (row.get("split") or "").strip().lower()
    if registry_split == "train":
        raise ValueError(
            f"Refusing train WSI {wsi_id}. This pipeline is validation/test only."
        )

    if registry_split not in ALLOWED_SPLITS:
        raise ValueError(
            f"WSI {wsi_id} has unsupported registry split {registry_split!r}."
        )

    if registry_split != requested_split:
        raise ValueError(
            f"Split mismatch for {wsi_id}: "
            f"registry={registry_split} requested={requested_split}."
        )

    class_name = row["class_name"].lower()
    if class_name not in {"normal", "tumor"}:
        raise ValueError(f"Unsupported class_name: {row['class_name']}")

    return row


def list_wsi_ids_for_split(requested_split: str) -> List[str]:
    if requested_split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Requested split must be validation or test, got {requested_split!r}."
        )
    registry = load_registry()
    ids = [
        wsi_id
        for wsi_id, row in registry.items()
        if (row.get("split") or "").strip().lower() == requested_split
        and (row.get("status") or "").strip().upper() == "VALID"
    ]
    return sorted(ids)


# ============================================================
# COORDINATE / SPATIAL HELPERS
# ============================================================

def iter_coordinates(width: int, height: int) -> Iterable[Tuple[int, int]]:
    max_x = width - PATCH_SIZE
    max_y = height - PATCH_SIZE
    if max_x < 0 or max_y < 0:
        return
    for y in range(0, max_y + 1, CANDIDATE_STRIDE):
        for x in range(0, max_x + 1, CANDIDATE_STRIDE):
            yield x, y


def tile_for(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    return (
        min(GRID_ROWS - 1, y * GRID_ROWS // max(height, 1)),
        min(GRID_COLS - 1, x * GRID_COLS // max(width, 1)),
    )


# ============================================================
# LOW-RESOLUTION SCREENING (Objective 1)
# ============================================================

def tissue_fraction(image: Image.Image) -> float:
    small = image.resize((56, 56), Image.Resampling.BILINEAR)
    hsv = np.asarray(small.convert("HSV"), dtype=np.uint8)
    tissue = (
        (hsv[:, :, 1] >= HSV_SATURATION_THRESHOLD)
        & (hsv[:, :, 2] >= HSV_VALUE_MIN)
        & (hsv[:, :, 2] <= HSV_VALUE_MAX)
    )
    return float(tissue.mean())


def gray_variance(image: Image.Image) -> float:
    small = image.resize((56, 56), Image.Resampling.BILINEAR).convert("L")
    return float(np.asarray(small, dtype=np.float32).var())


def low_res_quality_score(image: Image.Image) -> Tuple[float, float]:
    return tissue_fraction(image), gray_variance(image)


def gaussian_denoise(image: Image.Image) -> Image.Image:
    if not APPLY_GAUSSIAN_BLUR:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_SIGMA))


def feature_vector(image: Image.Image) -> Tuple[float, float, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    variance = float(gray.var())
    gradient_y, gradient_x = np.gradient(gray)
    gradient = float(np.hypot(gradient_x, gradient_y).mean())
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    hsv_mean = hsv.mean(axis=(0, 1)) / 255.0
    hsv_std = hsv.std(axis=(0, 1)) / 255.0
    thumbnail = (
        np.asarray(
            image.convert("L").resize((16, 16), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).ravel()
        / 255.0
    )
    descriptor = np.concatenate((hsv_mean, hsv_std, thumbnail))
    return variance, gradient, descriptor


# ============================================================
# EVALUATION QC (not train-hardcoded scripts)
# ============================================================

def stage2_qc_metrics(image: Image.Image) -> Tuple[float, float, float]:
    """Match src/qc_and_preprocess.py."""
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    tissue_mask = ~((saturation < 25) & (value > 210))
    tissue_ratio = float(np.mean(tissue_mask))
    white_mask = (rgb[:, :, 0] > 220) & (rgb[:, :, 1] > 220) & (rgb[:, :, 2] > 220)
    white_ratio = float(np.mean(white_mask))
    gray = np.asarray(image.convert("L"))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return tissue_ratio, white_ratio, blur_score


def stage3_qc_metrics(image: Image.Image) -> Tuple[float, float, float]:
    """Match src/second_stage_qc.py (in-memory RGB equivalent of cv2.imread)."""
    rgb = np.asarray(image.convert("RGB"))
    white_mask = np.all(rgb >= 235, axis=2)
    white_ratio = float(np.mean(white_mask))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    tissue_ratio = float(np.mean(hsv[:, :, 1] > 20))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return tissue_ratio, white_ratio, blur_score


def evaluation_qc_reasons(
    wsi_class_name: str,
    tissue_ratio: float,
    white_ratio: float,
    blur_score: float,
    stage3_blur_score: float,
) -> List[str]:
    reasons: List[str] = []
    if tissue_ratio < QC_MIN_TISSUE_RATIO:
        reasons.append("low_tissue")
    if white_ratio > QC_MAX_WHITE_RATIO:
        reasons.append("mostly_white")
    if blur_score < QC_MIN_LAPLACIAN_VARIANCE:
        reasons.append("blur")
    if (
        wsi_class_name == "normal"
        and stage3_blur_score < QC_NORMAL_FOLDER_STAGE3_BLUR_MIN
    ):
        reasons.append("normal_extreme_blur")
    return reasons


# ============================================================
# QUALITY + DIVERSITY SELECTION (Objective 1)
# ============================================================

def score_candidates(candidates: List[Candidate]) -> None:
    if not candidates:
        return
    max_variance = max(c.gray_variance for c in candidates)
    max_gradient = max(c.gradient_mean for c in candidates)
    for candidate in candidates:
        variance_score = min(
            candidate.gray_variance / max(max_variance, 1e-9),
            1.0,
        )
        gradient_score = min(
            candidate.gradient_mean / max(max_gradient, 1e-9),
            1.0,
        )
        candidate.quality_score = (
            0.50 * candidate.tissue_fraction
            + 0.25 * variance_score
            + 0.25 * gradient_score
        )


def select_candidates(candidates: List[Candidate], target: int) -> List[Candidate]:
    if not candidates or target <= 0:
        return []
    target = min(target, len(candidates))
    score_candidates(candidates)

    by_tile: Dict[Tuple[int, int], List[Candidate]] = {}
    for candidate in candidates:
        by_tile.setdefault(
            (candidate.tile_row, candidate.tile_col),
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

    remaining = [c for c in candidates if c not in selected]
    while remaining and len(selected) < target:

        def priority(candidate: Candidate):
            descriptor_gain = min(
                float(np.linalg.norm(candidate.descriptor - other.descriptor))
                for other in selected
            )
            spatial_gain = min(
                math.hypot(candidate.x - other.x, candidate.y - other.y)
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

        choice = max(remaining, key=priority)
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
    tumor_candidates = [
        c for c in candidates if c.region_category == "tumor_dominant"
    ]
    normal_candidates = [
        c for c in candidates if c.region_category == "normal_tissue"
    ]
    clean_total = len(tumor_candidates) + len(normal_candidates)
    desired = min(target, clean_total)
    tumor_target = min(TUMOR_TARGET, len(tumor_candidates))
    normal_target = min(NORMAL_TARGET, len(normal_candidates))
    remaining = desired - tumor_target - normal_target
    if remaining > 0:
        extra_tumor = min(remaining, len(tumor_candidates) - tumor_target)
        tumor_target += extra_tumor
        remaining -= extra_tumor
    if remaining > 0:
        extra_normal = min(remaining, len(normal_candidates) - normal_target)
        normal_target += extra_normal
    selected = select_candidates(tumor_candidates, tumor_target)
    selected.extend(select_candidates(normal_candidates, normal_target))
    return selected


def build_bounded_pool(
    screened: List[ScreenedItem],
    class_name: str,
) -> List[ScreenedItem]:
    by_tile: Dict[Tuple[int, int], List[ScreenedItem]] = {}
    for item in screened:
        by_tile.setdefault((item.tile_row, item.tile_col), []).append(item)

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
            bounded.extend(ordered[:CANDIDATES_PER_TILE])
            continue

        tumor_items = [
            item for item in tile_items
            if item.region_category == "tumor_dominant"
        ]
        normal_items = [
            item for item in tile_items
            if item.region_category == "normal_tissue"
        ]
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
        bounded.extend(tumor_items[:TUMOR_CANDIDATES_PER_TILE])
        bounded.extend(normal_items[:NORMAL_CANDIDATES_PER_TILE])
    return bounded


# ============================================================
# MANIFEST
# ============================================================

def manifest_row(candidate: Candidate, row: Dict[str, str]) -> Dict[str, object]:
    return {
        "wsi_id": row["wsi_id"],
        "source_wsi": row["path"],
        "split": row["split"],
        "class_name": (
            "ambiguous"
            if candidate.region_category == "boundary_mixed"
            else ("tumor" if candidate.label == 1 else "normal")
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
        "tissue_fraction": f"{candidate.tissue_fraction:.6f}",
        "overlap_percentage": f"{candidate.overlap:.6f}",
        "region_category": candidate.region_category,
        "quality_score": f"{candidate.quality_score:.6f}",
        "selection_score": f"{candidate.selection_score:.6f}",
        "spatial_tile_row": candidate.tile_row,
        "spatial_tile_column": candidate.tile_col,
        "annotation_region_id": candidate.annotation_region_id,
        "redundancy_status": candidate.redundancy_status,
        "selected": candidate.selected,
        "discarded_ambiguous": candidate.region_category == "boundary_mixed",
        "discard_reason": candidate.discard_reason,
        "tissue_ratio": f"{candidate.tissue_ratio:.6f}",
        "white_ratio": f"{candidate.white_ratio:.6f}",
        "blur_score": f"{candidate.blur_score:.6f}",
        "stage3_tissue_ratio": f"{candidate.stage3_tissue_ratio:.6f}",
        "stage3_white_ratio": f"{candidate.stage3_white_ratio:.6f}",
        "stage3_blur_score": f"{candidate.stage3_blur_score:.6f}",
        "final_decision": candidate.final_decision,
        "saved_path": candidate.saved_path,
    }


def read_manifest_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_manifest_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def replace_manifest_for_wsi(
    split: str,
    wsi_id: str,
    new_rows: List[Dict[str, object]],
) -> None:
    path = manifest_path_for_split(split)
    existing = read_manifest_rows(path)
    preserved = [row for row in existing if row.get("wsi_id") != wsi_id]
    combined = preserved + [
        {field: row.get(field, "") for field in MANIFEST_FIELDS}
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
    write_manifest_rows(path, combined)


def clean_wsi_outputs(split: str, wsi_id: str) -> int:
    removed = 0
    output_root = output_root_for_split(split)
    for class_name in ("normal", "tumor"):
        class_dir = output_root / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.glob(f"{wsi_id}_*.png"):
            path.unlink()
            removed += 1
    path = manifest_path_for_split(split)
    existing = read_manifest_rows(path)
    preserved = [row for row in existing if row.get("wsi_id") != wsi_id]
    if existing:
        write_manifest_rows(path, preserved)
    return removed


# ============================================================
# EXTRACTION
# ============================================================

def extract_one_wsi(
    wsi_id: str,
    requested_split: str,
    target: int = TARGET_PATCHES,
    analysis_only: bool = False,
) -> Dict[str, object]:
    row = validate_eval_wsi(wsi_id, requested_split)

    if openslide is None:
        raise ImportError("OpenSlide is required.")
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for evaluation QC.")

    class_name = row["class_name"].lower()
    wsi_path = PROJECT_ROOT / row["path"]
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    polygons: List[AnnotationPolygon] = []
    if class_name == "tumor":
        annotation_value = row.get("annotation_path", "")
        if not annotation_value:
            raise ValueError(f"Tumor WSI {wsi_id} has no annotation_path.")
        annotation_path = PROJECT_ROOT / annotation_value
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation not found: {annotation_path}")
        polygons = parse_annotation_xml(annotation_path)
        if not polygons:
            raise ValueError(f"No tumor annotation polygons parsed for {wsi_id}.")
        print(f"Tumor annotation polygons: {len(polygons)}")

    total_grid = 0
    thumbnail_rejected = 0
    preview_quality_rejected = 0
    annotation_boundary = 0
    annotation_tumor = 0
    annotation_normal = 0
    fullres_quality_rejected = 0
    unreadable_patches = 0
    qc_rejected = 0
    saved_count = 0
    candidates: List[Candidate] = []

    output_dir = output_root_for_split(requested_split) / class_name

    with openslide.OpenSlide(str(wsi_path)) as slide:
        width, height = slide.level_dimensions[OPENSLIDE_LEVEL]
        preview_max = 4096
        preview_scale = min(preview_max / width, preview_max / height, 1.0)
        preview_width = max(1, int(width * preview_scale))
        preview_height = max(1, int(height * preview_scale))

        print(f"WSI dimensions: {width} x {height}")
        print(f"Screening thumbnail: {preview_width} x {preview_height}")

        preview = slide.get_thumbnail((preview_width, preview_height)).convert("RGB")

        screened: List[ScreenedItem] = []
        for x, y in iter_coordinates(width, height):
            total_grid += 1
            px0 = int(x * preview_scale)
            py0 = int(y * preview_scale)
            preview_patch_size = max(8, int(PATCH_SIZE * preview_scale))
            px1 = min(preview.width, px0 + preview_patch_size)
            py1 = min(preview.height, py0 + preview_patch_size)
            if px1 <= px0 or py1 <= py0:
                thumbnail_rejected += 1
                continue
            preview_patch = preview.crop((px0, py0, px1, py1))
            fraction, variance = low_res_quality_score(preview_patch)
            if fraction < TISSUE_THRESHOLD:
                thumbnail_rejected += 1
                continue
            if variance < MIN_GRAY_VARIANCE:
                preview_quality_rejected += 1
                continue
            tile_row, tile_col = tile_for(x, y, width, height)
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

        print(f"Sliding-window candidates: {total_grid}")
        print(f"Rejected by tissue threshold: {thumbnail_rejected}")
        print(f"Rejected by low-res quality: {preview_quality_rejected}")
        print(f"Passed low-res screening: {len(screened)}")

        if class_name == "tumor":
            print("Classifying screened candidates against tumor annotations...")
            for index, item in enumerate(screened, start=1):
                item.overlap = patch_tumor_overlap_percentage(
                    (item.x, item.y, PATCH_SIZE, PATCH_SIZE),
                    polygons,
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
            print(f"Annotation normal candidates: {annotation_normal}")
            print(f"Annotation tumor-dominant candidates: {annotation_tumor}")
            print(f"Annotation boundary-mixed candidates: {annotation_boundary}")

        bounded_screened = build_bounded_pool(screened, class_name)
        print(
            "Bounded full-resolution candidate pool: "
            f"{len(bounded_screened)}"
        )

        total_bounded = len(bounded_screened)
        for index, item in enumerate(bounded_screened, start=1):
            try:
                patch = slide.read_region(
                    (item.x, item.y),
                    OPENSLIDE_LEVEL,
                    (PATCH_SIZE, PATCH_SIZE),
                ).convert("RGB")
                if patch.size != (PATCH_SIZE, PATCH_SIZE):
                    fullres_quality_rejected += 1
                    continue
                patch = gaussian_denoise(patch)
                variance, gradient, descriptor = feature_vector(patch)
                if variance < MIN_GRAY_VARIANCE:
                    fullres_quality_rejected += 1
                    continue
                s2_tissue, s2_white, s2_blur = stage2_qc_metrics(patch)
                s3_tissue, s3_white, s3_blur = stage3_qc_metrics(patch)
            except Exception as exc:
                unreadable_patches += 1
                print(f"Warning: unreadable patch ({item.x}, {item.y}): {exc}")
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
                tissue_ratio=s2_tissue,
                white_ratio=s2_white,
                blur_score=s2_blur,
                stage3_tissue_ratio=s3_tissue,
                stage3_white_ratio=s3_white,
                stage3_blur_score=s3_blur,
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

    if class_name == "tumor":
        selected = select_tumor_candidates(candidates, target)
    else:
        clean_candidates = [
            c for c in candidates if c.region_category != "boundary_mixed"
        ]
        selected = select_candidates(clean_candidates, target)

    selected_keys = {(c.x, c.y) for c in selected}

    keep_keys = set()
    for candidate in candidates:
        if (candidate.x, candidate.y) not in selected_keys:
            candidate.final_decision = "not_selected"
            continue
        reasons = evaluation_qc_reasons(
            class_name,
            candidate.tissue_ratio,
            candidate.white_ratio,
            candidate.blur_score,
            candidate.stage3_blur_score,
        )
        if reasons:
            candidate.final_decision = "REMOVE"
            extra = ";".join(reasons)
            if candidate.discard_reason:
                candidate.discard_reason = f"{candidate.discard_reason};{extra}"
            else:
                candidate.discard_reason = extra
            qc_rejected += 1
        else:
            candidate.final_decision = "KEEP"
            keep_keys.add((candidate.x, candidate.y))

    summary_base = {
        "wsi_id": wsi_id,
        "registry_split": row["split"],
        "requested_split": requested_split,
        "class_name": class_name,
        "total_grid": total_grid,
        "screened": len(screened),
        "bounded_pool": len(bounded_screened),
        "fullres_candidates": len(candidates),
        "fullres_quality_rejected": fullres_quality_rejected,
        "unreadable_patches": unreadable_patches,
        "selected_before_qc": len(selected),
        "qc_rejected": qc_rejected,
        "annotation_normal": annotation_normal,
        "annotation_tumor": annotation_tumor,
        "annotation_boundary": annotation_boundary,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path_for_split(requested_split)),
    }

    if analysis_only:
        print("\nANALYSIS-ONLY: no files written.")
        print_extraction_summary(
            summary_base
            | {
                "saved": 0,
                "label0_saved": 0,
                "label1_saved": 0,
                "region_counts_saved": {},
            }
        )
        return summary_base | {"saved": 0}

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []

    with openslide.OpenSlide(str(wsi_path)) as slide:
        for candidate in candidates:
            if (candidate.x, candidate.y) in keep_keys:
                filename = (
                    f"{wsi_id}"
                    f"_x{candidate.x}"
                    f"_y{candidate.y}"
                    f"_label{candidate.label}.png"
                )
                path = output_dir / filename
                if path.exists():
                    raise FileExistsError(
                        f"Output already exists: {path}. Run with --clean-wsi."
                    )
                patch = slide.read_region(
                    (candidate.x, candidate.y),
                    OPENSLIDE_LEVEL,
                    (PATCH_SIZE, PATCH_SIZE),
                ).convert("RGB")
                patch = gaussian_denoise(patch)
                patch.save(path)
                candidate.saved_path = path.relative_to(PROJECT_ROOT).as_posix()
                saved_count += 1
            manifest_rows.append(manifest_row(candidate, row))

    replace_manifest_for_wsi(requested_split, wsi_id, manifest_rows)

    label0_saved = sum(
        1
        for c in candidates
        if c.final_decision == "KEEP" and c.label == 0
    )
    label1_saved = sum(
        1
        for c in candidates
        if c.final_decision == "KEEP" and c.label == 1
    )
    region_counts_saved: Dict[str, int] = {}
    for c in candidates:
        if c.final_decision == "KEEP":
            region_counts_saved[c.region_category] = (
                region_counts_saved.get(c.region_category, 0) + 1
            )

    result = summary_base | {
        "saved": saved_count,
        "label0_saved": label0_saved,
        "label1_saved": label1_saved,
        "region_counts_saved": region_counts_saved,
    }
    print_extraction_summary(result)
    return result


def print_extraction_summary(result: Dict[str, object]) -> None:
    print("\n" + "=" * 64)
    print("EVALUATION EXTRACTION SUMMARY")
    print("=" * 64)
    print(f"WSI: {result.get('wsi_id')}")
    print(f"Registry split: {result.get('registry_split')}")
    print(f"Requested split: {result.get('requested_split')}")
    print(f"Class: {result.get('class_name')}")
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Candidate stride: {CANDIDATE_STRIDE}")
    print("Candidate overlap: 50%")
    print(f"Sliding-window candidates: {result.get('total_grid')}")
    print(f"Passed low-res screening: {result.get('screened')}")
    print(f"Bounded full-resolution pool: {result.get('bounded_pool')}")
    print(f"Full-resolution candidates accepted: {result.get('fullres_candidates')}")
    print(f"Full-resolution quality rejected: {result.get('fullres_quality_rejected')}")
    print(f"Unreadable candidates: {result.get('unreadable_patches')}")
    print(f"Selected before QC: {result.get('selected_before_qc')}")
    print(f"QC rejected (from selected): {result.get('qc_rejected')}")
    print(f"Final saved: {result.get('saved')}")
    print(f"Saved label 0: {result.get('label0_saved')}")
    print(f"Saved label 1: {result.get('label1_saved')}")
    print(f"Saved region categories: {result.get('region_counts_saved')}")
    print(f"Output directory: {result.get('output_dir')}")
    print(f"Manifest: {result.get('manifest_path')}")
    print("=" * 64)


# ============================================================
# OBJECTIVE-1 PRESERVATION SNAPSHOT / AUDIT
# ============================================================

def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def snapshot_frozen_assets() -> Dict[str, Dict[str, object]]:
    snapshot: Dict[str, Dict[str, object]] = {}
    for root in FROZEN_PATHS:
        files = list(_iter_files(root))
        digest = hashlib.sha256()
        records = []
        for path in sorted(files, key=lambda p: p.as_posix().lower()):
            stat = path.stat()
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            records.append((rel, stat.st_size, int(stat.st_mtime_ns)))
            digest.update(rel.encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        snapshot[str(root)] = {
            "file_count": len(files),
            "signature": digest.hexdigest(),
            "records": records,
        }
    return snapshot


def compare_frozen_snapshots(
    before: Dict[str, Dict[str, object]],
    after: Dict[str, Dict[str, object]],
) -> List[str]:
    issues: List[str] = []
    for key in FROZEN_PATHS:
        name = str(key)
        b = before.get(name)
        a = after.get(name)
        if b is None or a is None:
            issues.append(f"Missing snapshot for {name}")
            continue
        if b["file_count"] != a["file_count"]:
            issues.append(
                f"{name}: file count {b['file_count']} -> {a['file_count']}"
            )
        if b["signature"] != a["signature"]:
            issues.append(f"{name}: content/mtime signature changed")
    return issues


def parse_filename_meta(name: str) -> Optional[Tuple[str, int, int, int]]:
    if not name.endswith(".png"):
        return None
    stem = name[:-4]
    try:
        wsi_part, x_part, y_part, label_part = stem.rsplit("_", 3)
    except ValueError:
        return None
    if not x_part.startswith("x") or not y_part.startswith("y"):
        return None
    if not label_part.startswith("label"):
        return None
    try:
        x = int(x_part[1:])
        y = int(y_part[1:])
        label = int(label_part[5:])
    except ValueError:
        return None
    if label not in (0, 1):
        return None
    return wsi_part, x, y, label


def audit_eval_split(split: str) -> Dict[str, object]:
    registry = load_registry()
    output_root = output_root_for_split(split)
    manifest = manifest_path_for_split(split)
    issues: List[str] = []
    png_paths: List[Path] = []
    for class_name in ("normal", "tumor"):
        class_dir = output_root / class_name
        if class_dir.exists():
            png_paths.extend(sorted(class_dir.glob("*.png")))

    names = [p.name for p in png_paths]
    if len(names) != len(set(names)):
        issues.append("Duplicate filenames in saved PNGs.")

    seen_keys = set()
    train_ids = {
        wsi_id
        for wsi_id, row in registry.items()
        if (row.get("split") or "").strip().lower() == "train"
    }
    other_eval = "test" if split == "validation" else "validation"
    other_ids = {
        wsi_id
        for wsi_id, row in registry.items()
        if (row.get("split") or "").strip().lower() == other_eval
    }

    for path in png_paths:
        meta = parse_filename_meta(path.name)
        if meta is None:
            issues.append(f"Invalid filename: {path.name}")
            continue
        wsi_id, x, y, label = meta
        if wsi_id in train_ids:
            issues.append(f"Train WSI leaked into {split}: {path.name}")
        if wsi_id in other_ids:
            issues.append(f"{other_eval} WSI leaked into {split}: {path.name}")
        row = registry.get(wsi_id)
        if row is None:
            issues.append(f"Unknown WSI in filename: {path.name}")
        elif (row.get("split") or "").strip().lower() != split:
            issues.append(
                f"Filename WSI split mismatch: {path.name} "
                f"registry={row.get('split')}"
            )
        key = (wsi_id, x, y, label)
        if key in seen_keys:
            issues.append(f"Duplicate WSI/x/y/label: {key}")
        seen_keys.add(key)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if image.size != (PATCH_SIZE, PATCH_SIZE):
                issues.append(f"Bad size {image.size}: {path.name}")
            if rgb.mode != "RGB":
                issues.append(f"Non-RGB: {path.name}")
            if image.mode not in {"RGB", "RGBA"}:
                issues.append(f"Unexpected mode {image.mode}: {path.name}")

    saved_rows = []
    if manifest.exists():
        saved_rows = [
            r for r in read_manifest_rows(manifest)
            if str(r.get("final_decision", "")).upper() == "KEEP"
        ]
        for row in saved_rows:
            wsi_id = row.get("wsi_id", "")
            if wsi_id in train_ids:
                issues.append(f"Train WSI in manifest KEEP row: {wsi_id}")
            if (row.get("split") or "").strip().lower() != split:
                issues.append(
                    f"Manifest split mismatch: {wsi_id} {row.get('split')}"
                )
            if (row.get("region_category") or "") == "boundary_mixed":
                issues.append(f"Saved boundary_mixed: {row.get('saved_path')}")
            saved_path = row.get("saved_path") or ""
            abs_path = PROJECT_ROOT / saved_path
            if not saved_path or not abs_path.exists():
                issues.append(f"Missing saved_path: {saved_path}")
            else:
                meta = parse_filename_meta(abs_path.name)
                if meta is None:
                    issues.append(f"KEEP filename invalid: {abs_path.name}")
                else:
                    _, _, _, file_label = meta
                    try:
                        manifest_label = int(row.get("label", ""))
                    except ValueError:
                        issues.append(f"Non-integer manifest label: {row}")
                    else:
                        if file_label != manifest_label:
                            issues.append(
                                f"Label mismatch filename vs manifest: "
                                f"{abs_path.name} vs {manifest_label}"
                            )
            if str(row.get("selected", "")).lower() not in {"true", "1"}:
                issues.append(f"KEEP row not selected: {row.get('saved_path')}")

    other_root = output_root_for_split(other_eval)
    other_exists = other_root.exists() and any(other_root.rglob("*.png"))

    return {
        "split": split,
        "png_count": len(png_paths),
        "keep_rows": len(saved_rows),
        "issues": issues,
        "other_eval_has_pngs": other_exists,
        "other_eval_split": other_eval,
    }


# ============================================================
# CLI
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi", help="Process exactly one registry WSI.")
    parser.add_argument(
        "--split",
        required=True,
        choices=("validation", "test"),
        help="Expected registry split.",
    )
    parser.add_argument(
        "--all-in-split",
        action="store_true",
        help="Process every VALID WSI assigned to --split.",
    )
    parser.add_argument(
        "--target-patches",
        type=int,
        default=TARGET_PATCHES,
        help=f"Maximum selected patches before QC (default: {TARGET_PATCHES}).",
    )
    parser.add_argument(
        "--clean-wsi",
        action="store_true",
        help="Remove only this evaluation WSI's PNGs and manifest rows first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run selection/QC accounting without writing patches or manifests.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit existing evaluation outputs for --split; do not extract.",
    )
    args = parser.parse_args()

    if args.target_patches <= 0:
        parser.error("--target-patches must be positive.")

    if args.audit:
        report = audit_eval_split(args.split)
        print("=" * 64)
        print("EVALUATION AUDIT")
        print("=" * 64)
        for key, value in report.items():
            if key != "issues":
                print(f"{key}: {value}")
        if report["issues"]:
            print("ISSUES:")
            for issue in report["issues"]:
                print(f"  - {issue}")
            return 1
        print("No issues found.")
        return 0

    if bool(args.wsi) == bool(args.all_in_split):
        parser.error("Specify exactly one of --wsi or --all-in-split.")

    wsi_ids = (
        list_wsi_ids_for_split(args.split)
        if args.all_in_split
        else [args.wsi]
    )

    for wsi_id in wsi_ids:
        validate_eval_wsi(wsi_id, args.split)
        if args.dry_run:
            extract_one_wsi(
                wsi_id,
                args.split,
                args.target_patches,
                analysis_only=True,
            )
            continue
        if args.clean_wsi:
            removed = clean_wsi_outputs(args.split, wsi_id)
            print(f"Cleaned {removed} existing eval patch file(s) for {wsi_id}.")
        extract_one_wsi(wsi_id, args.split, args.target_patches)

    return 0


if __name__ == "__main__":
    sys.exit(main())
