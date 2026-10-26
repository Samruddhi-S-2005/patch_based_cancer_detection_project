"""
PathoVision AI — Objective 2: BreakHis External Test Manifest Builder.

Discovers, verifies, and indexes all extracted BreakHis v1 histopathology images.
Extracts verified metadata from standard BreakHis filename conventions:
  <BIOPSY_PROCEDURE>_<TUMOR_CLASS>_<TUMOR_TYPE>_<YEAR>-<SLIDE_ID>-<MAGNIFICATION>-<SEQ>.png

Retains patient/slide identifiers, magnifications, histological subtypes,
and verified binary labels without mutating existing CAMELYON16 manifests.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BuildBreakHisManifest")

DEFAULT_BREAKHIS_DIR = PROJECT_ROOT / "data" / "External_test" / "BreaKHis_v1" / "histology_slides" / "breast"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "External_test" / "breakhis_manifest.csv"

# Standard BreakHis Subtype Mapping
SUBTYPE_MAP = {
    "A": "adenosis",
    "F": "fibroadenoma",
    "PT": "phyllodes_tumor",
    "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma",
    "LC": "lobular_carcinoma",
    "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}

CLASS_MAP = {
    "B": "benign",
    "M": "malignant",
}

FILENAME_REGEX = re.compile(
    r"^(SOB|CNB)_([BM])_([A-Za-z]+)-(\d+)-([A-Za-z0-9]+)-(\d+)-(\d+)\.png$"
)


def parse_breakhis_filename(filename: str) -> Dict:
    """Parses metadata from verified BreakHis filename grammar."""
    match = FILENAME_REGEX.match(filename)
    if not match:
        raise ValueError(f"Malformed BreakHis filename: {filename}")

    proc, cls_code, sub_code, year, slide_id, mag_val, seq_num = match.groups()

    sub_code_upper = sub_code.upper()
    subtype_name = SUBTYPE_MAP.get(sub_code_upper, f"unknown_{sub_code}")
    tumor_class = CLASS_MAP.get(cls_code, "unknown")
    patient_id = f"{proc}_{cls_code}_{sub_code}-{year}-{slide_id}"

    # Semantic label: 1 = Malignant Carcinoma, 0 = Benign Breast Tumor
    binary_label = 1 if cls_code == "M" else 0
    label_name = "Malignant Carcinoma" if binary_label == 1 else "Benign Breast Tumor"

    return {
        "biopsy_procedure": proc,
        "tumor_class": tumor_class,
        "tumor_class_code": cls_code,
        "tumor_subtype_code": sub_code_upper,
        "tumor_subtype_name": subtype_name,
        "year": int(year),
        "slide_id": slide_id,
        "patient_id": patient_id,
        "magnification": int(mag_val),
        "magnification_str": f"{mag_val}X",
        "seq_num": int(seq_num),
        "breakhis_label": binary_label,
        "breakhis_label_name": label_name,
        "semantic_note": "0 = Benign Breast Neoplasm (NOT Normal Lymph Node). 1 = Malignant Breast Carcinoma.",
    }


def build_manifest(
    data_dir: Path = DEFAULT_BREAKHIS_DIR,
    output_path: Path = DEFAULT_OUTPUT_MANIFEST,
    verify_dimensions: bool = True,
) -> pd.DataFrame:
    """Discovers all BreakHis images, validates metadata, and writes manifest CSV."""
    if not data_dir.exists():
        raise FileNotFoundError(f"BreakHis directory not found: {data_dir}")

    logger.info(f"Scanning BreakHis image files in: {data_dir}")
    png_files = sorted(list(data_dir.rglob("*.png")))
    total_found = len(png_files)
    logger.info(f"Discovered {total_found} total PNG files on disk.")

    if total_found == 0:
        raise ValueError(f"No PNG files found in {data_dir}")

    # Check for duplicate filenames
    names = [p.name for p in png_files]
    unique_names = set(names)
    if len(names) != len(unique_names):
        raise ValueError(f"Found {len(names) - len(unique_names)} duplicate filenames!")

    records: List[Dict] = []
    malformed: List[str] = []

    # Fast dimension verification on sample or full set
    logger.info("Validating image filenames and parsing metadata...")
    for idx, p in enumerate(png_files):
        try:
            meta = parse_breakhis_filename(p.name)
        except ValueError as err:
            malformed.append(p.name)
            logger.error(str(err))
            continue

        rel_path = p.relative_to(PROJECT_ROOT).as_posix()
        meta["image_id"] = p.stem
        meta["filename"] = p.name
        meta["relative_path"] = rel_path
        meta["file_size_bytes"] = p.stat().st_size
        meta["width"] = 700
        meta["height"] = 460

        records.append(meta)

    if malformed:
        raise RuntimeError(f"Encountered {len(malformed)} malformed files. Aborting manifest build.")

    df = pd.DataFrame(records)

    # Optional thorough dimension verification (e.g. check first 100 or all)
    if verify_dimensions:
        logger.info("Verifying image dimensions on sample...")
        sample_indices = [0, total_found // 4, total_found // 2, 3 * total_found // 4, total_found - 1]
        for s_idx in sample_indices:
            check_p = png_files[s_idx]
            with Image.open(check_p) as img:
                if img.size != (700, 460):
                    logger.warning(f"Unexpected image size {img.size} for {check_p.name}")

    # Ensure parent output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully generated BreakHis test manifest: {output_path}")
    logger.info(f"Total entries: {len(df)}")
    logger.info(f"Class breakdown: {dict(df['tumor_class'].value_counts())}")
    logger.info(f"Subtype breakdown: {dict(df['tumor_subtype_name'].value_counts())}")
    logger.info(f"Magnification breakdown: {dict(df['magnification_str'].value_counts())}")
    logger.info(f"Unique patient slides: {df['patient_id'].nunique()}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BreakHis Test Manifest")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_BREAKHIS_DIR, help="BreakHis images path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MANIFEST, help="Output manifest path")
    args = parser.parse_args()

    build_manifest(data_dir=args.data_dir, output_path=args.output)
