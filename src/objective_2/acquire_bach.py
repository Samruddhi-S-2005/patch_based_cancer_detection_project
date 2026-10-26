"""
BACH / ICIAR 2018 Dataset Acquisition Script
=============================================
Downloads and extracts the BACH breast cancer histology dataset from
HuggingFace (1aurent/BACH), which mirrors the official ICIAR 2018
Grand Challenge Task A Microscopy dataset (Zenodo: 10.5281/zenodo.3632034).

Dataset facts (VERIFIED from parquet schema):
  - 462 total images: 362 train + 100 test
  - 4 diagnostic classes + Unknown: Benign(0), InSitu(1), Invasive(2), Normal(3), Unknown(4)
  - Image format: TIFF, 2048x1536 px, RGB
  - Label type: IMAGE-LEVEL (one class per image — NOT pixel/region annotations)
  - Source: 204 patients (per Zenodo abstract)
  - License: CC BY-NC-ND 4.0
  
Output:
  data/ANOMALIES/bach_iciar2018/
    images/
      Benign/   (100 images from train set)
      InSitu/   (100 images from train set)
      Invasive/ (100 images from train set)
      Normal/   (100 images from train set)
    metadata/
      bach_label_map.json
    bach_dataset_inventory.csv
    bach_metadata_summary.json
"""
import os
import io
import json
import csv
import time
import pyarrow.parquet as pq
from pathlib import Path
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\Samruddhi\Projects\patch_based_cancer_detection_project")
ANOMALY_DIR  = PROJECT_ROOT / "data" / "ANOMALIES" / "bach_iciar2018"
CACHE_DIR    = Path(r"C:\Users\samru\.cache\huggingface\hub\datasets--1aurent--BACH")

LABEL_NAMES = {0: "Benign", 1: "InSitu", 2: "Invasive", 3: "Normal", 4: "Unknown"}

# Chunk files for both splits
TRAIN_CHUNKS = [f"data/train-{i:05d}-of-00015" for i in range(15)]
TEST_CHUNKS  = [f"data/test-{i:05d}-of-00004"  for i in range(4)]

# ── Create output dirs ───────────────────────────────────────────────────────
for lbl in LABEL_NAMES.values():
    (ANOMALY_DIR / "images" / lbl).mkdir(parents=True, exist_ok=True)
(ANOMALY_DIR / "metadata").mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def find_cached_parquet(prefix: str):
    """Find a cached parquet file matching a prefix."""
    for snap in CACHE_DIR.glob("snapshots/*/data/*.parquet"):
        if snap.stem.startswith(prefix.replace("data/", "")):
            return snap
    return None

def extract_from_parquet(parquet_path: Path, split_prefix: str, inventory: list):
    """Read a parquet chunk and save images to class folders."""
    pf = pq.ParquetFile(str(parquet_path))
    rg = pf.read_row_group(0)
    d  = rg.to_pydict()
    count = 0
    for img_struct, label in zip(d["image"], d["label"]):
        lbl_name = LABEL_NAMES.get(label, "Unknown")
        orig_path = img_struct.get("path", "unknown.tif") or "unknown.tif"
        img_bytes = img_struct["bytes"]
        
        out_path = ANOMALY_DIR / "images" / lbl_name / orig_path
        if not out_path.exists():
            with open(str(out_path), "wb") as f:
                f.write(img_bytes)
        
        # Get image metadata
        try:
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            mode = img.mode
            fmt  = img.format or "TIFF"
        except Exception:
            w, h, mode, fmt = 0, 0, "UNKNOWN", "UNKNOWN"
        
        inventory.append({
            "image_id":         orig_path.replace(".tif", ""),
            "filename":         orig_path,
            "class_label":      lbl_name,
            "class_int":        label,
            "split":            split_prefix,
            "width":            w,
            "height":           h,
            "format":           fmt,
            "mode":             mode,
            "magnification":    "20x",
            "pixel_resolution_um_px": 0.42,
            "annotation_type": "image_level_diagnosis",
            "annotation_available": True,
            "annotation_coordinates_available": False,
            "source":           "1aurent/BACH (HuggingFace) / Zenodo 10.5281/zenodo.3632034",
            "quality_status":   "ok" if (w > 0) else "unreadable",
        })
        count += 1
    return count

def download_and_extract(hf_path: str, split_prefix: str, inventory: list):
    """Download a single parquet chunk from HuggingFace and extract images."""
    from huggingface_hub import hf_hub_download
    import warnings
    warnings.filterwarnings("ignore")
    
    local_path = hf_hub_download(
        repo_id="1aurent/BACH",
        filename=hf_path + ".parquet",  # need full filename
        repo_type="dataset"
    )
    return extract_from_parquet(Path(local_path), split_prefix, inventory)

# ── Main ─────────────────────────────────────────────────────────────────────
inventory = []
total_written = 0
errors = []

# List all HuggingFace chunk filenames (full names with hashes)
from huggingface_hub import HfApi
api = HfApi()
all_hf_files = sorted(api.list_repo_files("1aurent/BACH", repo_type="dataset"))
train_files = [f for f in all_hf_files if f.startswith("data/train-") and f.endswith(".parquet")]
test_files  = [f for f in all_hf_files if f.startswith("data/test-")  and f.endswith(".parquet")]

print(f"Train files: {len(train_files)}")
print(f"Test files:  {len(test_files)}")
print(f"Output dir:  {ANOMALY_DIR}")
print()

all_chunks = [(f, "train") for f in train_files] + [(f, "test") for f in test_files]

for i, (hf_path, split_name) in enumerate(all_chunks):
    stem = Path(hf_path).stem  # e.g. train-00000-of-00015-45961313a94a1995

    # Check if already cached
    cached = find_cached_parquet(f"data/{stem}")
    if cached:
        print(f"[{i+1:2d}/{len(all_chunks)}] CACHED  {stem[:40]} -> extracting...")
        try:
            n = extract_from_parquet(cached, split_name, inventory)
            total_written += n
            print(f"           -> {n} images extracted")
        except Exception as e:
            print(f"           -> ERROR: {e}")
            errors.append((stem, str(e)))
    else:
        print(f"[{i+1:2d}/{len(all_chunks)}] DOWNLOADING {stem[:40]}...")
        try:
            from huggingface_hub import hf_hub_download
            import warnings
            warnings.filterwarnings("ignore")
            local_path = hf_hub_download(
                repo_id="1aurent/BACH",
                filename=hf_path,
                repo_type="dataset"
            )
            n = extract_from_parquet(Path(local_path), split_name, inventory)
            total_written += n
            print(f"           -> {n} images extracted")
        except Exception as e:
            print(f"           -> ERROR: {e}")
            errors.append((stem, str(e)))

print(f"\nTotal images extracted: {total_written}")
print(f"Errors: {len(errors)}")
for stem, err in errors:
    print(f"  {stem}: {err}")

# ── Count per class ──────────────────────────────────────────────────────────
from collections import Counter
class_counts = Counter(row["class_label"] for row in inventory)
print("\nClass distribution:")
for cls, cnt in sorted(class_counts.items()):
    print(f"  {cls}: {cnt}")

# ── Write inventory CSV ──────────────────────────────────────────────────────
inv_path = ANOMALY_DIR / "bach_dataset_inventory.csv"
if inventory:
    fieldnames = list(inventory[0].keys())
    with open(str(inv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)
    print(f"\nInventory saved: {inv_path} ({len(inventory)} rows)")

# ── Write summary JSON ───────────────────────────────────────────────────────
summary = {
    "dataset_name": "BACH / ICIAR 2018 Task A Microscopy",
    "source": "1aurent/BACH (HuggingFace) mirroring Zenodo doi:10.5281/zenodo.3632034",
    "license": "CC BY-NC-ND 4.0",
    "total_images": total_written,
    "image_format": "TIFF",
    "image_dimensions": "2048x1536",
    "magnification": "20x",
    "pixel_resolution_um_px": 0.42,
    "label_type": "image_level_diagnosis",
    "annotation_type": "IMAGE-LEVEL ONLY — no pixel/region coordinates",
    "class_distribution": {k: v for k, v in sorted(class_counts.items())},
    "label_map": LABEL_NAMES,
    "splits": {
        "train": sum(1 for r in inventory if r["split"] == "train"),
        "test":  sum(1 for r in inventory if r["split"] == "test"),
    },
    "corrupted_images": sum(1 for r in inventory if r["quality_status"] == "unreadable"),
    "errors": len(errors),
}
summary_path = ANOMALY_DIR / "bach_metadata_summary.json"
with open(str(summary_path), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved:   {summary_path}")

# ── Write label map ──────────────────────────────────────────────────────────
lbl_path = ANOMALY_DIR / "metadata" / "bach_label_map.json"
with open(str(lbl_path), "w") as f:
    json.dump(LABEL_NAMES, f, indent=2)
print(f"Label map saved: {lbl_path}")
print("\nDONE.")
