"""
PathoVision AI — External Dataset Downloader.

Downloads independent, unseen external test data:
1. PatchCamelyon (PCam): 100 patches (50 normal, 50 tumor)
2. BreakHis: 100 patches (50 normal/benign, 50 tumor/malignant)
3. TCGA-BRCA: 2 Whole-Slide Images (.svs) from NCI GDC (1 Solid Tissue Normal, 1 Primary Tumor)

Kept strictly separated under `data/external_test/`.
STRICTLY NEVER USED FOR MODEL TRAINING OR TUNING.
"""

import io
import time
import logging
from pathlib import Path
from PIL import Image
import pandas as pd
import requests
import fsspec
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DownloadExternal")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external_test"

PCAM_DIR = EXTERNAL_ROOT / "patchcamelyon"
BREAKHIS_DIR = EXTERNAL_ROOT / "breakhis"
TCGA_DIR = EXTERNAL_ROOT / "tcga_brca_wsi"


def download_patchcamelyon(target_dir: Path, n_per_class: int = 50) -> list:
    """
    Downloads exactly n_per_class normal and n_per_class tumor patches from PatchCamelyon.
    """
    logger.info("=" * 60)
    logger.info(f"DOWNLOADING PATCHCAMELYON ({n_per_class} Normal, {n_per_class} Tumor)")
    logger.info("=" * 60)

    (target_dir / "normal").mkdir(parents=True, exist_ok=True)
    (target_dir / "tumor").mkdir(parents=True, exist_ok=True)

    url = "https://huggingface.co/datasets/1aurent/PatchCamelyon/resolve/main/data/test-00000-of-00002-bb04e6313f58efa0.parquet"
    fs, path = fsspec.core.url_to_fs(url)

    normal_count = 0
    tumor_count = 0
    entries = []

    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            if normal_count >= n_per_class and tumor_count >= n_per_class:
                break
            table = pf.read_row_group(rg)
            df = table.to_pandas()
            for idx, row in df.iterrows():
                is_tumor = bool(row["label"])
                if is_tumor and tumor_count < n_per_class:
                    img_dict = row["image"]
                    img_bytes = img_dict["bytes"]
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    # Scale from 96x96 to 224x224
                    im_224 = im.resize((224, 224), Image.Resampling.BILINEAR)
                    fname = f"pcam_test_tumor_{tumor_count:03d}.png"
                    out_path = target_dir / "tumor" / fname
                    im_224.save(out_path)
                    entries.append({
                        "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "dataset_source": "PatchCamelyon",
                        "label": 1,
                        "class_name": "Tumor",
                        "organ_site": "Axillary Sentinel Lymph Node",
                        "original_dimensions": "96x96",
                        "magnification": "10x equivalent (0.97 um/px)",
                        "expected_anomalies": "Metastatic tumor clusters, dense cellular displacement",
                    })
                    tumor_count += 1
                elif not is_tumor and normal_count < n_per_class:
                    img_dict = row["image"]
                    img_bytes = img_dict["bytes"]
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    im_224 = im.resize((224, 224), Image.Resampling.BILINEAR)
                    fname = f"pcam_test_normal_{normal_count:03d}.png"
                    out_path = target_dir / "normal" / fname
                    im_224.save(out_path)
                    entries.append({
                        "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "dataset_source": "PatchCamelyon",
                        "label": 0,
                        "class_name": "Normal",
                        "organ_site": "Axillary Sentinel Lymph Node",
                        "original_dimensions": "96x96",
                        "magnification": "10x equivalent (0.97 um/px)",
                        "expected_anomalies": "None (Benign germinal center / lymphoid tissue)",
                    })
                    normal_count += 1

    logger.info(f"PatchCamelyon download complete: {normal_count} normal, {tumor_count} tumor saved to {target_dir}")
    return entries


def download_breakhis(target_dir: Path, n_per_class: int = 50) -> list:
    """
    Downloads exactly n_per_class normal/benign and n_per_class tumor/malignant patches from BreakHis.
    """
    logger.info("=" * 60)
    logger.info(f"DOWNLOADING BREAKHIS ({n_per_class} Benign, {n_per_class} Malignant)")
    logger.info("=" * 60)

    (target_dir / "normal").mkdir(parents=True, exist_ok=True)
    (target_dir / "tumor").mkdir(parents=True, exist_ok=True)

    url = "https://huggingface.co/datasets/AntonXue/mcal-breakhis/resolve/main/data/train-00000-of-00001.parquet"
    fs, path = fsspec.core.url_to_fs(url)

    normal_count = 0
    tumor_count = 0
    entries = []

    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            if normal_count >= n_per_class and tumor_count >= n_per_class:
                break
            table = pf.read_row_group(rg)
            df = table.to_pandas()
            for idx, row in df.iterrows():
                lbl = int(row["label"])
                if lbl == 1 and tumor_count < n_per_class:
                    img_dict = row["image"]
                    img_bytes = img_dict["bytes"]
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    im_224 = im.resize((224, 224), Image.Resampling.BILINEAR) if im.size != (224, 224) else im
                    fname = f"breakhis_test_tumor_{tumor_count:03d}.png"
                    out_path = target_dir / "tumor" / fname
                    im_224.save(out_path)
                    entries.append({
                        "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "dataset_source": "BreakHis",
                        "label": 1,
                        "class_name": "Tumor",
                        "organ_site": "Primary Breast Tissue",
                        "original_dimensions": f"{im.size[0]}x{im.size[1]}",
                        "magnification": "Optical Microscope (40x / 100x / 200x)",
                        "expected_anomalies": "Ductal architectural distortion, nuclear pleomorphism, necrosis",
                    })
                    tumor_count += 1
                elif lbl == 0 and normal_count < n_per_class:
                    img_dict = row["image"]
                    img_bytes = img_dict["bytes"]
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    im_224 = im.resize((224, 224), Image.Resampling.BILINEAR) if im.size != (224, 224) else im
                    fname = f"breakhis_test_normal_{normal_count:03d}.png"
                    out_path = target_dir / "normal" / fname
                    im_224.save(out_path)
                    entries.append({
                        "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "dataset_source": "BreakHis",
                        "label": 0,
                        "class_name": "Normal",
                        "organ_site": "Primary Breast Tissue",
                        "original_dimensions": f"{im.size[0]}x{im.size[1]}",
                        "magnification": "Optical Microscope (40x / 100x / 200x)",
                        "expected_anomalies": "None (Benign fibroadenoma / adenosis / intact stroma)",
                    })
                    normal_count += 1

    logger.info(f"BreakHis download complete: {normal_count} normal, {tumor_count} tumor saved to {target_dir}")
    return entries


def download_tcga_brca_wsis(target_dir: Path) -> list:
    """
    Downloads 2 genuine TCGA-BRCA whole-slide images from NCI GDC:
      1 Solid Tissue Normal WSI
      1 Primary Tumor WSI
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING TCGA-BRCA WHOLE-SLIDE IMAGES (1 Normal, 1 Tumorous)")
    logger.info("=" * 60)

    (target_dir / "normal").mkdir(parents=True, exist_ok=True)
    (target_dir / "tumor").mkdir(parents=True, exist_ok=True)

    wsi_targets = [
        {
            "role": "tumor",
            "file_id": "3b8368d6-8fa6-4a53-81e2-2f9e82f1bc98",
            "filename": "TCGA-A7-A0DC-01B-04-BS4_PrimaryTumor.svs",
            "label": 1,
            "class_name": "Tumor",
            "sample_type": "Primary Tumor",
            "expected_anomalies": "Infiltrating carcinoma, high cellular density, pleomorphism, focal necrosis",
        },
        {
            "role": "normal",
            "file_id": "df4137d7-2b21-429b-90b8-9205dcc25e07",
            "filename": "TCGA-A7-A0DC-11A-04-TS4_SolidTissueNormal.svs",
            "label": 0,
            "class_name": "Normal",
            "sample_type": "Solid Tissue Normal",
            "expected_anomalies": "None (Normal resting lobules, ducts, and fibrous/adipose stroma)",
        },
    ]

    entries = []
    for item in wsi_targets:
        dest_path = target_dir / item["role"] / item["filename"]
        if dest_path.exists() and dest_path.stat().st_size > 10 * 1024 * 1024:
            logger.info(f"[EXISTS] TCGA slide already downloaded: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        else:
            url = f"https://api.gdc.cancer.gov/data/{item['file_id']}"
            logger.info(f"[STREAMING] Downloading {item['sample_type']} from GDC ({url}) -> {dest_path.name}...")
            t0 = time.time()
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_bytes = 0
                with open(dest_path, "wb") as f_out:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f_out.write(chunk)
                            total_bytes += len(chunk)
                            if total_bytes % (20 * 1024 * 1024) == 0:
                                logger.info(f"  Downloaded {total_bytes / 1e6:.1f} MB in {time.time() - t0:.1f}s...")
            logger.info(f"[DONE] Saved {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s.")

        entries.append({
            "image_path": str(dest_path.relative_to(PROJECT_ROOT)),
            "dataset_source": "TCGA-BRCA",
            "label": item["label"],
            "class_name": item["class_name"],
            "organ_site": "Primary Breast Invasive Carcinoma / Normal Margin",
            "original_dimensions": "Gigapixel Pyramidal Scan (.svs)",
            "magnification": "20x / 40x Digital Aperio Scanner",
            "expected_anomalies": item["expected_anomalies"],
        })

    return entries


def main():
    logger.info("Initializing External Dataset Ingestion Pipeline...")
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)

    all_entries = []

    # 1. Download PatchCamelyon (100 patches: 50 normal, 50 tumor)
    pcam_entries = download_patchcamelyon(PCAM_DIR, n_per_class=50)
    all_entries.extend(pcam_entries)

    # 2. Download BreakHis (100 patches: 50 normal, 50 tumor)
    breakhis_entries = download_breakhis(BREAKHIS_DIR, n_per_class=50)
    all_entries.extend(breakhis_entries)

    # 3. Download TCGA-BRCA WSIs (1 normal, 1 tumorous)
    tcga_entries = download_tcga_brca_wsis(TCGA_DIR)
    all_entries.extend(tcga_entries)

    # 4. Save Master External Manifest
    df_manifest = pd.DataFrame(all_entries)
    manifest_csv = EXTERNAL_ROOT / "external_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    logger.info(f"\nSaved Consolidated External Manifest: {manifest_csv} ({len(df_manifest)} items)")

    # 5. Summary Table
    print("\n" + "=" * 80)
    print("EXTERNAL TEST DATASET INVENTORY SUMMARY")
    print("=" * 80)
    summary_counts = df_manifest.groupby(["dataset_source", "class_name"]).size().reset_index(name="count")
    print(summary_counts.to_string(index=False))
    print("=" * 80)
    print("All datasets strictly isolated under data/external_test/ (NEVER trained on).")


if __name__ == "__main__":
    main()
