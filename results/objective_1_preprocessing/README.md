# Objective 1: Whole-Slide Image Patch Extraction and Essential Preprocessing

## Overview
Objective 1 establishes a deterministic, reproducible patch-based pipeline for breast cancer histopathology from CAMELYON16 whole-slide images (WSIs).

## Preprocessing Sequence
1. **WSI Ingestion & Tiling:** 60 CAMELYON16 WSIs processed at OpenSlide Level 0 (20X equivalent, 224×224 px, adaptive 112–224 px stride).
2. **Annotation-Aware Overlap:** Patches consult XML lesion polygons ($\ge 50\%$ tumor, $0\%$ normal, $0-50\%$ boundary discarded).
3. **Quality Control & Noise Filtering:**
   - Tissue Ratio $\ge 0.20$ (Otsu thresholding & HSV saturation $> 20$)
   - White Ratio $\le 0.80$ (Acellular glass exclusion)
   - Laplacian Blur Variance $\ge 20.0$ (Focus gate)
   - Gaussian Blur filtering applied for high-frequency noise suppression
4. **Standardization & Augmentation:**
   - Two-stage normalization: Min-Max scaling and Z-Score standardization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
   - Augmentation (horizontal flip, vertical flip, random rotation, zooming) applied only to training data.

## Directory Artifacts
- `level_wise_pipeline/`: Step-by-step visual records:
  - `01_wsi_thumbnail.png`: Whole slide macro scan
  - `02_raw_extracted_patch.png`: 224×224 raw patch
  - `03a_grayscale.png`: Grayscale conversion
  - `03b_gaussian_blur.png`: Gaussian noise reduction
  - `03c_thresholding.png`: Otsu tissue mask
  - `04_augmented_patch.png`: Spatial data augmentation
  - `05_normalized_patch.png`: Mean-Std normalized patch
  - `06_grad_cam_heatmap.png`: Explainability activation
- `scientific_evidences/`: Visual proof of WSI tessellation, normal vs tumor morphology, dataset split integrity:
  - `01_WSI_to_patches.png`
  - `03_normal_vs_tumor.png`
  - `05_dataset_statistics.png`
  - `06_split_integrity.png`
- `manifests/final_dataset_manifest.csv`: 14,803 patch audit records.
- `manifests/wsi_registry.csv`: 60 slide metadata and train/val/test splits.
- `quality_control/training_patch_preprocessing_audit.csv`: Comprehensive per-patch QC audit.
- `quality_control/training_patch_quality_control.xlsx`: 7-sheet executive Excel workbook.
- `masks/tissue_mask_manifest.csv`: Manifest for 14,719 generated binary tissue masks.
- `stage_visualizations/`: Composite 4-stage visualizations for representative sample patches.
