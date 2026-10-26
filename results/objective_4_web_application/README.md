# Objective 4: Web Application & Clinical Deployment

## Overview
Objective 4 delivers an interactive, clinical-decision-support web application (`app.py`) built with Streamlit, featuring secure role-based access control, single-patch analysis, gigapixel whole-slide imaging (WSI) reconstruction, multi-model benchmarking, and QuPath export capabilities.

## Core Application Modules
1. **User Authentication & Session Management (`src/auth.py`):**
   - PBKDF2-HMAC-SHA256 password hashing (100,000 iterations).
   - 1-Click quick sign-in for Lead Researcher / Student and Academic Reviewer / Examiner.
   - Private user workspaces and analysis logging.
2. **Single Patch Analysis (Section 1):**
   - Upload custom patches or choose pre-verified benchmark samples.
   - Outputs: Primary Verdict (`NORMAL HEALTHY TISSUE` vs `MALIGNANT TUMOR TISSUE`), Anomaly Score ($P(\text{Tumor})$ with visual meter), Risk Tier (`Low`, `Medium`, `High Risk`), and Predicted Anomaly Type.
   - Grad-CAM visual heatmaps with color scale indicators (`jet`, `turbo`, `inferno`, `viridis`).
   - Multi-model evaluation and one-click JSON/CSV clinical reports.
3. **Whole-Slide Image (WSI) Analysis (Section 2):**
   - 6-Step pipeline: Input WSI $\to$ 224×224 patch extraction $\to$ Preprocessing/focus masking $\to$ Deep feature analysis $\to$ Anomaly phenotyping $\to$ Spatial heatmap & Patient Slide Diagnostic Summary.
   - Side-by-side Level 0 optical scan and continuous spatial heatmap.
   - Interactive hotspot tile drill-down with live Grad-CAM.
   - Multi-architecture spatial heatmap comparison across all 4 models.
   - QuPath integration: Export GeoJSON annotation layers and TSV measurement tables.
4. **Model Evaluation & Feature Analysis (Sections 3 & 4):**
   - Dynamic Plotly interactive confusion matrices, comparative ROC/PR curves, and 2D/3D PCA latent space projections.

## Directory Artifacts
- `wsi_analysis/`:
  - `tumor_058_reconstructed_diagnostic_heatmap.png` & `tumor_058_spatial_diagnostic_dashboard.png`
  - `normal_004_reconstructed_diagnostic_heatmap.png` & `normal_004_spatial_diagnostic_dashboard.png`
  - `tumor_058_summary.json` & `normal_004_summary.json`
- `patch_analysis/`:
  - Clinical diagnostic panels and Grad-CAM overlays.
- `reports/`:
  - Sample exported clinical reports, JSON summaries, and QuPath files.
