# PathoVision AI — Objective-Wise Results and Artifact Directory Index

This directory provides a neatly structured, objective-by-objective organization of all experimental outputs, manifests, quality-control logs, feature analyses, level-wise records, scientific evidences, and evaluation benchmarks for the major project:
**“Patch-Based Unimodal Cancer Anomaly Detection in Tissue Images Using Deep Learning Models”**

Department of Computer Science & Engineering, Bapuji Institute of Engineering and Technology (BIET), Davanagere - VTU (2025–2026).

---

## Directory Navigation

- [Objective 1: WSI Patch Extraction & Preprocessing](objective_1_preprocessing/README.md)
  - `manifests/`: Master dataset and WSI registry manifests (60 WSIs, 14,803 patches)
  - `masks/`: Deterministic binary tissue masks manifest
  - `quality_control/`: Per-patch preprocessing audit CSV, group summaries, and executive Excel workbook
  - `stage_visualizations/`: Layerwise 4-panel visual composites for representative patches
  - `level_wise_pipeline/`: Step-by-step visual pipeline records (01_wsi_thumbnail to 06_grad_cam_heatmap)
  - `scientific_evidences/`: Visual proof of WSI tessellation, normal vs tumor morphology, dataset split integrity

- [Objective 2: Deep Learning Modeling & Anomaly Analysis](objective_2_model_and_anomaly/README.md)
  - `feature_extraction/`: Pre-classifier embeddings across 4 models (1280, 2048, 1024, 4096 dimensions) and 2D/3D PCA projections
  - `anomaly_analysis/`: Linear probe metrics, validation logs, and predictions for targeted abnormalities
  - `risk_analysis/`: Model-based anomaly score and risk tier distributions (Low, Medium, High Risk)
  - `scientific_evidences/`: Deep feature vectors, latent PCA clustering, and risk score examples
  - `reports/explicit_morphology_analysis.md`: Detailed audit distinguishing learned visual features from physical measurements

- [Objective 3: Comparative Evaluation & Explainable AI](objective_3_evaluation_and_xai/README.md)
  - `metrics/`: Master benchmark comparison table (CSV & JSON) evaluating EfficientNet-B0, ResNet-50, DenseNet-121, and VGG-16
  - `confusion_matrices/`: Confusion matrices for all 4 trained architectures and test cohorts
  - `roc_pr_curves/`: Comparative 4-model ROC and Precision-Recall curves
  - `gradcam/`: Diagnostic case studies with multi-layer Grad-CAM heatmaps for targeted anomalies
  - `predictions/`: Complete held-out per-patch prediction records ($N=2,881$)
  - `scientific_evidences/`: Comparative evaluation plots, AUC curves, and sensitivity benchmarks

- [Objective 4: Web Application & Clinical Deployment](objective_4_web_application/README.md)
  - `wsi_analysis/`: Reconstructed spatial heatmaps, spatial dashboards, and diagnostic summaries
  - `patch_analysis/`: Clinical single-patch diagnostic panels and Grad-CAM overlays
  - `reports/`: Exported clinical reports in JSON, CSV, QuPath GeoJSON, and QuPath TSV formats
  - Overview of Streamlit workstation deployment (`app.py`), secure authentication (`src/auth.py`), and live verification logs.
