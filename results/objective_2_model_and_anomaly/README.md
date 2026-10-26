# Objective 2: Deep Learning Modeling & Anomaly Analysis

## Overview
Objective 2 implements the primary EfficientNet-B0 deep convolutional neural network for patch-level cancer detection and performs multi-head anomaly profiling and risk analysis across targeted histopathological abnormalities.

## Architecture & Transfer Learning
- **Primary Backbone:** EfficientNet-B0 pre-trained on ImageNet, fine-tuned with transfer learning on 14,719 CAMELYON16 patches.
- **Sigmoid Binary Classification:** Outputs probability $P(\text{Tumor}) \in [0.0, 1.0]$.
- **Anomaly Score & Risk Categorization:**
  - **Low Risk:** Anomaly Score $\in [0.00, 0.30)$
  - **Medium Risk:** Anomaly Score $\in [0.30, 0.70)$
  - **High Risk:** Anomaly Score $\in [0.70, 1.00]$
- **Targeted Histopathological Abnormalities:**
  1. *Nuclear Pleomorphism / Nuclear Atypia:* Marked cellular variability in size, shape, and chromatin density.
  2. *Ductal Structure Distortion:* In-situ and invasive disruption of glandular architecture (BACH proxy, $\tau = 0.4038$).
  3. *Tumor Cell Density Clusters:* Hyper-cellular neoplastic overcrowding $\ge 10$ nuclei (NuCLS proxy, $\tau = 0.5315$).
  4. *Tumor Necrosis & Cellular Debris:* Eosinophilic necrotic dead zones and granular debris (BCSS proxy, $\tau = 0.2715$).

## Directory Artifacts
- `feature_extraction/`: Pre-classifier embeddings across 4 models (1280-D, 2048-D, 1024-D, 4096-D) and PCA projections.
- `risk_analysis/`:
  - `anomaly_risk_distribution.png`: Validation split risk score density distribution.
  - `test_anomaly_risk_distribution.png`: Held-out test split risk score density distribution.
- `anomaly_analysis/`: Linear probe metrics, threshold tuning, and validation logs.
- `scientific_evidences/`:
  - `04_feature_vector_sample.png`: Deep feature vector representation.
  - `05_feature_PCA.png`: PCA latent clustering of normal vs tumor tissue.
  - `13_risk_score_examples.png`: Representative Low, Medium, and High risk patch examples.
- `reports/explicit_morphology_analysis.md`: Authoritative audit distinguishing learned visual features from physical continuous measurements.
