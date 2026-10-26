# PathoVision AI: Diagnostic Methodology, Multi-Dataset Grounding & Threshold Justification

> **Authoritative Research Reference Document**  
> **Location:** `docs/research/methodology_and_dataset_provenance.md`  
> **Status:** Preserved Project Methodology & Scientific Provenance Ledger  
> **Date:** October 2026

---

## 1. Hierarchical 4-Layer Diagnostic Protocol

PathoVision AI decomposes digital histopathology evaluation into a disciplined, four-layer hierarchical protocol designed to mirror pathologist review while preventing false positives from non-tissue artifacts:

```
Input Patch / WSI Tile (224 × 224 px)
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Tissue Quality & Focus Verification                │
│ • Grayscale Laplacian Variance (Sharpness ≥ 100.0)          │
│ • Non-white tissue coverage (≥ 15.0% - 20.0%)               │
│ • Action: Discards empty slide glass and blurry tiles        │
└──────────────────────────────┬──────────────────────────────┘
                               │ Viable Tissue Tile
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Primary Malignancy Screening (CAMELYON16)          │
│ • Fine-tuned Deep CNN Backbone (EfficientNet-B0)            │
│ • Class-balanced BCE Loss (pos_weight = 8.2572)             │
│ • Sigmoid malignancy score: P(Tumor) ∈ [0, 1]               │
│ • Binary decision boundary: τ = 0.5000                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ P(Tumor) ≥ 0.5000 OR Exploratory Profiling
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Multi-Head Histopathological Anomaly Profiler      │
│ • 1,280-D frozen morphological feature embedding            │
│ • Branching linear probes for distinct tissue phenotypes:   │
│   - (B) Ductal Structure Distortion (BACH proxy, τ=0.4038)   │
│   - (C) Tumor Cell Density Clusters (NuCLS proxy, τ=0.5315) │
│   - (D) Tumor Necrosis & Debris (BCSS proxy, τ=0.2715)      │
│   - (A) Nuclear Pleomorphism [STRICTLY QUARANTINED]         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Visual Attention & Attribution (Grad-CAM)          │
│ • Backpropagates class gradients to features[-1]            │
│ • Localizes cellular clusters driving the model prediction  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Decision Thresholds ($\tau$) & Training Provenance

| Anomaly Head | Source Dataset | Training Task | Selected Threshold ($\tau$) | Optimization Criterion & Provenance |
| :--- | :--- | :--- | :---: | :--- |
| **Primary Malignancy** | **CAMELYON16** | Metastatic Breast Carcinoma to Sentinel Lymph Nodes | $\mathbf{\tau = 0.5000}$ | Standard calibrated sigmoid decision boundary; balanced BCE (`pos_weight = 8.2572`). |
| **Ductal Distortion** | **BACH (ICIAR 2018)** | In Situ & Invasive Ductal Carcinoma vs Normal/Benign | $\mathbf{\tau = 0.4038}$ | Selected via validation F1-score maximization on held-out BACH fold. |
| **Density Clusters** | **NuCLS** | High Tumor Cellularity (≥ 75th percentile density, ≥ 10 clustered nuclei) | $\mathbf{\tau = 0.5315}$ | Selected via validation F1-score maximization on instance-level annotations. |
| **Tumor Necrosis** | **BCSS** | Comedonecrosis & Cellular Debris (≥ 25% necrotic area) | $\mathbf{\tau = 0.2715}$ | Optimized for high clinical sensitivity to necrosis boundaries. |
| **Nuclear Pleomorphism** | **MITOS-ATYPIA-14** | Nuclear Atypia / Pleomorphism (Grade 1, 2, 3) | **QUARANTINED** | **Permanently deactivated** from production checkpoint (`anomaly_linear_heads_clean.pt`) due to unverified provenance and dataset cross-contamination risks. |

---

## 3. Multi-Cohort Dataset Inventory & Partitioning

| Dataset | Clinical Target Task | Clean Training Split | Validation Split | Test Split | Annotation Level | Split Isolation Strategy |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **CAMELYON16** | Metastatic Carcinoma vs Normal Lymph Node | **14,719 patches** (6,353 Normal, 8,366 Tumor) | **3,349 patches** (1,476 Normal, 1,873 Tumor) | **2,881 patches** (2,563 Normal, 318 Tumor) | Lesion Polygon Contours | Strict Patient / WSI-level isolation (42 Train, 10 Val, 8 Test WSIs) |
| **BACH / ICIAR 2018** | Ductal Carcinoma In Situ (DCIS) Distortion | **400 ROI Patches** (100 Benign, 100 InSitu, 100 Invasive, 100 Normal) | Independent Cross-Val Folds | Held-Out Test Subset | Histological Subtype Labels | Image / Subject-level isolation |
| **NuCLS (Amgad et al., 2022)** | Tumor Cell Clustering & Hyper-Density | **1,744 Fields of View** (59,485 annotated nuclei) | Independent Val FOVs | Held-Out Test FOVs | Instance-level Polyline & Mask Boundaries | Slide-Level Partitioning (124 WSIs) |
| **BCSS (Amgad et al., 2019)** | Tumor Necrosis & Cellular Debris | **151 Annotator ROIs** | Validation Slide Subset | Test Slide Subset | Pixel-Level Semantic Segmentation | WSI-Level Partitioning |
| **MITOS-ATYPIA-14 (QUARANTINED)** | Nuclear Atypia & Pleomorphism | **2,772 Patches** (Quarantined) | 924 Patches | 924 Patches | Nottingham Grade Criteria | Zero-Leakage Slide-Level (A07 & A17) |

---

## 4. Implementation Integrity & Verification Notes

1. **Threshold Adherence:**
   - The inference-time decision threshold in production code for primary malignancy is strictly $\tau = 0.5000$ across all 4 architectures (`EfficientNet-B0`, `ResNet-50`, `DenseNet-121`, `VGG-16`).
   - The anomaly profiler thresholds ($\tau_{\text{ductal}} = 0.4038$, $\tau_{\text{density}} = 0.5315$, $\tau_{\text{necrosis}} = 0.2715$) are loaded directly from the clean checkpoint metadata and verified by unit test `test_04_threshold_decisions`.
2. **Quarantined Status of Nuclear Atypia:**
   - The Nuclear Atypia head remains permanently quarantined and excluded from all live diagnostic aggregations, clinical risk summaries, and WSI reporting.
3. **Decoupling of Offline Features:**
   - NuCLS spatial feature CSVs (`nucls_features.csv`, `nucls_spatial_features.csv`) are offline training artifacts and are never queried during patient patch or WSI inference.

---

## 5. Clinical Research Prototype Disclaimer

PathoVision AI is an experimental computer-assisted diagnostic research prototype. Model probabilities, anomaly risk stratifications, and Grad-CAM saliency visualizations are intended solely for academic research and computational pathology verification. They are not cleared for standalone clinical use and do not replace formal review by a board-certified pathologist.
