# Actionable Guide: Recommended Changes for Final Project Report & Presentation (PPT)

**Project Title:** Patch-Based Unimodal Cancer Anomaly Detection in Tissue Images Using Deep Learning Models  
**Institution:** Bapuji Institute of Engineering and Technology (BIET), Davanagere — VTU (2025–2026)  
**Project Associates:** Prajwal M H (4BD23CS139), Raksha P (4BD23CS159), Samruddhi S (4BD23CS181), Shafia Kouser S (4BD23CS193)  
**Project Guide:** Prof. Rachana G Sunkad M.Tech., Assistant Professor, Dept. of CS&E  

---

## Executive Overview

In the Phase-I submission, your report and PPT represented a **preliminary project proposal and literature review**. The Phase-I report explicitly noted:
> *"the actual implementation, training, and performance evaluation of the system have not yet been carried out and remain as part of the next phase of the work."* (Phase-I Report, Page 38)

Because the full system is now **100% implemented, trained, benchmarked across 4 deep learning models, and deployed as an interactive clinical web workstation**, your final submission must be transitioned from prospective phrasing into a **completed engineering and scientific thesis**.

This document outlines the **exact section-by-section changes needed for your final report** and the **slide-by-slide improvements for your final project presentation (PPT)**.

---

## PART 1: Recommended Changes for the Final Project Report

### 1. Title Page, Certificate & Acknowledgements
- **Academic Year:** Maintain `2025–2026`.
- **Phase Designation:** Change `MAJOR PROJECT PHASE-I REPORT` to `MAJOR PROJECT FINAL REPORT` (or `PHASE-II REPORT`).
- **Course Code:** If VTU requires the 8th semester project code (e.g., `21CSP83` or `BCS885`), update from the progressive phase code (`BCS685`).

---

### 2. Abstract
- **Current State (Phase-I):** Describes the proposed conceptual framework in future/present tense without final empirical metrics.
- **Recommended Revision:** Update the abstract to report the **actual achieved experimental metrics**:
  > *"This project presents a completed, patch-based unimodal cancer anomaly detection system for breast histopathology. Using 60 Whole Slide Images (WSIs) from the CAMELYON16 cohort and multi-cohort anomaly datasets (NuCLS, BCSS, BACH), a deterministic preprocessing pipeline extracted 14,719 quality-controlled patches. Four deep convolutional architectures—EfficientNet-B0, ResNet-50, DenseNet-121, and VGG-16—were trained and evaluated on 2,881 held-out test patches. DenseNet-121 achieved the highest overall accuracy (97.22%) and ROC-AUC (0.9905), while EfficientNet-B0 demonstrated superior screening sensitivity (95.28% recall) with rapid inference (21.9 FPS) across a lightweight 4.01M parameter footprint. Four targeted histopathological abnormalities—nuclear pleomorphism, ductal structure distortion, tumor cell density clusters, and necrosis—were modeled via linear probes and visual attention. Multi-architecture Grad-CAM heatmaps were integrated to provide visual interpretability. The system was deployed into an interactive Streamlit clinical decision-support workstation ('PathoVision AI') supporting gigapixel slide streaming, tile drill-down, and QuPath medical integration."*

---

### 3. Literature Survey (Chapter 2)
- **Paper Count Harmonization:** The Phase-I report contains 25 literature papers, while your PPT table reviewed **30 papers** (adding landmark papers by Tan & Le 2020 on EfficientNet, Ilse et al. 2018 on Attention MIL, Liu et al. 2017 on Gigapixel WSI, Bejnordi et al. 2017 on CAMELYON16, and Selvaraju et al. 2017 on Grad-CAM).
- **Action:** Append papers [26] through [30] from your PPT into Chapter 2 and the References section of your Final Report so both documents are 100% aligned.

---

### 4. Methodology (Chapter 4)
Update Chapter 4 to document the exact operational parameters of the built system:
- **Tiling Grid:** OpenSlide Level 0 ($224 \times 224$ pixels, adaptive $112\text{--}224$ px stride).
- **Tissue QC Gates:** Laplacian focus variance $\ge 20.0$, Otsu HSV saturation $> 20$, tissue area ratio $\ge 0.20$, white background ratio $\le 0.80$.
- **Class Imbalance Loss:** Binary Cross-Entropy with positive class weighting ($\text{pos\_weight} = 8.2572$).
- **Calibrated Anomaly Decision Thresholds:**
  - Ductal Structure Distortion (BACH InSitu proxy): $\tau_{\text{ductal}} = 0.4038$
  - Tumor Cell Density Clusters (NuCLS proxy): $\tau_{\text{density}} = 0.5315$
  - Tumor Necrosis & Cellular Debris (BCSS proxy): $\tau_{\text{necrosis}} = 0.2715$
  - Nuclear Atypia (Deep activation threshold): $\tau_{\text{atypia}} = 0.0500$
- **Risk Score Tiering:** Low Risk ($[0.00, 0.30)$), Medium Risk ($[0.30, 0.70)$), High Risk ($[0.70, 1.00]$).

---

### 5. NEW CHAPTER: Implementation & Experimental Results (Chapter 5)
Insert this chapter to document your completed results:
1. **Master Comparative Benchmark Table:** Insert the complete 4-model evaluation table on held-out test data ($N=2,881$):

| Architecture | Parameters | Model Size | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC | Speed (FPS) |
|---|---|---|---|---|---|---|---|---|---|---|
| **DenseNet-121** | 6.95 M | 43.68 MB | **97.22%** | 89.94% | **98.13%** | **85.63%** | **87.73%** | **0.9905** | 0.9547 | 7.1 FPS |
| **ResNet-50** | 23.51 M | 204.20 MB | 96.60% | 94.03% | 96.92% | 79.10% | 85.92% | 0.9882 | 0.9595 | 10.2 FPS |
| **EfficientNet-B0** | **4.01 M** | **39.72 MB** | 96.42% | **95.28%** | 96.57% | 77.49% | 85.47% | 0.9875 | **0.9664** | **21.9 FPS** |
| **VGG-16** | 134.26 M | 1478.31 MB | 96.32% | 94.34% | 96.57% | 77.32% | 84.99% | 0.9830 | 0.9609 | 4.1 FPS |

2. **Confusion Matrices:** Include the test confusion matrix for EfficientNet-B0 ($TP=303$, $FP=88$, $TN=2,475$, $FN=15$) and the comparative matrix plots stored in `results/objective_3_evaluation_and_xai/confusion_matrices/`.
3. **Comparative ROC and PR Curves:** Insert `results/objective_3_evaluation_and_xai/roc_pr_curves/master_comparative_roc_pr_curves.png`.
4. **Grad-CAM Visual Explainability:** Include visual figures showing original H&E patches alongside Grad-CAM heatmaps for normal tissue, invasive ductal carcinoma, dense cellular nests, and necrosis debris (from `results/objective_3_evaluation_and_xai/gradcam/`).
5. **WSI Heatmap Reconstruction:** Include figure panels showing whole slide optical scans alongside continuous malignancy probability overlays for `tumor_058` (metastasis focus) and `normal_004` (benign margin) from `results/objective_4_web_application/wsi_analysis/`.

---

### 6. NEW CHAPTER: Web Application Deployment & System UI (Chapter 6)
Document the developed decision-support platform:
- Architecture of the Streamlit application (`app.py`).
- Secure role-based access control (PBKDF2-HMAC-SHA256 hashing, Researcher and Reviewer privileges).
- Walkthrough of Section 1: Single Patch Analysis with live Grad-CAM and anomaly phenotyping.
- Walkthrough of Section 2: Whole-Slide Image Analysis with continuous heatmap generation, hotspot tile inspection, and Patient Slide Diagnostic Summary.
- Walkthrough of Section 3: Model Evaluation dashboard.
- Walkthrough of Section 4: Learned Feature Analysis and 2D/3D PCA latent space exploration.
- QuPath medical integration (GeoJSON polygon export and TSV measurement tables).

---

### 7. Conclusion & Future Scope (Chapter 7)
- **Replace Phase-I concluding text:** Remove statements indicating training and evaluation have not yet occurred.
- **State Final Conclusions:**
  1. Achieved $96.42\%$ accuracy and $95.28\%$ clinical screening sensitivity using EfficientNet-B0.
  2. Demonstrated that DenseNet-121 achieves superior specificity ($98.13\%$) for second-opinion confirmation.
  3. Solved the clinical black-box problem by generating real-time Grad-CAM heatmaps and isolating four distinct histopathological anomaly types.
  4. Deployed a full clinical workstation capable of processing gigapixel slides and exporting annotations to digital pathology software (QuPath).
- **Future Scope:** Expanding to multi-modal integration (combining digital histopathology with patient genomic markers), multi-institutional clinical trial validation, and exploring Vision Transformers (ViTs) for ultra-wide field of view.

---

## PART 2: Slide-by-Slide Guide for Final Presentation (PPT)

Below is the recommended structure for your **Final Project Presentation (15–18 slides)**:

```mermaid
flowchart LR
    S1["Slide 1: Title & Team"] --> S2["Slide 2: Objectives & Agenda"]
    S2 --> S3["Slide 3: Clinical Problem & Targeted Anomalies"]
    S3 --> S4["Slide 4: 6-Step Methodology Pipeline"]
    S4 --> S5["Slide 5: Obj 1 - WSI Tiling & Preprocessing"]
    S5 --> S6["Slide 6: Obj 2 - Deep Learning & Anomaly Heads"]
    S6 --> S7["Slide 7: Obj 3 - 4-Model Comparative Evaluation"]
    S7 --> S8["Slide 8: Obj 3 - Grad-CAM Visual Explainability"]
    S8 --> S9["Slide 9: Obj 4 - WSI Spatial Heatmaps & Summary"]
    S9 --> S10["Slide 10: Obj 4 - Live Web Application Walkthrough"]
    S10 --> S11["Slide 11: Clinical Impact & QuPath Integration"]
    S11 --> S12["Slide 12: Conclusion & Viva Questions"]
```

### Slide 1: Title Slide
- **Title:** "Patch-Based Unimodal Cancer Anomaly Detection in Tissue Images Using Deep Learning Models"
- **Project Associates:** Prajwal M H (4BD23CS139), Raksha P (4BD23CS159), Samruddhi S (4BD23CS181), Shafia Kouser S (4BD23CS193)
- **Guide:** Prof. Rachana G Sunkad M.Tech., Assistant Professor, Dept. of CS&E
- **Department & Institute:** Department of Computer Science & Engineering, BIET Davanagere — VTU (2025–2026).

---

### Slide 2: Table of Contents & Project Deliverables
- Introduction & Problem Statement
- 4 Targeted Histopathological Abnormalities
- System Methodology & 6-Step Diagnostic Flow
- Objective 1: WSI Patch Extraction & Preprocessing Pipeline
- Objective 2: Deep Learning Model Development & Anomaly Profiling
- Objective 3: Four-Architecture Comparative Performance Evaluation
- Objective 3: Explainable AI via Grad-CAM Visual Attention
- Objective 4: Whole-Slide Heatmap Reconstruction & Clinical Web Application
- Conclusion, Clinical Viability & References

---

### Slide 3: Clinical Motivation & 4 Targeted Abnormalities
- Keep your excellent 4-panel image from Phase-I Slide 3 showing:
  1. **Nuclear Pleomorphism / Atypia:** Variability in size, shape, hyperchromasia.
  2. **Ductal Structure Distortion:** Glandular architectural infiltration (DCIS/IDC).
  3. **Tumor Cell Density Clusters:** Severe spatial overcrowding ($\ge 10$ nuclei).
  4. **Necrosis / Cellular Debris:** Granular eosinophilic dead tissue zones.
- Highlight the clinical gap: Traditional CAD systems act as black boxes and fail to explain *which* specific tissue deformation triggered the malignancy alarm.

---

### Slide 4: Problem Statement & Objectives
- **Problem Statement:** Gigapixel Whole Slide Images (WSIs) exceed GPU memory constraints, while naive patch extraction causes context loss and deep models suffer from lack of clinical interpretability.
- **The 4 Core Objectives:**
  1. *Objective 1:* WSI tiling, automated tissue segmentation, and quality preprocessing.
  2. *Objective 2:* EfficientNet transfer learning, anomaly scoring, and risk tiering.
  3. *Objective 3:* Comparative benchmarking across ResNet, DenseNet, EfficientNet, and VGG with Grad-CAM explainability.
  4. *Objective 4:* Deployment of an interactive clinical decision-support web application.

---

### Slide 5: Methodology Diagram
- Present your 6-step diagram:
  $$\text{1. Data Collection} \to \text{2. Preprocessing} \to \text{3. Model Training} \to \text{4. Anomaly \& Risk} \to \text{5. Model Comparison} \to \text{6. Output Deployment}$$
- Highlight dataset rigor: 60 CAMELYON16 WSIs, 14,719 QC-cleared training patches, plus NuCLS, BCSS, BACH, and BreakHis test cohorts.

---

### Slide 6: Objective 1 — Preprocessing & Quality Verification
- **Visuals:** Use the 6-panel composite from `results/objective_1_preprocessing/level_wise_pipeline/`:
  - 1. Raw WSI $\to$ 2. Extracted Patch $\to$ 3. Otsu Tissue Mask $\to$ 4. Gaussian Blur $\to$ 5. Data Augmentation $\to$ 6. Normalization Statistics ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- **Quality Gates:** Show that $100\%$ of acellular glass and out-of-focus blur was filtered using automated Laplacian variance ($\ge 20.0$) and HSV saturation gating.

---

### Slide 7: Objective 2 — Model Development & Risk Stratification
- **Backbone Architecture:** EfficientNet-B0 with ImageNet transfer learning fine-tuned with class-weighted loss ($\text{pos\_weight}=8.2572$).
- **Anomaly Scoring & Risk Tiering:**
  - Low Risk ($<0.30$), Medium Risk ($0.30\text{--}0.70$), High Risk ($\ge 0.70$).
- **Visual:** Replace the old preliminary ROC curve ($0.6878$) with the **calibrated anomaly risk score distribution** from `results/objective_2_model_and_anomaly/risk_analysis/anomaly_risk_distribution.png` showing distinct separation between normal tissue and malignant carcinoma.

---

### Slide 8: Objective 3 — Four-Architecture Benchmark Evaluation
- **Visuals:** 
  1. Master Comparative Benchmark Table (highlighting DenseNet's $97.22\%$ accuracy and EfficientNet's $95.28\%$ recall).
  2. Comparative ROC & Precision-Recall curves from `results/objective_3_evaluation_and_xai/roc_pr_curves/master_comparative_roc_pr_curves.png`.
- **Key Talking Points:**
  - EfficientNet-B0 is the clinical screening champion: catches $95.28\%$ of metastases at $21.9$ FPS with only 4M parameters.
  - DenseNet-121 is the confirmation specialist: eliminates false positives with $98.13\%$ specificity and $0.9905$ ROC-AUC.

---

### Slide 9: Objective 3 — Explainable AI via Multi-Architecture Grad-CAM
- **Visuals:** Include 2–3 sample patches with their Grad-CAM heatmaps from `results/objective_3_evaluation_and_xai/gradcam/`.
- Show how the attention maps highlight:
  - Invasive ductal carcinoma nests (warm red activation).
  - Hyper-dense tumor clusters.
  - Inactive cool blue response over normal collagen stroma and adipose tissue.
- **Color Scale Bar:** Explain the intuitive clinical color gradient (`[Blue: Normal Stroma] ➔ [Red: High Suspicious Anomaly Focus]`).

---

### Slide 10: Objective 4 — Whole-Slide Image (WSI) Spatial Analysis
- **Visuals:** Side-by-side display of the Level 0 Whole Slide scan against the continuous spatial malignancy heatmap for `tumor_058` (from `results/objective_4_web_application/wsi_analysis/tumor_058_spatial_diagnostic_dashboard.png`).
- **Terminal Summary Box:** Feature the **Patient Slide Diagnostic Summary** (matching your Phase-I terminal screenshot!):
  ```
  PATIENT SLIDE DIAGNOSTIC SUMMARY
  Slide ID: tumor_058 | Total Patches: 44 viable tiles
  Average Slide Anomaly: 94.92% (Peak: 99.98%)
  • Low Risk (<0.30): 2 patches (4.5%)
  • Medium Risk (0.30-0.70): 3 patches (6.8%)
  • High Risk (>=0.70): 39 patches (88.6%)
  FINAL CLINICAL VERDICT: MALIGNANT METASTASIS (TUMOR BURDEN: 31.8%)
  ```

---

### Slide 11: Objective 4 — PathoVision AI Web Application
- **Visuals:** Screenshots of the Streamlit application (`app.py`):
  1. Sidebar with 1-click Demo Sign-In (`👤 Lead Researcher` and `🛡️ Academic Reviewer`).
  2. Patch Analysis module showing the primary verdict card, anomaly score meter, risk tier, and Grad-CAM colormap.
  3. Whole-Slide module with 6-step flow banner and hotspot tile drill-down.
  4. QuPath GeoJSON / TSV export interface.
- Highlight that the application is fully interactive, responsive, and ready for live demonstration.

---

### Slide 12: Architectural Comparison & Hardware Feasibility
- Emphasize computational feasibility:
  - EfficientNet-B0 requires only **39.7 MB** of storage and **4.01M parameters**.
  - Compared to VGG-16 (1.47 GB, 134M parameters), EfficientNet delivers **5.3x faster inference** (21.9 FPS vs 4.1 FPS) with virtually identical accuracy.
  - Demonstrates that hospitals do not require multimillion-dollar GPU clusters to run clinical AI triage.

---

### Slide 13: Conclusion & Key Achievements
- **Summary of Achievements:**
  - ✅ Objective 1: WSI extraction, focus QC, and artifact rejection completed.
  - ✅ Objective 2: EfficientNet-B0 fine-tuned; 4 targeted anomalies profiled; clinical risk tiers established.
  - ✅ Objective 3: Evaluated 4 CNN backbones on 2,881 test patches; integrated Grad-CAM heatmaps for visual transparency.
  - ✅ Objective 4: Production Streamlit workstation deployed with live WSI spatial heatmaps and QuPath export.
- **Viva Readiness:** Entire project modularized in `results/`, backed by reproducible test scripts and verified statistical outputs.

---

### Slide 14: References
- Include references [1] through [30] from your literature review.

---

## PART 3: Summary of Artifact Locations for Viva & Presentation

During your demonstration or oral examination, you can access all supporting evidence directly from these paths:

| Evidence Requested by Examiner | File Location | Key Verification Point |
|---|---|---|
| **WSI Preprocessing Steps** | `results/objective_1_preprocessing/level_wise_pipeline/` | Step-by-step images: WSI $\to$ Patch $\to$ Mask $\to$ Blur $\to$ Augmentation $\to$ Norm |
| **Dataset Manifests & QC** | `results/objective_1_preprocessing/quality_control/` | `training_patch_quality_control.xlsx` (7-sheet executive workbook) |
| **Model Weights Checkpoints** | `result/objective_2/checkpoints/` | `best_model.pt` (EfficientNet), `anomaly_linear_heads_clean.pt` |
| **4-Model Benchmark Table** | `results/objective_3_evaluation_and_xai/metrics/` | `master_test_benchmark_comparison.csv` and `.json` |
| **Confusion Matrices** | `results/objective_3_evaluation_and_xai/confusion_matrices/` | Individual and comparative confusion matrices for all 4 models |
| **ROC & PR Curves** | `results/objective_3_evaluation_and_xai/roc_pr_curves/` | `master_comparative_roc_pr_curves.png` |
| **Grad-CAM Attention Maps** | `results/objective_3_evaluation_and_xai/gradcam/` | Multi-layer visual attribution case studies |
| **WSI Heatmaps & Dashboards** | `results/objective_4_web_application/wsi_analysis/` | `tumor_058` and `normal_004` spatial heatmaps and JSON summaries |
| **Live Web Application** | Run `streamlit run app.py` | Full interactive workstation on `localhost:8501` |
