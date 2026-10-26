# PathoVision AI — Learned Feature Analysis & Latent Space Exploration
**Document Version:** 1.0 (Archival Reference)  
**Applicability:** Research & Technical Evaluation Artifact  
**Preserved Scripts:** `src/objective_2/feature_analysis.py`  
**Preserved Projection Artifacts:** `result/objective_2/feature_analysis/feature_projections.json`  

---

## 1. Executive Summary & Purpose

This document records the empirical methodology, neural embedding architectures, and principal component analysis (PCA) latent space projections evaluated across the four trained convolutional backbones in PathoVision AI:
- **EfficientNet-B0** (Production Backbone)
- **ResNet-50**
- **DenseNet-121**
- **VGG-16**

The feature representations described here capture pre-classifier convolutional activations extracted from the final pooling layer of each architecture prior to linear classification. These high-dimensional vectors encode deep morphological texture, spatial chromatin patterns, and glandular architecture.

---

## 2. Pre-Classifier Architecture Feature Dimensions

| Architecture | Feature Dimension | Backbone Layer Origin | Parameter Count | Checkpoint Size | Inference Speed (FPS) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **EfficientNet-B0** *(Production)* | **1,280-D** | Adaptive AvgPool2d (MBConv + Squeeze-and-Excitation) | 4.01M | 39.7 MB | **21.9 FPS** |
| **ResNet-50** | **2,048-D** | Bottleneck Layer 4 Output (Residual Skips) | 23.51M | 204.2 MB | 10.2 FPS |
| **DenseNet-121** | **1,024-D** | Dense Block 4 Output (Dense Feature Reuse) | 6.95M | 43.7 MB | 7.1 FPS |
| **VGG-16** | **4,096-D** | Linear Layer `classifier[0]` (Post-Flattening) | 134.26M | 1,478.3 MB | 4.1 FPS |

### Architectural Insights:
1. **Compactness & Efficiency:** EfficientNet-B0 produces a 1,280-D feature embedding that achieves the highest PR-AUC (0.9664) and sensitivity (95.28%) while operating at 21.9 FPS—more than 5× faster than VGG-16 (4,096-D, 4.1 FPS).
2. **Feature Density:** DenseNet-121 utilizes the most compact representation (1,024-D) due to dense channel concatenation, yielding extremely conservative false-positive rates (1.87% FP rate).

---

## 3. Dimensionality Reduction Methodology (2D PCA)

High-dimensional feature representations ($D \in \{1024, 1280, 2048, 4096\}$) are mapped onto a 2-dimensional orthogonal plane using Principal Component Analysis (PCA):
$$\mathbf{z} = \mathbf{W}^\top (\mathbf{x} - \boldsymbol{\mu})$$
where $\mathbf{x} \in \mathbb{R}^D$ is the standardized feature vector, $\boldsymbol{\mu}$ is the empirical feature mean, and $\mathbf{W} \in \mathbb{R}^{D \times 2}$ represents the top two eigenvectors of the empirical covariance matrix.

### Scientific Interpretation & Dimensionality Caveat:
- **Exploratory Visual Manifold:** 2D PCA visualizes global variance in neural representations. Clusters of points reflect visual morphological similarity across patches (e.g. dense hyper-cellular clusters vs sparse connective stroma).
- **Not a Diagnostic Metric:** Separation in 2D PCA space illustrates learned representation structure, not direct clinical performance. High overlap in 2D does not imply poor high-dimensional separability.

---

## 4. Morphology Reality Audit

To ensure complete scientific transparency, PathoVision AI documents the strict operational boundaries of its feature representations:

### What PathoVision AI Actually Computes:
- **Spatial Convolutions:** Multi-scale feature extraction across convolutional receptive fields.
- **Global Feature Vectors:** Fixed-length bottleneck embeddings ($1,024$ to $4,096$ floating-point scalars).
- **Sigmoid Classifier Activations:** $P(\text{Tumor})$ model score calculated via calibrated linear heads branching from frozen embeddings.
- **Grad-CAM Attribution:** Gradient-weighted spatial activation maps identifying regions influencing model logits.

### What PathoVision AI Does NOT Do:
- **No Instance Nuclear Segmentation:** PathoVision AI does not segment individual cell nuclei or draw per-cell boundaries during inference.
- **No Physical Caliper Measurements:** The system does not calculate physical cell dimensions in micrometers ($\mu\text{m}$), nuclear area, perimeter, or circularity.
- **No Direct Mitotic Figures Counting:** Mitotic counts require specialized high-power oil-immersion microscopy ($40\times/100\times$) with verified mitotic spindle annotations.
- **No Automated Planimetric Staging:** PathoVision AI does not perform TNM cancer staging or continuous tumor volume calculations.

---

## 5. Artifact Locations & Reproducibility

All code and data artifacts for feature extraction and PCA projection are preserved:
- **Feature Extraction Runner:** `src/objective_2/feature_analysis.py`
  - `extract_patch_features(model, tensor)`
  - `compute_pca_projections(feature_matrix, n_components=2)`
  - `save_feature_embeddings(embeddings_dict, output_path)`
- **Stored Projections:** `result/objective_2/feature_analysis/feature_projections.json`
- **Model Checkpoints:** `checkpoints/`
  - `checkpoints/best_model.pt` (EfficientNet-B0)
  - `checkpoints/resnet50_best.pt`
  - `checkpoints/densenet121_best.pt`
  - `checkpoints/vgg16_best.pt`
