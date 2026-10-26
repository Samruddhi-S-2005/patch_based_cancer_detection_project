# Objective 2: Explicit Morphology Analysis vs Learned Neural Embeddings

**Audit & Verification Date:** 2026-10-20  
**Project:** PathoVision AI (Objective 2: Deep Learning Models & Anomaly Analysis)  
**Document Status:** Complete & Verified Technical Specification  

---

## 1. Core Principle: Latent Neural Features vs Biological Measurements

Convolutional Neural Networks (CNNs) encode visual input patches into continuous high-dimensional latent vectors $\mathbf{z} \in \mathbb{R}^D$:
- **EfficientNet-B0:** $\mathbf{z} \in \mathbb{R}^{1280}$
- **ResNet-50:** $\mathbf{z} \in \mathbb{R}^{2048}$
- **DenseNet-121:** $\mathbf{z} \in \mathbb{R}^{1024}$
- **VGG-16:** $\mathbf{z} \in \mathbb{R}^{4096}$

These latent representations respond to visual textures, chromatin distributions, cellular boundaries, and stain gradients. However, individual dimensions or linear combinations of these vectors **are not explicit, continuous physical biological measurements**.

---

## 2. Model-by-Model Feature Layer & Tensor Dimensions

Every model architecture in PathoVision AI extracts a distinct pre-classifier embedding tensor through frozen backbone weights:

### 2.1 EfficientNet-B0 (Production Backbone)
- **Checkpoint Path:** `result/objective_2/checkpoints/best_model.pt`
- **Checkpoint SHA-256:** `cc6764e073a21e523b2ae6a6b24b6c718bf4159a778f2563a307c2e88900b3ef`
- **Input Tensor:** $(B, 3, 224, 224)$ float32, normalized via ImageNet parameters ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- **Feature Extraction Layer:** `model.backbone.features` (the final stage 8 inverted residual block).
- **Pre-Pooling Tensor Dimension:** $(B, 1280, 7, 7)$.
- **Pooling & Flattening Operation:** Spatial adaptive average pooling `AdaptiveAvgPool2d(output_size=(1, 1))` followed by `torch.flatten(x, 1)`.
- **Resulting Feature Embedding:** **1280-D vector** $\mathbf{z} \in \mathbb{R}^{1280}$.
- **Classifier Usage:** Feed-forward linear layer `Linear(in_features=1280, out_features=2)` yielding tumor/normal logits, followed by softmax for malignancy probability $p_{\text{tumor}} = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$.

### 2.2 ResNet-50
- **Checkpoint Path:** `result/objective_2/checkpoints/resnet/best_model.pt`
- **Checkpoint SHA-256:** `a1e47d42e1a5b732eeadb5cea620807add979ebea938f9f7b73072b6fd4f8864`
- **Input Tensor:** $(B, 3, 224, 224)$ float32, ImageNet normalized.
- **Feature Extraction Layer:** `model.backbone.layer4` (the final residual bottleneck block with expansion 4).
- **Pre-Pooling Tensor Dimension:** $(B, 2048, 7, 7)$.
- **Pooling & Flattening Operation:** `AdaptiveAvgPool2d(output_size=(1, 1))` followed by `torch.flatten(x, 1)`.
- **Resulting Feature Embedding:** **2048-D vector** $\mathbf{z} \in \mathbb{R}^{2048}$.
- **Classifier Usage:** `Linear(in_features=2048, out_features=2)` mapping the 2048-D embedding to binary logits.

### 2.3 DenseNet-121
- **Checkpoint Path:** `result/objective_2/checkpoints/densenet/best_model.pt`
- **Checkpoint SHA-256:** `13e80add08df8dd471b95697b50e30c4745c08fc511ed2da8b1782fb53094303`
- **Input Tensor:** $(B, 3, 224, 224)$ float32, ImageNet normalized.
- **Feature Extraction Layer:** `model.backbone.features` followed by non-linear `F.relu`.
- **Pre-Pooling Tensor Dimension:** $(B, 1024, 7, 7)$.
- **Pooling & Flattening Operation:** `F.adaptive_avg_pool2d(x, (1, 1))` followed by `torch.flatten(x, 1)`.
- **Resulting Feature Embedding:** **1024-D vector** $\mathbf{z} \in \mathbb{R}^{1024}$.
- **Classifier Usage:** `Linear(in_features=1024, out_features=2)` projecting concatenated dense feature maps to binary logits.

### 2.4 VGG-16
- **Checkpoint Path:** `result/objective_2/checkpoints/vgg/best_model.pt`
- **Checkpoint SHA-256:** `353f1fcb8d13e412a6dc690242edbf2126d537ec99152223b4b8f1fb1fba24dd`
- **Input Tensor:** $(B, 3, 224, 224)$ float32, ImageNet normalized.
- **Feature Extraction Layer:** `model.backbone.features` -> `avgpool` -> `model.backbone.classifier[:6]`.
- **Pre-Pooling Tensor Dimension:** $(B, 512, 7, 7)$ before pooling.
- **Pooling & Flattening Operation:** `AdaptiveAvgPool2d((7, 7))` followed by `flatten(1)` to $(B, 25088)$, then passed through the dense fully-connected block: `Linear(25088, 4096)` $\to$ `ReLU` $\to$ `Dropout` $\to$ `Linear(4096, 4096)` $\to$ `ReLU` $\to$ `Dropout`.
- **Resulting Feature Embedding:** **4096-D vector** $\mathbf{z} \in \mathbb{R}^{4096}$ (penultimate FC layer activations).
- **Classifier Usage:** Final classification layer `Linear(in_features=4096, out_features=2)` producing binary class scores.

---

## 3. Explicit Audit of Biological Morphology vs Learned Representations

PathoVision AI distinguishes between **dynamically measured physical quantities** and **learned latent proxy features**:

| Biological / Morphological Attribute | Dynamically Measured in Code? | Status | Implementation Truth & Scientific Justification |
| :--- | :---: | :---: | :--- |
| **Nuclear Count / Cellular Density** | **NO** | `NOT EXPLICITLY MEASURED` | The pipeline does **NOT** run an instance segmentation network (e.g. StarDist, Cellpose) or nuclear centroid counter during live inference. "Tumor Cell Density Clusters" is a linear probe evaluating cosine similarity to high-density NuCLS training patterns ($\ge 10$ nuclei). |
| **Nuclear Area / Diameter ($\mu\text{m}$)** | **NO** | `NOT EXPLICITLY MEASURED` | No physical continuous micrometer calipers or morphological fitting ellipses are computed on individual nuclei. |
| **Nuclear Eccentricity / Circularity** | **NO** | `NOT EXPLICITLY MEASURED` | Individual nuclear boundaries are not delineated; aspect ratios, circularity, and perimeter roughness are not measured. |
| **Nuclear-to-Cytoplasmic (N:C) Ratio** | **NO** | `NOT EXPLICITLY MEASURED` | Cell membranes are not segmented; continuous cytoplasmic surface areas are not computed. |
| **Ductal Boundary / Geometry** | **NO** | `NOT EXPLICITLY MEASURED` | Ductal distortion is evaluated via a linear probe on frozen 1280-D features trained on BACH in-situ patterns. No glandular polygon perimeter or lumen convexity is measured geometrically. |
| **Necrotic Area Planimetry ($\mu\text{m}^2$)** | **NO** | `NOT EXPLICITLY MEASURED` | Tumor Necrosis is evaluated via a linear probe trained on BCSS necrotic debris patterns. Physical square micrometers of debris are not calculated. |
| **Tissue / Stroma Area Fraction** | **YES** | `VERIFIED DIRECT MEASUREMENT` | Measured directly in Layer 1 Quality Control via HSV saturation ($> 20$) and luminance ($30 \le V \le 235$) thresholding. Quality gate: $\text{Tissue Ratio} \ge 0.20$. |
| **Focus / Blurriness Score** | **YES** | `VERIFIED DIRECT MEASUREMENT` | Measured directly in Layer 1 Quality Control via the variance of the 2D discrete Laplacian operator $\text{Var}(\nabla^2 I)$. Quality gate: Blur score $\ge 20.0$ (and $\ge 100.0$ in strict live screening). |

---

## 4. Grad-CAM Interpretation: Attribution vs Segmentation

> [!IMPORTANT]
> Grad-CAM visual heatmaps generated by PathoVision AI are **gradient-weighted class activation attributions**, calculated by backpropagating gradients from the target class logit to the final convolutional feature maps.
>
> They identify spatial image regions that positively influenced the network's classification score. They **must not be interpreted as semantic pixel-level segmentation masks** or physical boundary delineations of the tumor.

---

## 5. Multi-Head Anomaly Probing Architecture

The multi-head anomaly profiler branches strictly from the frozen **1280-D EfficientNet-B0** feature embedding:
$$\hat{y}_k = \sigma(\mathbf{w}_k^T \mathbf{z} + b_k)$$

1. **Ductal Structure Distortion (BACH InSitu Proxy):** Threshold $\tau = 0.4038$. Evaluates glandular architectural disruption.
2. **Tumor Cell Density Clusters (NuCLS Proxy):** Threshold $\tau = 0.5315$. Evaluates hyper-cellular neoplastic clustering.
3. **Tumor Necrosis & Cellular Debris (BCSS Proxy):** Threshold $\tau = 0.2715$. Evaluates eosinophilic necrotic breakdown.
4. **Nuclear Pleomorphism / Atypia:** **Permanently quarantined** in `anomaly_linear_heads_clean.pt`. Excluded from active inference.

---

## 6. Historical Offline CSV Artifacts Notice

The files `nucls_features.csv` and `nucls_spatial_features.csv` in the repository record historical nearest-neighbor and bounding box statistics extracted offline from external NuCLS training annotations. They are **NOT** queried, executed, or referenced during live inference on patient tissue patches.
