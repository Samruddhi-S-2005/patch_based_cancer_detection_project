# Objective 3: Comparative Evaluation & Explainable AI (Grad-CAM)

## Overview
Objective 3 conducts a comprehensive comparative benchmark across four state-of-the-art CNN architectures (EfficientNet-B0, ResNet-50, DenseNet-121, VGG-16) on held-out test patches ($N=2,881$) and integrates Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explainability.

## Comparative Performance Matrix (CAMELYON16 Test Set)
| Architecture | Parameters | Checkpoint | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC | PR-AUC | Speed (FPS) |
|---|---|---|---|---|---|---|---|---|---|
| **DenseNet-121** | 6.95 M | 43.68 MB | **97.22%** | 89.94% | **98.13%** | **87.73%** | **0.9905** | 0.9547 | 7.1 FPS |
| **ResNet-50** | 23.51 M | 204.20 MB | 96.60% | 94.03% | 96.92% | 85.92% | 0.9882 | 0.9595 | 10.2 FPS |
| **EfficientNet-B0** | **4.01 M** | **39.72 MB** | 96.42% | **95.28%** | 96.57% | 85.47% | 0.9875 | **0.9664** | **21.9 FPS** |
| **VGG-16** | 134.26 M | 1478.31 MB | 96.32% | 94.34% | 96.57% | 84.99% | 0.9830 | 0.9609 | 4.1 FPS |

*Key Findings:*
- **Highest Sensitivity:** EfficientNet-B0 ($95.28\%$), minimizing false negatives in clinical cancer screening.
- **Highest Accuracy & ROC-AUC:** DenseNet-121 ($97.22\%$, $0.9905$ AUC).
- **Fastest Inference:** EfficientNet-B0 ($21.9$ frames/sec), optimal for gigapixel whole-slide scanning.

## Directory Artifacts
- `metrics/`:
  - `master_test_benchmark_comparison.csv`: Complete comparative table with all metrics.
  - `master_test_benchmark_comparison.json`: Machine-readable benchmark record.
- `confusion_matrices/`:
  - `test_confusion_matrix.png`: EfficientNet-B0 held-out test confusion matrix ($TP=303, FP=88, TN=2475, FN=15$).
  - `resnet_confusion_matrix.png`, `densenet_confusion_matrix.png`, `vgg_confusion_matrix.png`.
- `roc_pr_curves/`:
  - `master_comparative_roc_pr_curves.png`: Comparative ROC and PR curves for all 4 models.
  - `test_roc_curve.png`, `test_pr_curve.png`.
- `gradcam/`: Layered diagnostic cases illustrating Grad-CAM visual attribution across anomaly types:
  - `01_normal_case.png`
  - `02_malignant_case.png`
  - `03_dense_malignant_case.png`
  - `04_ductal_distortion_case.png`
  - `05_tumor_necrosis_case.png`
  - `06_hyperdensity_atypia_case.png`
- `scientific_evidences/`: Visual evaluation proofs (`07_confusion_matrix.png`, `08_ROC_curve.png`, `09_PR_curve.png`).
- `predictions/`: `test_predictions.csv` with per-patch probabilities and labels.
