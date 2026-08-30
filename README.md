# Patch-Based Breast Cancer Detection from Whole Slide Images

A patch-based deep learning framework for breast cancer detection from
histopathological Whole Slide Images (WSIs).

The project processes large WSIs by extracting 224×224 image patches,
performing tissue and quality filtering, assigning tumor/normal labels
using WSI annotations, and preparing a clean dataset for deep learning
classification.

The primary deep learning model for the project is EfficientNet-B0.

---

## Project Status

### Current Phase

The project has completed the major data preparation and patch
extraction stage.

The repository currently contains the WSI registry, patch extraction
pipeline, annotation handling, quality-control utilities, visualization
utilities, and final patch dataset manifest.

The next major stage is deep learning model training using EfficientNet-B0.

---

## Current Completed Work

### 1. WSI Dataset Organization

The project uses CAMELYON16-style Whole Slide Images.

WSIs are registered in:

    data/wsi_registry.csv

The registry records:

- WSI ID
- filename
- dataset
- class
- annotation availability
- train/validation/test split
- WSI dimensions
- resolution/magnification metadata
- vendor information
- processing status

### 2. WSI-Level Dataset Separation

Train, validation, and test sets are separated at the WSI level.

This is important to prevent data leakage between training and evaluation.

Patches from the same WSI must not be distributed across different
dataset splits.

Do not change the existing split assignments without project approval.

---

## Patch Extraction

The current patch extraction methodology uses:

- Patch size: 224 × 224 pixels
- WSI level: Level 0
- Candidate stride: 112 pixels
- Candidate overlap: 50%
- Sliding-window candidate generation
- Tissue/background filtering
- Quality filtering
- Tumor annotation-based labeling
- Spatially diverse patch selection
- Class-aware selection

The extractor is designed to process one WSI at a time.

---

## Patch Selection Method

The extraction pipeline does not simply extract 500 patches and then
discard the rest.

The process is:

1. Generate a large candidate pool using sliding windows.
2. Perform low-resolution tissue/background screening.
3. Evaluate candidate quality.
4. Use tumor annotations where available.
5. Categorize tumor WSI candidates as:
   - tumor-dominant
   - normal-tissue
   - boundary-mixed
6. Exclude ambiguous boundary-mixed patches from the clean dataset.
7. Apply quality and spatial diversity selection.
8. Select approximately 500 representative patches per WSI where
   sufficient candidates are available.

For tumor WSIs, the preferred target is approximately:

- 250 tumor patches
- 250 normal patches

If insufficient candidates are available in one category, the selection
procedure can fill remaining slots from the other clean category.

The exact number selected can therefore be lower than the target for some
WSIs.

---

## Tumor Annotation Handling

For annotated tumor WSIs:

- 0% tumor overlap → normal tissue
- 0–50% tumor overlap → boundary-mixed / ambiguous
- ≥50% tumor overlap → tumor-dominant

Boundary-mixed patches are excluded from the clean training dataset.

This is intended to reduce label ambiguity around tumor boundaries.

---

## Quality Control

Candidate patches are evaluated using tissue and image-quality
information.

The pipeline includes:

- tissue fraction estimation
- background/white-region filtering
- grayscale variance
- gradient/texture information
- quality scoring
- spatial diversity
- duplicate/redundancy handling

The final processing information is recorded in:

    result/final_dataset_manifest.csv

---

## Saved Patch Dataset

The extracted patches are intentionally NOT stored in the GitHub
repository because the dataset contains thousands of image files.

The patch images should be obtained from the project's shared dataset
storage.

Expected local structure:

    data/
    └── PATCHES/
        ├── CLEANED/
        │   ├── normal/
        │   └── tumor/
        │
        ├── TRAIN/
        │   ├── normal/
        │   └── tumor/
        │
        ├── VALIDATION/
        │   ├── normal/
        │   └── tumor/
        │
        └── TEST/
            ├── normal/
            └── tumor/

The exact directory names must match the current dataset provided by the
project team.

Do not rename or redistribute patches between train, validation and test.

---

## Primary Model

### EfficientNet-B0

EfficientNet-B0 is the primary/base deep learning model for this project.

The intended approach is transfer learning using ImageNet-pretrained
weights followed by fine-tuning on the histopathology patch dataset.

The model will perform binary classification:

    0 → Normal
    1 → Tumor

The saved PNG patches remain RGB uint8 images.

Model normalization is applied during the training pipeline rather than
permanently modifying the saved PNG files.

The current extraction code defines ImageNet-style normalization:

    Mean = (0.485, 0.456, 0.406)
    Std  = (0.229, 0.224, 0.225)

---

## Data Augmentation

Data augmentation is NOT performed during patch extraction.

Augmentation should be applied only during training.

Possible training augmentations include:

- horizontal/vertical flipping
- rotation
- controlled cropping/zoom
- brightness/contrast variation

Validation and test images must not receive random training augmentation.

---

## Current Repository Components

### src/

Contains the project's processing and extraction code.

Important files include:

- `extract_patches_optimized.py`
- `extract_patches_optimized_modified.py`
- `annotation_parser.py`
- `inspect_wsi.py`
- `visualize_wsi.py`
- `visualize_patches.py`
- quality-control utilities

---

## Important Rule for Future Development

Do not change the existing patch extraction methodology merely for
convenience.

The current extracted dataset should be treated as a frozen handoff
snapshot for model training.

If the extraction methodology is changed, create a new experiment/version
rather than silently replacing the existing dataset.

---

## Next Development Stage

The immediate next stage is:

### EfficientNet-B0 Training

Planned workflow:

1. Obtain the prepared patch dataset.
2. Verify train/validation/test structure.
3. Verify WSI-level separation.
4. Create PyTorch/TensorFlow dataset loaders.
5. Apply training-only augmentation.
6. Apply ImageNet normalization.
7. Load EfficientNet-B0 pretrained weights.
8. Replace the final classification layer with binary classification.
9. Train on training patches.
10. Monitor validation performance.
11. Save the best checkpoint.
12. Evaluate on the held-out test set.

---

## Planned Evaluation

The model will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC where appropriate

Training and validation loss/accuracy curves should also be recorded.

---

## Future Work

After the baseline EfficientNet-B0 model:

1. Model evaluation and error analysis
2. Grad-CAM explainability
3. Patch-level prediction visualization
4. WSI-level prediction aggregation
5. Risk/anomaly analysis
6. Optional comparison with other CNN architectures
7. Final deployment/demo

---

## Repository Structure

    patch_based_cancer_detection_project/
    │
    ├── README.md
    ├── .gitignore
    ├── requirements.txt
    │
    ├── data/
    │   ├── wsi_registry.csv
    │   └── PATCHES/          # excluded from Git
    │
    ├── src/
    │   ├── annotation_parser.py
    │   ├── extract_patches_optimized.py
    │   ├── extract_patches_optimized_modified.py
    │   ├── inspect_wsi.py
    │   ├── visualize_wsi.py
    │   └── ...
    │
    └── result/
        └── final_dataset_manifest.csv

---

## Setup

Clone the repository:

    git clone <REPOSITORY_URL>

Enter the project:

    cd patch_based_cancer_detection_project

Create a virtual environment:

    python -m venv .venv

Activate it on Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

---

## Dataset Setup

The WSI files and extracted patch images are not stored in GitHub.

Obtain the project dataset from the shared storage provided by the
project team.

After downloading/copying the dataset, verify that the directory
structure matches the paths documented in this README.

Do not modify the train/validation/test assignments.

---

## Reproducibility

The extraction pipeline is deterministic and records patch coordinates,
WSI IDs, split information, tissue measurements, tumor overlap and
selection information in its manifest.

The extractor processes one WSI per command.

Example:

    python -m src.extract_patches_optimized \
        --wsi normal_006 \
        --split train

Before running extraction on an existing WSI, verify that the current
dataset is backed up and that the intended experiment requires
re-extraction.

---

## Team Handoff

The repository is the source of truth for:

- processing code
- extraction methodology
- WSI registry
- manifests
- project configuration
- training code as it is added
- documentation

The shared dataset storage is the source of truth for:

- WSI image files
- extracted patch image files

Do not commit large WSI files, thousands of patch images, model
checkpoints or training outputs to the normal Git repository.

---

## Project Status Summary

Current:

    WSI organization              ✓
    WSI-level split               ✓
    Annotation handling           ✓
    Patch extraction              ✓
    224×224 patches               ✓
    50% candidate overlap        ✓
    Tissue/background filtering   ✓
    Quality filtering             ✓
    Spatial selection             ✓
    Patch manifest                ✓
    Dataset preparation           ✓
    EfficientNet-B0 training      → NEXT
    Model evaluation              → NEXT
    Grad-CAM                      → FUTURE
    WSI-level prediction         → FUTURE
    Deployment                   → FUTURE