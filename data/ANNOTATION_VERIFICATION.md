# CAMELYON16 Annotation Verification & WSI Dataset Audit

**Audit Date:** August 21, 2026  
**Target Architecture:** WSI Ingestion & Partitioning (Step 1-2 for EfficientNet-B0 Pipeline)

---

## 1. Executive Summary & Inventory

* **Total Downloaded Training WSIs:** **60 WSIs**
  * **Normal Training WSIs:** **30 WSIs** (31.27 GB)
  * **Tumor Training WSIs:** **30 WSIs** (38.32 GB)
* **Total Downloaded XML Annotation Files:** **111 files** (55.83 MB)
* **Annotation Matching Rate for Downloaded Tumor WSIs:** **30 / 30 (100.0% Matched)**
* **Unmatched Downloaded Tumor WSIs:** **0**
* **Annotation Files for Non-Downloaded WSIs:** **81 files** (retained from the complete CAMELYON16 training set)
* **Total RAW Dataset Disk Size:** **69.60 GB** (69.66 GB including `lesion_annotations.zip` backup)

---

## 2. Comprehensive WSI & Annotation Matching Table

| # | WSI ID | Class | Dimensions (Lvl 0) | Levels | MPP (X) | Matching XML File | Status | Split (Seed 42) |
|---|---|---|---|---|---|---|---|---|
| 1 | `normal_004` | Normal | $96256 \times 218624$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TEST** |
| 2 | `normal_006` | Normal | $97792 \times 217088$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 3 | `normal_008` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 4 | `normal_012` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TEST** |
| 5 | `normal_013` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **VAL** |
| 6 | `normal_016` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 7 | `normal_017` | Normal | $97792 \times 220672$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 8 | `normal_019` | Normal | $97792 \times 221696$ | 10 | 0.243 | *N/A (Normal)* | VALID | **VAL** |
| 9 | `normal_020` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **VAL** |
| 10 | `normal_021` | Normal | $97792 \times 217088$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 11 | `normal_023` | Normal | $97792 \times 217088$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 12 | `normal_027` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 13 | `normal_038` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 14 | `normal_042` | Normal | $97280 \times 217600$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 15 | `normal_045` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 16 | `normal_046` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 17 | `normal_048` | Normal | $97792 \times 211968$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 18 | `normal_054` | Normal | $96768 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 19 | `normal_060` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 20 | `normal_062` | Normal | $97792 \times 215552$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 21 | `normal_083` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TEST** |
| 22 | `normal_084` | Normal | $97792 \times 221184$ | 10 | 0.243 | *N/A (Normal)* | VALID | **TRAIN** |
| 23 | `normal_102` | Normal | $102400 \times 82432$ | 9 | 0.227 | *N/A (Normal)* | VALID | **TRAIN** |
| 24 | `normal_103` | Normal | $102400 \times 57344$ | 9 | 0.227 | *N/A (Normal)* | VALID | **TEST** |
| 25 | `normal_108` | Normal | $45056 \times 35840$ | 8 | 0.227 | *N/A (Normal)* | VALID | **VAL** |
| 26 | `normal_109` | Normal | $114688 \times 75264$ | 9 | 0.226 | *N/A (Normal)* | VALID | **TRAIN** |
| 27 | `normal_116` | Normal | $131072 \times 39424$ | 9 | 0.227 | *N/A (Normal)* | VALID | **TRAIN** |
| 28 | `normal_123` | Normal | $90112 \times 64512$ | 9 | 0.227 | *N/A (Normal)* | VALID | **TRAIN** |
| 29 | `normal_141` | Normal | $131072 \times 46592$ | 9 | 0.227 | *N/A (Normal)* | VALID | **VAL** |
| 30 | `normal_150` | Normal | $65536 \times 60928$ | 8 | 0.227 | *N/A (Normal)* | VALID | **TRAIN** |
| 31 | `tumor_005` | Tumor | $97792 \times 219648$ | 10 | 0.243 | `tumor_005.xml` (4 poly) | VALID | **TEST** |
| 32 | `tumor_016` | Tumor | $97792 \times 221184$ | 10 | 0.243 | `tumor_016.xml` (9 poly) | VALID | **TRAIN** |
| 33 | `tumor_019` | Tumor | $97792 \times 219648$ | 10 | 0.243 | `tumor_019.xml` (4 poly) | VALID | **TRAIN** |
| 34 | `tumor_031` | Tumor | $97792 \times 221184$ | 10 | 0.243 | `tumor_031.xml` (86 poly)| VALID | **TRAIN** |
| 35 | `tumor_034` | Tumor | $97792 \times 221184$ | 10 | 0.243 | `tumor_034.xml` (8 poly) | VALID | **TRAIN** |
| 36 | `tumor_035` | Tumor | $97792 \times 221184$ | 10 | 0.243 | `tumor_035.xml` (1 poly) | VALID | **VAL** |
| 37 | `tumor_038` | Tumor | $97792 \times 221184$ | 10 | 0.243 | `tumor_038.xml` (4 poly) | VALID | **TRAIN** |
| 38 | `tumor_057` | Tumor | $97792 \times 220672$ | 10 | 0.243 | `tumor_057.xml` (20 poly)| VALID | **TRAIN** |
| 39 | `tumor_058` | Tumor | $97792 \times 221184$ | 10 | 0.243 | `tumor_058.xml` (15 poly)| VALID | **TEST** |
| 40 | `tumor_059` | Tumor | $97280 \times 221184$ | 10 | 0.243 | `tumor_059.xml` (10 poly)| VALID | **TRAIN** |
| 41 | `tumor_065` | Tumor | $97792 \times 220672$ | 10 | 0.243 | `tumor_065.xml` (11 poly)| VALID | **VAL** |
| 42 | `tumor_067` | Tumor | $97792 \times 221184$ | 10 | 0.243 | `tumor_067.xml` (1 poly) | VALID | **TRAIN** |
| 43 | `tumor_075` | Tumor | $90112 \times 78848$ | 9 | 0.226 | `tumor_075.xml` (1 poly) | VALID | **TRAIN** |
| 44 | `tumor_081` | Tumor | $90112 \times 100352$ | 9 | 0.226 | `tumor_081.xml` (2 poly) | VALID | **VAL** |
| 45 | `tumor_082` | Tumor | $200704 \times 60928$ | 10 | 0.226 | `tumor_082.xml` (6 poly) | VALID | **TRAIN** |
| 46 | `tumor_084` | Tumor | $65536 \times 86016$ | 9 | 0.226 | `tumor_084.xml` (2 poly) | VALID | **TRAIN** |
| 47 | `tumor_087` | Tumor | $106496 \times 100352$ | 9 | 0.226 | `tumor_087.xml` (2 poly) | VALID | **TRAIN** |
| 48 | `tumor_091` | Tumor | $61440 \times 53760$ | 8 | 0.226 | `tumor_091.xml` (7 poly) | VALID | **TRAIN** |
| 49 | `tumor_093` | Tumor | $98304 \times 89600$ | 9 | 0.226 | `tumor_093.xml` (1 poly) | VALID | **TEST** |
| 50 | `tumor_094` | Tumor | $118784 \times 100352$ | 9 | 0.226 | `tumor_094.xml` (1 poly) | VALID | **TRAIN** |
| 51 | `tumor_096` | Tumor | $131072 \times 71680$ | 9 | 0.226 | `tumor_096.xml` (1 poly) | VALID | **TRAIN** |
| 52 | `tumor_097` | Tumor | $122880 \times 86016$ | 9 | 0.226 | `tumor_097.xml` (4 poly) | VALID | **TRAIN** |
| 53 | `tumor_098` | Tumor | $94208 \times 107520$ | 9 | 0.227 | `tumor_098.xml` (1 poly) | VALID | **VAL** |
| 54 | `tumor_099` | Tumor | $114688 \times 100352$ | 9 | 0.226 | `tumor_099.xml` (1 poly) | VALID | **TRAIN** |
| 55 | `tumor_100` | Tumor | $118784 \times 89600$ | 9 | 0.226 | `tumor_100.xml` (2 poly) | VALID | **TEST** |
| 56 | `tumor_101` | Tumor | $139264 \times 71680$ | 10 | 0.226 | `tumor_101.xml` (1 poly) | VALID | **TEST** |
| 57 | `tumor_105` | Tumor | $106496 \times 93184$ | 9 | 0.226 | `tumor_105.xml` (160 poly)| VALID | **TRAIN** |
| 58 | `tumor_107` | Tumor | $98304 \times 82432$ | 9 | 0.226 | `tumor_107.xml` (5 poly) | VALID | **TRAIN** |
| 59 | `tumor_109` | Tumor | $131072 \times 89600$ | 9 | 0.226 | `tumor_109.xml` (1 poly) | VALID | **TRAIN** |
| 60 | `tumor_110` | Tumor | $94208 \times 71680$ | 9 | 0.227 | `tumor_110.xml` (39 poly)| VALID | **TRAIN** |

---

## 3. XML Technical Integrity Verification

All 30 tumor annotation files were parsed and programmatically validated:
1. **XML Schema Integrity:** Successfully parsed `<ASAP_Annotations>`, `<Annotations>`, `<Annotation>`, `<Coordinates>`, `<Coordinate>` hierarchy without syntax errors.
2. **Coordinate Space Alignment:** All polygon vertices $(X, Y)$ were confirmed to lie strictly within the Level-0 pixel dimensions $[0, W_{\text{lvl0}}] \times [0, H_{\text{lvl0}}]$ of the corresponding WSI.
3. **Lesion Morphology Range:** Captures both focal micro-metastases (e.g., `tumor_067`, `tumor_075` with 1 compact contour) and extensive multinodular infiltrations (e.g., `tumor_105` with 160 distinct lesion contours and `tumor_031` with 86 contours).

---

## 4. Reproducible WSI-Level Splitting (Seed = 42)

To prevent data leakage, dataset partitioning is enforced strictly at the **Whole Slide Image level**:

```text
+-------------------------------------------------------------------------------+
| SPLIT          | NORMAL WSIs       | TUMOR WSIs        | TOTAL WSIs (RATIO)   |
+-------------------------------------------------------------------------------+
| TRAIN          | 21 WSIs           | 21 WSIs           | 42 WSIs (70.0%)      |
| VALIDATION     |  5 WSIs           |  4 WSIs           |  9 WSIs (15.0%)      |
| INTERNAL TEST  |  4 WSIs           |  5 WSIs           |  9 WSIs (15.0%)      |
+-------------------------------------------------------------------------------+
| TOTAL          | 30 WSIs           | 30 WSIs           | 60 WSIs (100.0%)     |
+-------------------------------------------------------------------------------+
```

* **No Data Leakage:** Patches extracted from a training WSI will never appear in validation or testing sets.
* **Balanced Class Distributions:** Both training ($50/50$), validation ($55/45$), and test ($45/55$) splits maintain strict class balance.
* **Official Test Set Intact:** No official CAMELYON16 test slides were downloaded or mixed into the internal test set.

---

## 5. Dataset Readiness Assessment

* **Status:** **READY FOR STAGE 2 (Annotation-Aware Patch Extraction)**.
* All 60 WSIs are validated, categorized, indexed, and partitioned in [`data/wsi_registry.csv`](file:///d:/Samruddhi/Projects/patch_based_cancer_detection_project/data/wsi_registry.csv).
* The 40 prototype patches in [`data/dataset_manifest.csv`](file:///d:/Samruddhi/Projects/patch_based_cancer_detection_project/data/dataset_manifest.csv) remain preserved and functional.
