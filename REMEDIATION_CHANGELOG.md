# Phase 1 Reliability Remediation — Change Log

**Start:** 2026-09-02
**Branch:** `remediation/phase1-reliability`
**Backup location:** `backups/pre_remediation_20260902/`

## Safety Actions

- [x] Created git branch `remediation/phase1-reliability` from `main`
- [x] Backed up `best_model.pt`, `final_model.pt`, `anomaly_linear_heads_best.pt`
- [x] Backed up `wsi_registry.csv`, all patch manifests, all anomaly manifests
- [x] Backed up original `predict_anomalies.py` and `wsi_manager.py`

## Changes

(Entries appended as changes are made)

## Task 1 + Task 4 — predict_anomalies.py fixes (2026-10-18)

### Task 1: Fixed evaluation sample
- **Root cause:** `select_patches` used parent directory names (`normal/` vs `tumor/`) instead of manifest labels. The `tumor/` directory contains mostly label0 patches (normal tissue from tumor slides), so all 100 sampled patches were label0.
- **Fix:** Added `select_patches_from_manifest()` function that reads `test_patch_manifest.csv`, filters to `final_decision=='KEEP'`, samples by verified `label` column (seed=42), and includes both label0 and label1 patches.
- **Verification:** 50 label0 + 50 label1 patches confirmed. All 8 test WSIs represented. Per-patch log shows patch ID, label, WSI. Fails with RuntimeError if either class is missing.
- **Output CSV:** Now includes `reference_label` and `wsi_id` columns.

### Task 4: Unified preprocessing
- **Root cause:** `predict_anomalies.py` defined `IMAGENET_MEAN`, `IMAGENET_STD`, and `EVAL_TRANSFORM` locally instead of importing from `dataset.py`.
- **Fix:** Replaced local definitions with `from src.objective_2.dataset import get_eval_transforms` and `EVAL_TRANSFORM = get_eval_transforms()`. Removed `from torchvision import transforms` import (no longer needed).
- **Verification:** Confirmed transforms produce identical output on test image. Sanity test passes with imported transform.

### Inference re-run results (100 patches, 50/50 split)
- Tumor prob: label0 mean=0.079, label1 mean=0.958 (model correctly separates classes)
- All 4 probabilities in [0,1], determinism verified

## Task 2 — Atypia-head violation investigation and clean checkpoint (2026-10-20)

### Forensic findings
- **When:** Oct 14 2026, 19:25–19:27 (confirmed by `train_anomaly_heads.log` timestamp)
- **What happened:** `train_anomaly_heads.py` included `atypia` in `datasets_info` with `pos_weight=9.50` and the MITOS-ATYPIA-14 manifest. The script detected the three permitted heads already trained in the existing checkpoint, preserved them, and then trained only the missing `atypia` head. The overwritten checkpoint file then contained all 4 heads.
- **Why it ran:** The `tasks_to_train` logic at line 439 enumerates `datasets_info.keys()` and trains any task not yet in `existing_thresholds`. Since `atypia` had no existing threshold, it was selected.
- **Atypia performance:** Val PR-AUC = 0.310 (near-random), ROC-AUC = 0.814, best epoch 29, opt threshold = 0.050. Consistent with the known slide confounding (Score 3 exclusively from A07, Score 1 exclusively from A17).
- **Effect on permitted heads:** Three permitted head weights are **bit-for-bit identical** to the original Sep-2 training. Verified on PR-AUC, threshold, and best_epoch for all three tasks.
- **Inference impact:** `predict_anomalies.py` never called the atypia head (not in its `ANOMALY_THRESHOLDS`). Output was never corrupted.

### Actions taken
1. **Contaminated checkpoint preserved** (read-only): `backups/pre_remediation_20260902/anomaly_checkpoints/anomaly_linear_heads_best.pt`
2. **Clean checkpoint created**: `result/objective_2/anomaly_model/checkpoints/anomaly_linear_heads_clean.pt`
   - Contains: backbone + ductal/density/necrosis heads only
   - atypia keys (`heads.atypia.1.weight`, `heads.atypia.1.bias`) **removed**
   - `tasks` field: `['ductal', 'density', 'necrosis']`
   - `remediation_note` field documents the forensic history
3. **`anomaly_model.py`** updated:
   - `PERMITTED_TASKS = ["ductal", "density", "necrosis"]`
   - `TASKS` aliased to `PERMITTED_TASKS` (doc comment explains the history)
   - `forward_all_heads()` iterates `PERMITTED_TASKS` only
   - Module docstring warns against calling `forward_features(feat, 'atypia')`
4. **`predict_anomalies.py`**: `ANOMALY_MODEL_CKPT` now points to `anomaly_linear_heads_clean.pt`; `load_state_dict(..., strict=False)` with guard that errors if any permitted head key is missing
5. **`evaluate_anomaly_heads.py`**: loads `anomaly_linear_heads_clean.pt` with same `strict=False` guard

### Verification
- Sanity test against clean checkpoint: ALL PASS
- Probabilities identical to Sep-2 run (identical weights for permitted heads)
- atypia task not in `optimal_thresholds`, not in `PERMITTED_TASKS`, not called in any inference path
