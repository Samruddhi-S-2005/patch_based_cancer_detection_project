import numpy as np
from PIL import Image
from pathlib import Path
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.objective_2.layered_pipeline import LayeredDiagnosticPipeline
from src.objective_2.wsi_live_engine import process_wsi_canvas

wsi_path = Path('data/sample_wsi/sample_wsi_tumor_058_biopsy.tif')
assert wsi_path.exists(), f"WSI missing: {wsi_path}"

img = Image.open(wsi_path)
wsi_rgb = np.array(img.convert('RGB'))
w, h = img.size

print("=== WSI CALCULATION AUDIT ===")
print(f"File: {wsi_path.name}")
print(f"Source Canvas Dimensions: {w} x {h} px")
print(f"Pyramid Level: Level 0 (Stitched full-resolution canvas)")

cls_ckpt = Path('result/objective_2/checkpoints/best_model.pt')
anom_ckpt = Path('result/objective_2/anomaly_model/checkpoints/anomaly_linear_heads_clean.pt')
pipeline = LayeredDiagnosticPipeline(classifier_ckpt=cls_ckpt, anomaly_ckpt=anom_ckpt, device='cpu')

res = process_wsi_canvas(wsi_rgb=wsi_rgb, pipeline=pipeline, orig_dims=img.size)
df = res['df_tiles']

print(f"\nTotal Grid Tiles: {len(df)}")
rejected = df[df['is_tissue'] == False]
accepted = df[df['is_tissue'] == True]

print(f"Rejected Tiles Count: {len(rejected)}")
print(f"Accepted Viable Tiles Count: {len(accepted)}")
assert len(df) == 48, f"Expected 48 tiles, got {len(df)}"
assert len(rejected) == 4, f"Expected 4 rejected tiles, got {len(rejected)}"
assert len(accepted) == 44, f"Expected 44 accepted tiles, got {len(accepted)}"

print("\n--- Detailed Audit of 4 Rejected Tiles ---")
for _, r in rejected.iterrows():
    print(f"  Tile {r['tile_id']} (Row {r['row']}, Col {r['col']}): X={r['wsi_x']}, Y={r['wsi_y']} | QC Status={r['qc_status']} | Tissue Frac={r['tissue_fraction']:.1f}% | Blur Score={r['blur_score']:.1f} | Assigned P={r['pred_prob']} | Risk={r['risk_tier']}")
    assert r['pred_prob'] == 0.0, f"Rejected tile has non-zero probability: {r['pred_prob']}"
    assert r['verdict'] == 'Background', f"Rejected tile verdict not Background: {r['verdict']}"
    assert r['qc_status'] == 'BACKGROUND_GLASS', f"QC status not BACKGROUND_GLASS: {r['qc_status']}"

# Check accepted tiles
malignant_accepted = accepted[accepted['pred_label'] == 1]
benign_accepted = accepted[accepted['pred_label'] == 0]

print(f"\n--- Accepted Viable Tissue Tiles Accounting ---")
print(f"  Malignant Tiles (P >= 0.50): {len(malignant_accepted)}")
print(f"  Benign Tiles (P < 0.50):    {len(benign_accepted)}")
assert len(malignant_accepted) == 14, f"Expected 14 malignant tiles, got {len(malignant_accepted)}"
assert len(benign_accepted) == 30, f"Expected 30 benign tiles, got {len(benign_accepted)}"

# Recalculate Tumor Burden
recalc_burden = (len(malignant_accepted) / len(accepted)) * 100.0
reported_burden = res['summary']['tumor_burden_pct']
print(f"  Recalculated Tumor Burden: {len(malignant_accepted)} / {len(accepted)} = {recalc_burden:.2f}%")
print(f"  Reported Tumor Burden:     {reported_burden:.2f}%")
assert round(recalc_burden, 2) == reported_burden, f"Tumor burden mismatch: {recalc_burden} vs {reported_burden}"

# Verify Hotspot Crop Coordinates
print("\n--- Hotspot Crops Coordinate Verification ---")
tile_crops = res['tile_crops']
top_tiles = accepted.sort_values(by='pred_prob', ascending=False).head(3)
for _, t in top_tiles.iterrows():
    t_id = t['tile_id']
    crop = tile_crops[t_id]
    assert crop.shape == (224, 224, 3), f"Crop shape invalid: {crop.shape}"
    x0, y0 = t['wsi_x'], t['wsi_y']
    x1, y1 = x0 + 224, y0 + 224
    expected_slice = wsi_rgb[y0:y1, x0:x1]
    assert np.array_equal(crop, expected_slice), f"Hotspot crop {t_id} does not match WSI slice at ({x0}, {y0})!"
    print(f"  Verified Hotspot Crop: {t_id} at X=[{x0}, {x1}], Y=[{y0}, {y1}] (Exact pixel match with WSI Level 0 canvas)")

print("\nALL WSI CALCULATIONS INDEPENDENTLY REPRODUCED & VERIFIED!")
