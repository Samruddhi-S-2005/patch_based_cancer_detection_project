from pathlib import Path
import csv
from PIL import Image
import numpy as np

# ============================================================
# MITOS-ATYPIA-14 X20 -> 224x224 patch extraction
# A07 + A17
#
# Official frame-level atypia score:
#   1 = low
#   2 = moderate
#   3 = high
#
# A17_02B is excluded because its official score files are empty.
#
# IMPORTANT:
# These are FRAME-LEVEL labels inherited by patches.
# Therefore they are weak/frame-derived labels, not patch-level
# nuclear annotations.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = PROJECT_ROOT / "data" / "ANOMALIES" / "mitos_atypia14"
OUTPUT_ROOT = PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "mitos_atypia"

PATCH_ROOT = OUTPUT_ROOT / "patches"
MANIFEST_PATH = OUTPUT_ROOT / "mitos_atypia_patch_manifest.csv"

PATCH_SIZE = 224
STRIDE = 112

# Background/tissue filtering.
# A pixel is considered tissue-like if at least one RGB channel
# is sufficiently below white.
WHITE_THRESHOLD = 240

# Keep a patch if >=50% of its pixels are tissue-like.
MIN_TISSUE_FRACTION = 0.50

# Exclude extremely uniform patches.
MIN_STD = 8.0

EXCLUDED_FRAMES = {"A17_02B"}

records = []
summary = []

for slide in ["A07", "A17"]:
    frame_dir = SOURCE_ROOT / slide / "frames" / "x20"
    score_dir = SOURCE_ROOT / slide / "atypia" / "x20"

    frames = sorted(frame_dir.glob("*.tiff"))

    slide_total = 0
    slide_kept = 0
    slide_discarded = 0

    print(f"\nProcessing {slide}: {len(frames)} X20 frames found")

    for frame_path in frames:
        frame_id = frame_path.stem

        if frame_id in EXCLUDED_FRAMES:
            print(f"  SKIP {frame_id}: official score files are empty")
            continue

        score_path = score_dir / f"{frame_id}_cna_score_decision.csv"

        if not score_path.exists():
            print(f"  WARNING {frame_id}: score file missing")
            continue

        score_text = score_path.read_text(encoding="utf-8").strip()

        if not score_text:
            print(f"  SKIP {frame_id}: score file empty")
            continue

        try:
            score = int(score_text)
        except ValueError:
            print(f"  WARNING {frame_id}: invalid score '{score_text}'")
            continue

        if score not in (1, 2, 3):
            print(f"  WARNING {frame_id}: unexpected score {score}")
            continue

        image = Image.open(frame_path).convert("RGB")
        arr = np.asarray(image)

        height, width = arr.shape[:2]

        frame_candidates = 0
        frame_kept = 0

        for y in range(0, height - PATCH_SIZE + 1, STRIDE):
            for x in range(0, width - PATCH_SIZE + 1, STRIDE):

                patch = arr[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                frame_candidates += 1
                slide_total += 1

                # Identify tissue-like pixels.
                tissue_mask = np.any(
                    patch < WHITE_THRESHOLD,
                    axis=2
                )

                tissue_fraction = float(tissue_mask.mean())

                # Reject mostly-background patches.
                if tissue_fraction < MIN_TISSUE_FRACTION:
                    slide_discarded += 1
                    continue

                # Reject almost completely uniform patches.
                patch_std = float(patch.std())

                if patch_std < MIN_STD:
                    slide_discarded += 1
                    continue

                # Save according to official frame-level score.
                score_dir_out = PATCH_ROOT / f"score{score}"
                score_dir_out.mkdir(parents=True, exist_ok=True)

                patch_name = (
                    f"{frame_id}_x{x}_y{y}"
                    f"_score{score}.png"
                )

                patch_path = score_dir_out / patch_name

                Image.fromarray(patch).save(
                    patch_path,
                    format="PNG"
                )

                records.append({
                    "frame_id": frame_id,
                    "source_frame": str(frame_path.relative_to(PROJECT_ROOT)),
                    "patch_path": str(patch_path.relative_to(PROJECT_ROOT)),
                    "x": x,
                    "y": y,
                    "patch_size": PATCH_SIZE,
                    "tissue_fraction": round(tissue_fraction, 6),
                    "patch_std": round(patch_std, 6),
                    "atypia_score": score,
                    "label_source": "official_frame_level_score"
                })

                frame_kept += 1
                slide_kept += 1

        image.close()

        print(
            f"  {frame_id}: score={score}, "
            f"candidates={frame_candidates}, "
            f"kept={frame_kept}"
        )

    summary.append({
        "slide": slide,
        "candidates": slide_total,
        "kept": slide_kept,
        "discarded": slide_discarded
    })

# Write manifest.
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "frame_id",
    "source_frame",
    "patch_path",
    "x",
    "y",
    "patch_size",
    "tissue_fraction",
    "patch_std",
    "atypia_score",
    "label_source"
]

with MANIFEST_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print("\n" + "=" * 60)
print("MITOS EXTRACTION COMPLETE")
print("=" * 60)

print(f"Total patches saved : {len(records)}")
print(f"Manifest             : {MANIFEST_PATH}")

print("\nScore distribution:")
for score in (1, 2, 3):
    count = sum(
        1 for r in records
        if r["atypia_score"] == score
    )
    print(f"  Score {score}: {count}")

print("\nSlide summary:")
for s in summary:
    print(
        f"  {s['slide']}: "
        f"candidates={s['candidates']}, "
        f"kept={s['kept']}, "
        f"discarded={s['discarded']}"
    )

print("\nExcluded frame:")
print("  A17_02B - official score files are empty")

print("\nNOTE:")
print("All retained patches inherit the X20 frame-level")
print("nuclear atypia score. These are weak/frame-derived labels.")
