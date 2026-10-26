import csv
from pathlib import Path
from PIL import Image
import pandas as pd


ROOT = Path("data/ANOMALIES/nucls")
ANNOTATION_DIR = ROOT / "annotations"
IMAGE_DIR = ROOT / "images"
OUTPUT = Path("result/objective_2/nucls_features.csv")


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(ANNOTATION_DIR.glob("*.csv"))

    records = []

    print(f"Found annotation files: {len(annotation_files)}")

    for idx, ann_path in enumerate(annotation_files, start=1):

        image_name = ann_path.stem
        image_path = IMAGE_DIR / f"{image_name}.png"

        if not image_path.exists():
            print(f"WARNING: Missing image: {image_name}")
            continue

        # Read annotation CSV.
        # The first column is an unnamed object/index column in NuCLS.
        rows = []

        with open(
            ann_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                rows.append(row)

        if not rows:
            continue

        # Count nuclei using the official NuCLS classification fields.
        total_nuclei = len(rows)

        tumor_nuclei = sum(
            1
            for row in rows
            if str(row.get("super_classification", "")).strip()
            == "tumor_any"
        )

        # Read image dimensions.
        with Image.open(image_path) as img:
            width, height = img.size

        area_pixels = width * height

        tumor_ratio = (
            tumor_nuclei / total_nuclei
            if total_nuclei > 0
            else 0.0
        )

        total_density = (
            total_nuclei / area_pixels
            if area_pixels > 0
            else 0.0
        )

        tumor_density = (
            tumor_nuclei / area_pixels
            if area_pixels > 0
            else 0.0
        )

        records.append({
            "fov_id": image_name,
            "image_filename": f"{image_name}.png",
            "annotation_filename": ann_path.name,
            "width": width,
            "height": height,
            "area_pixels": area_pixels,
            "total_nuclei": total_nuclei,
            "tumor_nuclei": tumor_nuclei,
            "tumor_nuclei_ratio": tumor_ratio,
            "total_nuclei_density_per_pixel": total_density,
            "tumor_nuclei_density_per_pixel": tumor_density,
        })

        if idx % 100 == 0:
            print(f"Processed: {idx}/{len(annotation_files)}")

    df = pd.DataFrame(records)

    df.to_csv(OUTPUT, index=False)

    print("\n===== NUCLS FEATURE EXTRACTION COMPLETE =====")
    print(f"FOVs processed : {len(df)}")
    print(f"Output         : {OUTPUT}")

    if len(df) > 0:
        print("\n===== SUMMARY =====")
        print(df[
            [
                "total_nuclei",
                "tumor_nuclei",
                "tumor_nuclei_ratio",
                "total_nuclei_density_per_pixel",
                "tumor_nuclei_density_per_pixel",
            ]
        ].describe())


if __name__ == "__main__":
    main()
