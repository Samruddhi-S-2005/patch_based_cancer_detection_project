import csv
import math
from pathlib import Path

import pandas as pd


ROOT = Path("data/ANOMALIES/nucls")
ANNOTATION_DIR = ROOT / "annotations"

INPUT = Path("result/objective_2/nucls_features.csv")
OUTPUT = Path("result/objective_2/nucls_spatial_features.csv")


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def nearest_neighbor_distances(points):
    """Return nearest-neighbor distance for each point."""

    if len(points) < 2:
        return []

    distances = []

    for i, (x1, y1) in enumerate(points):

        nearest = None

        for j, (x2, y2) in enumerate(points):

            if i == j:
                continue

            d = euclidean_distance(x1, y1, x2, y2)

            if nearest is None or d < nearest:
                nearest = d

        distances.append(nearest)

    return distances


def main():

    base_df = pd.read_csv(INPUT)

    records = []

    annotation_files = sorted(ANNOTATION_DIR.glob("*.csv"))

    print(f"Annotation files: {len(annotation_files)}")

    for idx, ann_path in enumerate(annotation_files, start=1):

        rows = list(
            csv.DictReader(
                open(
                    ann_path,
                    encoding="utf-8-sig",
                    newline=""
                )
            )
        )

        tumor_points = []

        for row in rows:

            if row.get("super_classification", "").strip() != "tumor_any":
                continue

            try:
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
            except (TypeError, ValueError, KeyError):
                continue

            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0

            tumor_points.append((cx, cy))

        tumor_count = len(tumor_points)

        nn = nearest_neighbor_distances(tumor_points)

        if nn:

            mean_nn = sum(nn) / len(nn)

            sorted_nn = sorted(nn)

            middle = len(sorted_nn) // 2

            if len(sorted_nn) % 2 == 0:
                median_nn = (
                    sorted_nn[middle - 1] +
                    sorted_nn[middle]
                ) / 2.0
            else:
                median_nn = sorted_nn[middle]

            variance = sum(
                (d - mean_nn) ** 2
                for d in nn
            ) / len(nn)

            std_nn = math.sqrt(variance)

        else:
            mean_nn = 0.0
            median_nn = 0.0
            std_nn = 0.0

        matching = base_df[base_df["fov_id"] == ann_path.stem]

        if matching.empty:
            print(f"WARNING: FOV missing from base feature table: {ann_path.stem}")
            continue

        width = float(matching.iloc[0]["width"])
        height = float(matching.iloc[0]["height"])

        diagonal = math.sqrt(width ** 2 + height ** 2)

        normalized_mean_nn = (
            mean_nn / diagonal
            if diagonal > 0
            else 0.0
        )

        records.append({
            "fov_id": ann_path.stem,
            "tumor_nuclei_count_spatial": tumor_count,
            "tumor_mean_nearest_neighbor_px": mean_nn,
            "tumor_median_nearest_neighbor_px": median_nn,
            "tumor_std_nearest_neighbor_px": std_nn,
            "tumor_normalized_mean_nn": normalized_mean_nn,
        })

        if idx % 200 == 0:
            print(f"Processed: {idx}/{len(annotation_files)}")

    spatial_df = pd.DataFrame(records)

    merged = base_df.merge(
        spatial_df,
        on="fov_id",
        how="left",
        validate="one_to_one"
    )

    merged.to_csv(OUTPUT, index=False)

    print("\n===== NUCLS SPATIAL FEATURES COMPLETE =====")
    print(f"FOVs processed : {len(spatial_df)}")
    print(f"Output         : {OUTPUT}")

    print("\n===== SPATIAL SUMMARY =====")

    print(
        merged[
            [
                "tumor_nuclei_count_spatial",
                "tumor_mean_nearest_neighbor_px",
                "tumor_median_nearest_neighbor_px",
                "tumor_std_nearest_neighbor_px",
                "tumor_normalized_mean_nn",
            ]
        ].describe()
    )

    print("\nMissing spatial values:")

    print(
        merged[
            [
                "tumor_nuclei_count_spatial",
                "tumor_mean_nearest_neighbor_px",
                "tumor_median_nearest_neighbor_px",
                "tumor_std_nearest_neighbor_px",
                "tumor_normalized_mean_nn",
            ]
        ].isna().sum()
    )


if __name__ == "__main__":
    main()
