import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("result/objective_2/nucls_spatial_features.csv")
OUT = Path("result/objective_2/nucls_spatial_plots")

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# --------------------------------------------------
# 1. Tumor nuclei count
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["tumor_nuclei_count_spatial"],
    bins=30
)

plt.xlabel("Tumor nuclei count per FOV")
plt.ylabel("Number of FOVs")
plt.title("NuCLS Tumor Nuclei Count Distribution")

plt.tight_layout()

plt.savefig(
    OUT / "tumor_nuclei_count_distribution.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 2. Tumor nuclei ratio
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["tumor_nuclei_ratio"],
    bins=30
)

plt.xlabel("Tumor nuclei ratio")
plt.ylabel("Number of FOVs")
plt.title("NuCLS Tumor Nuclei Ratio Distribution")

plt.tight_layout()

plt.savefig(
    OUT / "tumor_nuclei_ratio_distribution.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 3. Normalized nearest-neighbor distance
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["tumor_normalized_mean_nn"],
    bins=30
)

plt.xlabel("Normalized mean nearest-neighbor distance")
plt.ylabel("Number of FOVs")
plt.title("NuCLS Tumor Spatial Clustering Distribution")

plt.tight_layout()

plt.savefig(
    OUT / "tumor_spatial_clustering_distribution.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 4. Density vs spatial distance
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["tumor_nuclei_density_per_pixel"],
    df["tumor_normalized_mean_nn"],
    alpha=0.5
)

plt.xlabel("Tumor nuclei density")
plt.ylabel("Normalized mean nearest-neighbor distance")
plt.title("Tumor Density vs Spatial Distribution")

plt.tight_layout()

plt.savefig(
    OUT / "tumor_density_vs_spatial_distance.png",
    dpi=200
)

plt.close()


print("===== NUCLS SPATIAL VISUALIZATION COMPLETE =====")
print(f"Output directory: {OUT}")

for file in sorted(OUT.glob("*.png")):
    print(file)
