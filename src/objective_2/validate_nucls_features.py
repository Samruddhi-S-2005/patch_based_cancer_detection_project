import csv
from pathlib import Path
import pandas as pd

root = Path("data/ANOMALIES/nucls")
annotations = sorted((root / "annotations").glob("*.csv"))

df = pd.read_csv("result/objective_2/nucls_features.csv")
feature_lookup = dict(zip(df["fov_id"], df["tumor_nuclei"]))

raw_total = 0
raw_tumor = 0
raw_all_tumor = 0
raw_no_tumor = 0
mismatches = []

for ann in annotations:

    rows = list(csv.DictReader(
        open(ann, encoding="utf-8-sig", newline="")
    ))

    total = len(rows)

    tumor = sum(
        1 for row in rows
        if row.get("super_classification", "").strip() == "tumor_any"
    )

    raw_total += total
    raw_tumor += tumor

    if tumor == total and total > 0:
        raw_all_tumor += 1

    if tumor == 0:
        raw_no_tumor += 1

    predicted = feature_lookup.get(ann.stem)

    if predicted is None or int(predicted) != tumor:
        mismatches.append(
            (ann.name, tumor, predicted)
        )

print("\n===== RAW vs FEATURE VALIDATION =====")
print("Raw annotation FOVs :", len(annotations))
print("Feature-table FOVs  :", len(df))
print("Raw total nuclei    :", raw_total)
print("Raw tumor nuclei    :", raw_tumor)
print("All-tumor FOVs      :", raw_all_tumor)
print("No-tumor FOVs       :", raw_no_tumor)
print("Mismatched FOVs     :", len(mismatches))

if mismatches:
    print("\nFirst mismatches:")
    for item in mismatches[:10]:
        print(item)

print("\n===== RAW GLOBAL TUMOR FRACTION =====")
print(
    "Tumor nuclei / total nuclei =",
    round(raw_tumor / raw_total, 6)
)
