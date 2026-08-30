import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

df = pd.read_csv("result/second_stage_qc_audit.csv")

r = (
    df[(df.qc_flag == "REVIEW") & (df.tissue_ratio < 0.20)]
    .sort_values("tissue_ratio")
    .copy()
)

print("Creating visual review of", len(r), "patches...")

W, H = 320, 380
cols = 2
rows = len(r)

out = Image.new("RGB", (cols * W, rows * H), "white")
draw = ImageDraw.Draw(out)

try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for i, (_, row) in enumerate(r.iterrows()):
    patch = Image.open(row.patch_path).convert("RGB")
    patch.thumbnail((300, 300))

    x = (i % cols) * W + 10
    y = (i // cols) * H + 10

    out.paste(patch, (x, y))

    draw.text(
        (x, y + 305),
        f"{row.patch_name}",
        fill="black",
        font=font
    )

    draw.text(
        (x, y + 325),
        f"class={row['class']} tissue={row.tissue_ratio:.4f}",
        fill="black",
        font=font
    )

    draw.text(
        (x, y + 345),
        f"blur={row.blur_score:.2f}",
        fill="black",
        font=font
    )

out.save("result/low_tissue_review_7.png")

print("Created: result/low_tissue_review_7.png")
