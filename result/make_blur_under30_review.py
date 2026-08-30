import pandas as pd
from PIL import Image, ImageDraw, ImageFont

df = pd.read_csv("result/second_stage_qc_audit.csv")

r = df[
    (df["qc_flag"] == "REVIEW") &
    (df["blur_score"] < 30)
].copy()

tumor = r[r["class"] == "tumor"].sort_values("blur_score").head(15)
normal = r[r["class"] == "normal"].sort_values("blur_score").head(15)

parts = pd.concat([tumor, normal], ignore_index=True)

cols = 5
W = 320
H = 370
rows = (len(parts) + cols - 1) // cols

out = Image.new("RGB", (cols * W, rows * H), "white")
draw = ImageDraw.Draw(out)

try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for i, (_, row) in enumerate(parts.iterrows()):

    patch = Image.open(row["patch_path"]).convert("RGB")
    patch.thumbnail((290, 290))

    x = (i % cols) * W + 15
    y = (i // cols) * H + 10

    out.paste(patch, (x, y))

    draw.text(
        (x, y + 295),
        f"{row['class']} | blur={row['blur_score']:.2f}",
        fill="black",
        font=font
    )

    draw.text(
        (x, y + 315),
        f"tissue={row['tissue_ratio']:.3f}",
        fill="black",
        font=font
    )

    draw.text(
        (x, y + 335),
        f"{row['patch_name'][:35]}",
        fill="black",
        font=font
    )

out.save("result/blur_under30_review_30.png")

print("Created: result/blur_under30_review_30.png")
print("Tumor patches:", len(tumor))
print("Normal patches:", len(normal))
print("Total:", len(parts))
