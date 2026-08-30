import pandas as pd
from PIL import Image, ImageDraw, ImageFont

df = pd.read_csv("result/second_stage_qc_audit.csv")

r = df[(df.qc_flag == "REVIEW") & (df.removal_reason == "blur")].copy()

r["bin"] = pd.qcut(r.blur_score, 10, duplicates="drop")
s = r.groupby("bin", observed=True).sample(n=1, random_state=42)
s = s.sort_values("blur_score")

thumb = 256
cols = 5
rows = (len(s) + cols - 1) // cols

out = Image.new("RGB", (thumb * cols, 330 * rows), "white")
d = ImageDraw.Draw(out)
font = ImageFont.load_default()

for i, (_, row) in enumerate(s.iterrows()):
    im = Image.open(row.patch_path).convert("RGB")
    im.thumbnail((thumb, thumb))

    x = (i % cols) * thumb
    y = (i // cols) * 330

    out.paste(im, (x, y))

    d.text((x, y + 260), str(row.patch_name), fill="black", font=font)
    d.text((x, y + 275), "class={} blur={:.1f}".format(row["class"], row.blur_score), fill="black", font=font)
    d.text((x, y + 290), "tissue={:.3f} white={:.3f}".format(row.tissue_ratio, row.white_ratio), fill="black", font=font)
    d.text((x, y + 305), "reason=blur", fill="black", font=font)

out.save("result/blur_review_representative.png")

print("Created: result/blur_review_representative.png")
print("Total blur REVIEW patches:", len(r))
print("Sampled:", len(s))
print()
print(s[["patch_name", "class", "blur_score", "tissue_ratio", "white_ratio"]].to_string(index=False))
