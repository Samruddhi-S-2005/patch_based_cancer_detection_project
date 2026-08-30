from pathlib import Path
import openslide
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = Path("data/RAW/camelyon16")

slides = [
    ("normal", "normal_083.tif"),
    ("normal", "normal_084.tif"),
    ("tumor", "tumor_105.tif"),
    ("tumor", "tumor_107.tif"),
]

for label, filename in slides:

    path = DATA_DIR / label / filename

    print(f"Opening: {path}")

    slide = openslide.OpenSlide(str(path))

    # Use a low-resolution level for thumbnail
    level = slide.level_count - 1

    thumbnail = slide.read_region(
        (0, 0),
        level,
        slide.level_dimensions[level]
    ).convert("RGB")

    print(
        f"{filename}: "
        f"original={slide.dimensions}, "
        f"thumbnail={thumbnail.size}"
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(thumbnail)
    plt.title(f"{label.upper()} - {filename}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    slide.close()