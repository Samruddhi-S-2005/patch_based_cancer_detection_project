from pathlib import Path
import openslide

# Path to CAMELYON16 data
DATA_DIR = Path("data/RAW/camelyon16")

folders = ["normal", "tumor"]

for folder in folders:
    folder_path = DATA_DIR / folder

    print("\n" + "=" * 60)
    print(f"{folder.upper()} WSI FILES")
    print("=" * 60)

    for wsi_path in sorted(folder_path.glob("*.tif")):

        print(f"\nFile: {wsi_path.name}")

        slide = openslide.OpenSlide(str(wsi_path))

        print(f"Dimensions: {slide.dimensions}")
        print(f"Number of levels: {slide.level_count}")
        print(f"Level dimensions: {slide.level_dimensions}")
        print(f"Level downsamples: {slide.level_downsamples}")

        print("Properties:")
        for key, value in slide.properties.items():
            if "MPP" in key or "OBJECTIVE" in key or "MAGNIFICATION" in key:
                print(f"  {key}: {value}")

        slide.close()