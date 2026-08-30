from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATCH_ROOT = PROJECT_ROOT / "data" / "PATCHES" / "REDESIGNED" / "train"

OUTPUT_DIR = PROJECT_ROOT / "data" / "PATCHES" / "QC_SAMPLE"

CLASSES = ["normal", "tumor"]

SAMPLES_PER_CLASS = 100

GRID_COLUMNS = 10
THUMBNAIL_SIZE = 160

RANDOM_SEED = 42


# ============================================================
# SETUP
# ============================================================

random.seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONTACT SHEET CREATION
# ============================================================

def create_contact_sheet(image_paths, output_path, title):

    rows = (len(image_paths) + GRID_COLUMNS - 1) // GRID_COLUMNS

    title_height = 50

    sheet_width = GRID_COLUMNS * THUMBNAIL_SIZE
    sheet_height = title_height + rows * THUMBNAIL_SIZE

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (10, 15),
        title,
        fill="black"
    )

    for index, image_path in enumerate(image_paths):

        try:
            image = Image.open(image_path).convert("RGB")

            image.thumbnail(
                (THUMBNAIL_SIZE - 10, THUMBNAIL_SIZE - 10)
            )

            x = (index % GRID_COLUMNS) * THUMBNAIL_SIZE
            y = title_height + (index // GRID_COLUMNS) * THUMBNAIL_SIZE

            paste_x = x + (THUMBNAIL_SIZE - image.width) // 2
            paste_y = y + (THUMBNAIL_SIZE - image.height) // 2

            sheet.paste(image, (paste_x, paste_y))

        except Exception as e:
            print(f"Could not read {image_path}: {e}")

    sheet.save(output_path)

    print(f"Created: {output_path}")


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in CLASSES:

    class_dir = PATCH_ROOT / class_name

    if not class_dir.exists():
        print(f"ERROR: Directory does not exist: {class_dir}")
        continue

    image_paths = list(class_dir.glob("*.png"))

    print()
    print("=" * 60)
    print(f"CLASS: {class_name}")
    print(f"Available patches: {len(image_paths)}")

    if len(image_paths) <= SAMPLES_PER_CLASS:
        selected = image_paths
    else:
        selected = random.sample(
            image_paths,
            SAMPLES_PER_CLASS
        )

    output_path = OUTPUT_DIR / f"{class_name}_inspection.png"

    create_contact_sheet(
        selected,
        output_path,
        f"{class_name.upper()} PATCHES - RANDOM SAMPLE"
    )

print()
print("=" * 60)
print("INSPECTION COMPLETE")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)