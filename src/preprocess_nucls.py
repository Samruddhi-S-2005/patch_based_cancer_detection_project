import os
import cv2
from pathlib import Path

INPUT_IMG_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES\nucls\images"
OUTPUT_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES_PATCHES\Nuclei_Pleomorphism"

PATCH_SIZE = 224

def process_nucls_image(image_path):
    """
    NuCLS images are relatively small FOVs (~400-500px). 
    We will extract a 224x224 center crop.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return

    h, w, _ = image.shape
    
    # If image is smaller than 224, pad it. Otherwise, center crop.
    if h < PATCH_SIZE or w < PATCH_SIZE:
        # Resize it up slightly or pad it (here we resize for simplicity)
        image = cv2.resize(image, (max(w, PATCH_SIZE), max(h, PATCH_SIZE)))
        h, w, _ = image.shape

    # Center crop
    start_y = (h - PATCH_SIZE) // 2
    start_x = (w - PATCH_SIZE) // 2
    
    patch = image[start_y:start_y+PATCH_SIZE, start_x:start_x+PATCH_SIZE]
    
    base_name = image_path.stem
    out_path = os.path.join(OUTPUT_DIR, f"nucls_{base_name}.png")
    cv2.imwrite(out_path, patch)

def main():
    if not os.path.exists(INPUT_IMG_DIR):
        print("NuCLS image directory not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    count = 0
    
    for img_file in os.listdir(INPUT_IMG_DIR):
        if img_file.lower().endswith('.png'):
            process_nucls_image(Path(INPUT_IMG_DIR) / img_file)
            count += 1
            if count % 500 == 0:
                print(f"Processed {count} NuCLS images...")
                
    print(f"Total NuCLS images processed: {count}")

if __name__ == "__main__":
    print("Starting NuCLS Preprocessing...")
    main()
    print("NuCLS Preprocessing Completed.")
