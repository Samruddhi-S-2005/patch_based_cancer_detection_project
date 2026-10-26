import os
import cv2
import numpy as np
from pathlib import Path

# Paths
INPUT_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES\bach_iciar2018\images"
OUTPUT_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES_PATCHES"

PATCH_SIZE = 224
STRIDE = 224

# Mapping BACH classes to our Anomaly classes
CLASS_MAPPING = {
    "Invasive": "Tumor_Cell_Density_Clusters",
    "InSitu": "Ductal_Structure_Distortion",
    "Normal": "Normal",
    "Benign": "Benign"
}

def is_valid_patch(patch, threshold=220, white_ratio_max=0.8):
    """
    Check if a patch contains enough tissue (not mostly white background).
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > threshold)
    total_pixels = patch.shape[0] * patch.shape[1]
    white_ratio = white_pixels / total_pixels
    return white_ratio < white_ratio_max

def process_image(image_path, target_dir):
    """
    Read an image, extract 224x224 patches, and save them.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read {image_path}")
        return

    h, w, _ = image.shape
    base_name = image_path.stem
    count = 0
    
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            if is_valid_patch(patch):
                out_path = os.path.join(target_dir, f"{base_name}_x{x}_y{y}.png")
                cv2.imwrite(out_path, patch)
                count += 1
                
    print(f"Extracted {count} patches from {base_name}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} does not exist.")
        return

    for bach_class, anomaly_class in CLASS_MAPPING.items():
        in_class_dir = os.path.join(INPUT_DIR, bach_class)
        if not os.path.exists(in_class_dir):
            print(f"Class directory {in_class_dir} not found. Skipping.")
            continue
            
        out_class_dir = os.path.join(OUTPUT_DIR, anomaly_class)
        os.makedirs(out_class_dir, exist_ok=True)
        
        for img_file in os.listdir(in_class_dir):
            if img_file.lower().endswith(('.tif', '.png', '.jpg')):
                img_path = Path(in_class_dir) / img_file
                process_image(img_path, out_class_dir)

if __name__ == "__main__":
    print("Starting BACH Preprocessing...")
    main()
    print("BACH Preprocessing Completed.")
