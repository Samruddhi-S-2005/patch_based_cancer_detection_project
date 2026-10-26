import os
import cv2
import numpy as np
from pathlib import Path

# Paths
INPUT_IMG_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES\bcss\images"
INPUT_MASK_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES\bcss\masks"
OUTPUT_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES_PATCHES"

PATCH_SIZE = 224
STRIDE = 224

# BCSS Class mapping (Assuming typical mask values, adjust if needed)
# 1 = Tumor, 4 = Necrosis (Example values, standard for BCSS is often RGB mapped or index mapped)
TUMOR_VAL = 1 
NECROSIS_VAL = 4

def get_dominant_class(mask_patch):
    """
    Determine the dominant tissue type in the mask patch.
    """
    # Just checking for presence/density of necrosis or tumor
    # This assumes mask is a single channel image with class indices
    total_pixels = mask_patch.size
    
    necrosis_ratio = np.sum(mask_patch == NECROSIS_VAL) / total_pixels
    tumor_ratio = np.sum(mask_patch == TUMOR_VAL) / total_pixels
    
    if necrosis_ratio > 0.3:
        return "Necrosis"
    elif tumor_ratio > 0.5:
        return "Tumor_Cell_Density_Clusters"
    else:
        return None

def process_wsi(image_path, mask_path):
    print(f"Processing {image_path.name}...")
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) # read as grayscale/index
    
    if image is None or mask is None:
        print(f"Failed to read image or mask for {image_path.name}")
        return

    # Ensure mask and image are same size
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w, _ = image.shape
    base_name = image_path.stem
    count = 0
    
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            anomaly_class = get_dominant_class(mask_patch)
            
            if anomaly_class:
                patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                out_dir = os.path.join(OUTPUT_DIR, anomaly_class)
                os.makedirs(out_dir, exist_ok=True)
                
                out_path = os.path.join(out_dir, f"bcss_{base_name}_x{x}_y{y}.png")
                cv2.imwrite(out_path, patch)
                count += 1
                
    print(f"Extracted {count} patches from {base_name}")

def main():
    if not os.path.exists(INPUT_IMG_DIR) or not os.path.exists(INPUT_MASK_DIR):
        print("BCSS image or mask directory not found.")
        return

    for img_file in os.listdir(INPUT_IMG_DIR):
        if img_file.lower().endswith(('.png', '.tif')):
            img_path = Path(INPUT_IMG_DIR) / img_file
            
            # Find corresponding mask
            # Assuming mask has same name or ends with _mask
            mask_name = img_file
            mask_path = Path(INPUT_MASK_DIR) / mask_name
            
            if not mask_path.exists():
                mask_name = img_file.replace('.png', '_mask.png')
                mask_path = Path(INPUT_MASK_DIR) / mask_name
                
            if mask_path.exists():
                process_wsi(img_path, mask_path)
            else:
                print(f"Mask not found for {img_file}")

if __name__ == "__main__":
    print("Starting BCSS Preprocessing...")
    main()
    print("BCSS Preprocessing Completed.")
