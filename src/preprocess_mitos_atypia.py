import os
import cv2
import numpy as np
from pathlib import Path

# Paths
INPUT_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES\mitos_atypia14"
OUTPUT_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES_PATCHES\Nuclei_Pleomorphism"

PATCH_SIZE = 224
STRIDE = 224

def process_frame(image_path):
    """
    Extract patches from an x40 frame. 
    Assuming these frames contain atypia/mitosis (Nuclei Pleomorphism).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return

    h, w, _ = image.shape
    base_name = image_path.stem
    count = 0
    
    # Simple sliding window for x40 frames
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Simple tissue check (ignore mostly white/black patches)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 230 and np.mean(gray) > 20: 
                out_path = os.path.join(OUTPUT_DIR, f"mitos_{image_path.parent.parent.parent.name}_{base_name}_x{x}_y{y}.png")
                cv2.imwrite(out_path, patch)
                count += 1
                
    print(f"Extracted {count} patches from {image_path.name}")

def main():
    if not os.path.exists(INPUT_DIR):
        print("MITOS-Atypia14 directory not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Walk through the dataset looking for x40 frames
    for root, dirs, files in os.walk(INPUT_DIR):
        if "x40" in root and "frames" in root:
            for file in files:
                if file.lower().endswith(('.png', '.tif', '.jpg', '.bmp')):
                    process_frame(Path(root) / file)

if __name__ == "__main__":
    print("Starting MITOS-Atypia Preprocessing...")
    main()
    print("MITOS-Atypia Preprocessing Completed.")
