import os
import shutil
import random

# Split ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BASE_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\ANOMALIES_PATCHES"

def split_dataset():
    """
    Splits the extracted anomaly patches into TRAIN, VALIDATION, and TEST sets.
    """
    classes = ["Nuclei_Pleomorphism", "Ductal_Structure_Distortion", "Tumor_Cell_Density_Clusters", "Necrosis", "Normal", "Benign"]
    
    for cls in classes:
        cls_dir = os.path.join(BASE_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        # Create directories
        for split in ["TRAIN", "VALIDATION", "TEST"]:
            os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)
            
        # Move files
        for img in train_imgs:
            shutil.move(os.path.join(cls_dir, img), os.path.join(BASE_DIR, "TRAIN", cls, img))
        for img in val_imgs:
            shutil.move(os.path.join(cls_dir, img), os.path.join(BASE_DIR, "VALIDATION", cls, img))
        for img in test_imgs:
            shutil.move(os.path.join(cls_dir, img), os.path.join(BASE_DIR, "TEST", cls, img))
            
        # Remove empty class directory
        os.rmdir(cls_dir)
        print(f"Split {n_total} images for class {cls} -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

if __name__ == "__main__":
    print("Splitting extracted patches into Train/Val/Test...")
    split_dataset()
    print("Dataset split complete.")
