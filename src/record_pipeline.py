import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import openslide
from torchvision import transforms
from pathlib import Path

# Add the objective_2 model to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'objective_2'))
try:
    from model import EfficientNetB0TumorClassifier
except ImportError:
    print("Warning: Could not import EfficientNetB0TumorClassifier. Model prediction will be skipped.")
    EfficientNetB0TumorClassifier = None

WSI_PATH = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\data\RAW\camelyon16\normal\normal_004.tif"
OUTPUT_DIR = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\result\pipeline_record"

PATCH_SIZE = 224

def save_image(img_array, filename, title=""):
    """Saves an image to the output directory with an optional title."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    
    # If the image is grayscale or 2D, convert to BGR for saving
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 3:
        # Convert RGB to BGR for cv2.imwrite if needed
        # Assuming input is RGB if it comes from PIL/OpenSlide
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if title:
        # Add a title bar
        h, w = img_array.shape[:2]
        title_bar = np.zeros((40, w, 3), dtype=np.uint8)
        cv2.putText(title_bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        img_array = np.vstack((title_bar, img_array))

    cv2.imwrite(out_path, img_array)
    print(f"Saved: {filename}")

def get_gradcam(model, img_tensor):
    """
    Basic Grad-CAM implementation.
    """
    # EfficientNet-B0 final conv feature map is at backbone.features[-1]
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    target_layer = model.backbone.features[-1]
    f_hook = target_layer.register_forward_hook(forward_hook)
    b_hook = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    pred = model(img_tensor)
    
    # Backward pass
    model.zero_grad()
    pred.backward(retain_graph=True)

    f_hook.remove()
    b_hook.remove()

    if not gradients or not activations:
        return None

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    # Global average pooling on gradients
    weights = np.mean(grads, axis=(1, 2))
    
    # Weighted combination of activations
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (PATCH_SIZE, PATCH_SIZE))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    return cam

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Recording pipeline steps to: {OUTPUT_DIR}")

    if not os.path.exists(WSI_PATH):
        print(f"WSI not found at {WSI_PATH}. Please ensure the path is correct.")
        return

    # --- STEP 1: WSI THUMBNAIL ---
    print("\n--- Step 1: Loading WSI ---")
    slide = openslide.OpenSlide(WSI_PATH)
    # Get a thumbnail
    thumb = slide.get_thumbnail((1024, 1024))
    thumb_np = np.array(thumb)[..., :3] # Remove alpha
    save_image(thumb_np, "01_wsi_thumbnail.png", "1. Raw Whole Slide Image")

    # --- STEP 2: PATCH EXTRACTION ---
    print("\n--- Step 2: Patch Extraction ---")
    # Grab a patch from the middle of the slide
    w, h = slide.dimensions
    center_x, center_y = w // 2, h // 2
    
    # Try to find a tissue patch
    raw_patch = None
    for offset_x in range(0, 5000, 500):
        patch_pil = slide.read_region((center_x + offset_x, center_y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        patch_np = np.array(patch_pil)
        # Check if it has tissue (not completely white)
        if np.mean(patch_np) < 220:
            raw_patch = patch_np
            break
            
    if raw_patch is None:
        raw_patch = np.array(slide.read_region((center_x, center_y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))

    save_image(raw_patch, "02_raw_extracted_patch.png", "2. Raw Extracted Patch")

    # --- STEP 3: PREPROCESSING SUB-LAYERS ---
    print("\n--- Step 3: Preprocessing Sub-Layers ---")
    
    # A. Grayscale
    gray_patch = cv2.cvtColor(raw_patch, cv2.COLOR_RGB2GRAY)
    save_image(gray_patch, "03a_grayscale.png", "3A. Grayscale Conversion")
    
    # B. Gaussian Blur (Noise reduction)
    blurred_patch = cv2.GaussianBlur(gray_patch, (5, 5), 0)
    save_image(blurred_patch, "03b_gaussian_blur.png", "3B. Gaussian Blur")
    
    # C. Thresholding (Tissue vs Background)
    _, thresh_patch = cv2.threshold(blurred_patch, 220, 255, cv2.THRESH_BINARY_INV)
    save_image(thresh_patch, "03c_thresholding.png", "3C. Binary Thresholding (Tissue Mask)")
    
    # --- STEP 4: DATA AUGMENTATION (Training only) ---
    print("\n--- Step 4: Data Augmentation ---")
    # Simulate an augmentation (Rotation and Flip)
    (h, w) = raw_patch.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
    rotated = cv2.warpAffine(raw_patch, M, (w, h))
    flipped = cv2.flip(rotated, 1) # Horizontal flip
    save_image(flipped, "04_augmented_patch.png", "4. Augmented Patch (Rotated + Flipped)")

    # --- STEP 5: NORMALIZATION ---
    print("\n--- Step 5: Normalization ---")
    # ImageNet Normalization
    img_float = flipped.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (img_float - mean) / std
    
    # To visualize normalized image, we must scale it back to 0-255 for saving
    # It will look color-shifted, which accurately represents what the network sees
    norm_vis = ((normalized - normalized.min()) / (normalized.max() - normalized.min()) * 255).astype(np.uint8)
    save_image(norm_vis, "05_normalized_patch.png", "5. Normalized Patch (ImageNet stats)")

    # --- STEP 6: MODEL PREDICTION & GRAD-CAM ---
    print("\n--- Step 6: Model Prediction & Grad-CAM ---")
    if EfficientNetB0TumorClassifier is not None:
        model_path = r"d:\Samruddhi\Projects\patch_based_cancer_detection_project\result\objective_2\checkpoints\best_model.pt"
        if os.path.exists(model_path):
            # Load model
            model = EfficientNetB0TumorClassifier(pretrained=False)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Prepare tensor
            tensor_in = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0).float()
            
            # Predict
            with torch.no_grad():
                logit = model(tensor_in)
                prob = torch.sigmoid(logit).item()
            
            pred_class = "Tumor" if prob > 0.5 else "Normal"
            print(f"Prediction: {pred_class} (Probability: {prob:.4f})")
            
            # Get Grad-CAM
            # Re-enable gradients for the tensor for Grad-CAM
            tensor_in.requires_grad_(True)
            cam = get_gradcam(model, tensor_in)
            
            if cam is not None:
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                
                # Overlay heatmap on the raw flipped image (before normalization for visual clarity)
                flipped_bgr = cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(flipped_bgr, 0.6, heatmap, 0.4, 0)
                
                title = f"6. Grad-CAM Prediction: {pred_class} ({prob:.2f})"
                save_image(overlay, "06_grad_cam_heatmap.png", title)
        else:
            print(f"Model checkpoint not found at {model_path}")

    print("\nPipeline recording complete! Check the 'result/pipeline_record' folder.")

if __name__ == "__main__":
    main()
