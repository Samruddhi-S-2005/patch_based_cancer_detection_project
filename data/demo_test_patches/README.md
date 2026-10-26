# Curated Demo Test Patches for Live Viva Demonstration

This directory contains **10 pre-curated, verified histopathology test patches** sampled from the **strictly held-out CAMELYON16 test WSIs** (patients never seen during model training or validation).

Use these patches during your live viva presentation to demonstrate manual file uploads in the web application:

### 🟢 Normal Tissue Patches (Expected: Normal Lymph Node, P(Tumor) < 1.0%)
1. `Demo_Normal_Patch_01_normal_004_x21168_y73360_label0.png` -> P(Tumor) = 0.21%
2. `Demo_Normal_Patch_02_normal_004_x21168_y73472_label0.png` -> P(Tumor) = 0.63%
3. `Demo_Normal_Patch_03_normal_004_x21168_y82656_label0.png` -> P(Tumor) = 0.65%
4. `Demo_Normal_Patch_04_normal_004_x21168_y82768_label0.png` -> P(Tumor) = 0.01%
5. `Demo_Normal_Patch_05_normal_004_x21168_y82880_label0.png` -> P(Tumor) = 0.07%

### 🔴 Metastatic Carcinoma Patches (Expected: Metastatic Carcinoma, P(Tumor) > 99.5%)
1. `Demo_Tumor_Patch_01_tumor_005_x57904_y137760_label1.png` -> P(Tumor) = 99.98%
2. `Demo_Tumor_Patch_02_tumor_005_x57904_y137872_label1.png` -> P(Tumor) = 99.97%
3. `Demo_Tumor_Patch_03_tumor_005_x57904_y137984_label1.png` -> P(Tumor) = 99.83%
4. `Demo_Tumor_Patch_04_tumor_005_x58016_y137984_label1.png` -> P(Tumor) = 99.62%
5. `Demo_Tumor_Patch_05_tumor_005_x58016_y138096_label1.png` -> P(Tumor) = 99.91%

### How to use during the Viva Demonstration:
1. Log in to the web workstation (`researcher` / `ResearchPass2026!`).
2. Go to **🔬 Patch Analysis**.
3. Under **"Upload Custom Histopathology Patch"**, click **Browse files**.
4. Navigate to `data/demo_test_patches/` and select any Normal patch to show a clean negative diagnosis.
5. Select any Tumor patch to show a clean positive diagnosis with focused Grad-CAM heatmaps.
