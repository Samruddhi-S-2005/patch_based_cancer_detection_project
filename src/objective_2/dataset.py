"""
Objective 2 Dataset & DataLoader Module.

Handles loading histopathology patches for Train and Validation sets.
Enforces strict split separation, patch-level label parsing from verified filenames,
and ImageNet normalization.

TEST dataset loading is intentionally excluded from training and validation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """Returns training data augmentation & normalization transforms."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Returns deterministic validation data preprocessing transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class HistopathPatchDataset(Dataset):
    """
    PyTorch Dataset for 224x224 histopathology patches.
    Extracts patch-level ground truth binary label (0=normal, 1=tumor)
    from the verified filename pattern and cross-validates against manifests.
    """

    def __init__(
        self,
        patch_dir: str | Path,
        manifest_path: Optional[str | Path] = None,
        split_name: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.patch_dir = Path(patch_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.split_name = split_name.lower()
        self.transform = transform or (get_train_transforms() if self.split_name == "train" else get_eval_transforms())

        if not self.patch_dir.exists():
            raise FileNotFoundError(f"Patch directory does not exist: {self.patch_dir}")

        # Scan patches from disk
        self.samples: List[Dict] = []
        self._load_and_validate_samples()

    def _parse_filename_metadata(self, file_path: Path) -> Tuple[str, int, int, int]:
        """
        Parses {wsi_id}_x{x}_y{y}_label{label}... from filename.
        Returns (wsi_id, x, y, label).
        """
        name = file_path.stem
        # Match standard pattern: normal_083_x32343_y116899_label0... or tumor_105_x16167_y46657_label1...
        match = re.search(r'^(normal_\d+|tumor_\d+)_x(\d+)_y(\d+)_label(\d+)', name)
        if match:
            wsi_id = match.group(1)
            x = int(match.group(2))
            y = int(match.group(3))
            label = int(match.group(4))
            return wsi_id, x, y, label
        
        # Fallback pattern if format varies
        label_match = re.search(r'label(\d+)', name)
        if not label_match:
            raise ValueError(f"Could not parse binary label from filename: {file_path.name}")
        label = int(label_match.group(1))
        parts = name.split("_")
        wsi_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "unknown"
        return wsi_id, -1, -1, label

    def _load_and_validate_samples(self) -> None:
        # Load manifest if available (especially for validation/test manifests)
        manifest_lookup = {}
        if self.manifest_path and self.manifest_path.exists():
            df_manifest = pd.read_csv(self.manifest_path)
            # If manifest has patch-level label column matching filename naming
            name_col = None
            for col in ['saved_path', 'patch_name', 'filename']:
                if col in df_manifest.columns:
                    name_col = col
                    break
            
            if name_col and 'label' in df_manifest.columns:
                for _, row in df_manifest.iterrows():
                    raw_name = str(row[name_col])
                    base_name = Path(raw_name).name
                    # Only map if this manifest contains patch-level annotations
                    manifest_lookup[base_name] = int(row['label'])

        # Discover all PNG files in normal/ and tumor/ subdirectories
        png_files = sorted(list(self.patch_dir.rglob("*.png")))
        if len(png_files) == 0:
            raise ValueError(f"No PNG patch files found in {self.patch_dir}")

        for file_path in png_files:
            wsi_id, x, y, label = self._parse_filename_metadata(file_path)
            
            # Cross-check with validation/eval manifest if present
            if self.split_name in ["validation", "test"] and file_path.name in manifest_lookup:
                manifest_lbl = manifest_lookup[file_path.name]
                if manifest_lbl != label:
                    raise ValueError(
                        f"Label mismatch for {file_path.name}: "
                        f"filename indicated label {label} but manifest indicated {manifest_lbl}!"
                    )

            self.samples.append({
                "path": str(file_path),
                "filename": file_path.name,
                "wsi_id": wsi_id,
                "x": x,
                "y": y,
                "label": label,
                "class_name": "tumor" if label == 1 else "normal",
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        with Image.open(sample["path"]) as img:
            img = img.convert("RGB")
            tensor_img = self.transform(img)

        label_tensor = torch.tensor([sample["label"]], dtype=torch.float32)
        return tensor_img, label_tensor, sample

    def get_class_counts(self) -> Dict[str, int]:
        normal_count = sum(1 for s in self.samples if s["label"] == 0)
        tumor_count = sum(1 for s in self.samples if s["label"] == 1)
        return {"normal": normal_count, "tumor": tumor_count, "total": len(self.samples)}

    def get_wsi_ids(self) -> List[str]:
        return sorted(list(set(s["wsi_id"] for s in self.samples)))


def create_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path,
    train_manifest: Optional[str | Path] = None,
    val_manifest: Optional[str | Path] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, HistopathPatchDataset, HistopathPatchDataset]:
    """
    Creates reproducible DataLoaders for Training and Validation.
    """
    generator = torch.Generator().manual_seed(seed)

    train_dataset = HistopathPatchDataset(
        patch_dir=train_dir,
        manifest_path=train_manifest,
        split_name="train",
        transform=get_train_transforms(),
    )

    val_dataset = HistopathPatchDataset(
        patch_dir=val_dir,
        manifest_path=val_manifest,
        split_name="validation",
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset, val_dataset
