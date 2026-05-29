import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(mode: str = "train", size: int = 512) -> A.Compose:
    """Return augmentation pipeline. mode='train' adds augmentations."""
    if mode == "train":
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])

class DRDataset(Dataset):
    """PyTorch Dataset for Diabetic Retinopathy images.

    Args:
        csv_path: Path to CSV with columns: image_id, label
        img_dir:  Directory containing preprocessed PNG images
        mode:     'train', 'val', or 'test'
        size:     Image size in pixels

    Example:
        ds = DRDataset("data/processed/train/processed_labels.csv",
                       "data/processed/train", mode="train")
        img, label, img_id = ds[0]
    """
    def __init__(self, csv_path: str, img_dir: str,
                 mode: str = "train", size: int = 512) -> None:
        self.df = pd.read_csv(csv_path)
        if "keep" in self.df.columns:
            self.df = self.df[self.df["keep"] == True].reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = get_transforms(mode, size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["image_id"] + ".png")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return img, label, row["image_id"]

def get_dataloaders(csv_path: str, img_dir: str,
                    val_split: float = 0.15,
                    batch_size: int = 16,
                    num_workers: int = 2):
    """Split dataset and return train + val DataLoaders."""
    import sklearn.model_selection as ms
    df = pd.read_csv(csv_path)
    if "keep" in df.columns:
        df = df[df["keep"] == True].reset_index(drop=True)
    train_df, val_df = ms.train_test_split(
        df, test_size=val_split, stratify=df["label"], random_state=42)
    train_df.to_csv("/tmp/train_split.csv", index=False)
    val_df.to_csv("/tmp/val_split.csv", index=False)
    train_ds = DRDataset("/tmp/train_split.csv", img_dir, mode="train")
    val_ds   = DRDataset("/tmp/val_split.csv",   img_dir, mode="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader
