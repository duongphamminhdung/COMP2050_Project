# [OWN WORK] Dataset class for Kvasir-SEG polyp segmentation dataset
# Handles loading, splitting, and augmentation of images and masks

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class KvasirSEGDataset(Dataset):
    """Kvasir-SEG polyp segmentation dataset.

    Expected folder structure:
        root/
            images/        (*.png or *.jpg)
            masks/         (*.png or *.jpg)

    Images and masks are matched by sorted order (alphabetical filename).
    """

    def __init__(self, root, split=None, transform=None, seed=42):
        self.root = root
        self.transform = transform

        image_dir = os.path.join(root, "images")
        mask_dir = os.path.join(root, "masks")

        # Kvasir-SEG images are in Kvasir-SEG/images/ and masks in Kvasir-SEG/masks/
        if not os.path.isdir(image_dir):
            image_dir = os.path.join(root, "Kvasir-SEG", "images")
            mask_dir = os.path.join(root, "Kvasir-SEG", "masks")

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.*")))

        # Filter to common image extensions
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.image_paths = [p for p in self.image_paths if os.path.splitext(p)[1].lower() in exts]
        self.mask_paths = [p for p in self.mask_paths if os.path.splitext(p)[1].lower() in exts]

        assert len(self.image_paths) == len(self.mask_paths), \
            f"Image count ({len(self.image_paths)}) != mask count ({len(self.mask_paths)})"

        # Create train/val/test split indices
        if split is not None:
            indices = self._get_split_indices(len(self.image_paths), split, seed)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

    def _get_split_indices(self, n, split, seed):
        """Return indices for a specific split: 'train' (70%), 'val' (15%), 'test' (15%)."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n).tolist()

        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if split == "train":
            return indices[:train_end]
        elif split == "val":
            return indices[train_end:val_end]
        elif split == "test":
            return indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # grayscale

        # Resize to 256x256
        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        # Binarize mask (some masks have intermediate values at boundaries)
        mask = (mask > 0.5).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors if not already
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)

        return image, mask


class SimpleTransform:
    """Basic torchvision-based transform (no albumentations dependency).

    Normalizes images to ImageNet mean/std and resizes to target size.
    For training, applies random horizontal/vertical flips.
    """

    def __init__(self, is_train=False):
        transforms_list = []
        if is_train:
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        transforms_list.extend([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(transforms_list)

    def __call__(self, image, mask):
        # image is numpy (H, W, 3), convert to tensor (3, H, W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Apply transforms to image only (mask is already [0,1])
        image = self.transform(image)
        return image, mask


def get_dataloaders(data_root, batch_size=16, seed=42):
    """Create train/val/test dataloaders for Kvasir-SEG.

    Args:
        data_root: Path to Kvasir-SEG root directory.
        batch_size: Batch size for all loaders.
        seed: Random seed for reproducible splits.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders.
    """
    train_dataset = KvasirSEGDataset(data_root, split="train", seed=seed)
    val_dataset = KvasirSEGDataset(data_root, split="val", seed=seed)
    test_dataset = KvasirSEGDataset(data_root, split="test", seed=seed)

    # Apply transforms
    train_transform = SimpleTransform(is_train=True)
    eval_transform = SimpleTransform(is_train=False)

    # Wrap datasets with transforms
    train_dataset.transform = lambda image, mask: train_transform(image, mask)
    val_dataset.transform = lambda image, mask: eval_transform(image, mask)
    test_dataset.transform = lambda image, mask: eval_transform(image, mask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
