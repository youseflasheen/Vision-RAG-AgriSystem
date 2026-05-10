"""Image preprocessing pipeline for the pest detection model.

Defines the transforms applied to raw images before training
and inference. Used by Lab 5 (model training) and Lab 6 (GAN).

Dependencies:
    - torchvision
    - Pillow
    - numpy
"""

import os
from typing import Tuple

import numpy as np
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Standard ImageNet normalization values used by the pretrained ResNet-50
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# Target image size after preprocessing
TARGET_SIZE: int = 224

# GAN image size
GAN_TARGET_SIZE: int = 64


def get_training_transforms(target_size: int = TARGET_SIZE) -> transforms.Compose:
    """Return the augmentation pipeline used during model training.

    Includes random horizontal flip, rotation, and colour jitter
    to improve model generalization.

    Args:
        target_size: Final image size in pixels (square).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(int(target_size * 1.15)),
        transforms.RandomCrop(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_validation_transforms(target_size: int = TARGET_SIZE) -> transforms.Compose:
    """Return the transform pipeline for validation and test data.

    No augmentation — just resize, center crop, and normalize.

    Args:
        target_size: Final image size in pixels (square).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(int(target_size * 1.15)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(target_size: int = TARGET_SIZE) -> transforms.Compose:
    """Return the transform pipeline for single-image inference.

    Same as validation — no augmentation.

    Args:
        target_size: Final image size in pixels (square).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_gan_transforms(target_size: int = GAN_TARGET_SIZE) -> transforms.Compose:
    """Return the transform pipeline for GAN training data.

    Normalizes to [-1, 1] range to match GAN Tanh output.

    Args:
        target_size: GAN output size (default 64).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def verify_image(image_path: str) -> bool:
    """Check whether a file is a valid, non-corrupted image.

    Args:
        image_path: Path to the image file.

    Returns:
        True if the image can be opened and has valid RGB data.
    """
    try:
        with Image.open(image_path) as image:
            image.verify()
        # verify() closes the file — reopen to check mode
        with Image.open(image_path) as image:
            image.convert("RGB")
        return True
    except Exception:
        return False


def compute_dataset_statistics(
    data_dir: str,
) -> Tuple[list[float], list[float]]:
    """Compute per-channel mean and std of images in a directory tree.

    Useful for calculating custom normalization values.

    Args:
        data_dir: Root directory with class subdirectories.

    Returns:
        Tuple of (mean_list, std_list) each with 3 float values (R, G, B).
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            filepath = os.path.join(root, filename)
            try:
                with Image.open(filepath) as image:
                    image_array = np.array(image.convert("RGB"), dtype=np.float64) / 255.0
                    pixel_sum += image_array.sum(axis=(0, 1))
                    pixel_sq_sum += (image_array ** 2).sum(axis=(0, 1))
                    pixel_count += image_array.shape[0] * image_array.shape[1]
            except Exception:
                continue

    if pixel_count == 0:
        return IMAGENET_MEAN, IMAGENET_STD

    mean = (pixel_sum / pixel_count).tolist()
    std = np.sqrt((pixel_sq_sum / pixel_count) - np.array(mean) ** 2).tolist()
    return mean, std


if __name__ == "__main__":
    # Smoke test — verify all transform pipelines instantiate
    train_transform = get_training_transforms()
    val_transform = get_validation_transforms()
    inference_transform = get_inference_transforms()
    gan_transform = get_gan_transforms()

    print(f"Training transforms: {len(train_transform.transforms)} steps")
    print(f"Validation transforms: {len(val_transform.transforms)} steps")
    print(f"Inference transforms: {len(inference_transform.transforms)} steps")
    print(f"GAN transforms: {len(gan_transform.transforms)} steps")
    print("Preprocessor smoke test passed.")
