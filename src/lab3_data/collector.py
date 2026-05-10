"""Data collection module for the pest detection pipeline.

Handles downloading, organizing, and splitting the PlantVillage
dataset into train/validation/test sets. The actual dataset is
large (~3GB) and stored on Google Colab / external storage.

This module provides the utilities and documents the process.

Dependencies:
    - torchvision (for ImageFolder compatibility)
    - Pillow (for image verification)
"""

import os
import shutil
import json
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR: str = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR: str = os.path.join(PROJECT_ROOT, "data", "processed")
KNOWLEDGE_BASE_DIR: str = os.path.join(PROJECT_ROOT, "data", "knowledge_base")

# PlantVillage class names matching the trained ResNet-50 model
# These 21 classes are the hybrid subset used in the project
CLASS_NAMES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Train/validation/test split ratios
TRAIN_RATIO: float = 0.80
VALIDATION_RATIO: float = 0.10
TEST_RATIO: float = 0.10


class DataCollector:
    """Manages dataset organization and splitting for the vision model.

    The raw PlantVillage images are downloaded via Kaggle or Google Colab
    and placed in data/raw/<class_name>/. This class then creates the
    train/val/test split under data/processed/.

    Args:
        raw_dir: Path to directory containing raw class folders.
        processed_dir: Path to output directory for the split dataset.
    """

    def __init__(
        self,
        raw_dir: str = RAW_DATA_DIR,
        processed_dir: str = PROCESSED_DATA_DIR,
    ) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def verify_raw_data(self) -> dict[str, int]:
        """Check which classes are available in the raw data directory.

        Returns:
            Dictionary mapping class name to image count.
            Empty dict if raw_dir does not exist.
        """
        if not os.path.isdir(self.raw_dir):
            print(f"[DataCollector] Raw data directory not found: {self.raw_dir}")
            print("[DataCollector] Dataset is stored on Google Colab.")
            return {}

        class_counts: dict[str, int] = {}
        for class_dir in sorted(os.listdir(self.raw_dir)):
            class_path = os.path.join(self.raw_dir, class_dir)
            if os.path.isdir(class_path):
                image_count = len([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                class_counts[class_dir] = image_count

        return class_counts

    def create_split(
        self,
        train_ratio: float = TRAIN_RATIO,
        validation_ratio: float = VALIDATION_RATIO,
        test_ratio: float = TEST_RATIO,
        random_seed: int = 42,
    ) -> dict[str, int]:
        """Split raw data into train/validation/test sets.

        Creates subdirectories under processed_dir:
            processed/train/<class>/
            processed/val/<class>/
            processed/test/<class>/

        Args:
            train_ratio: Fraction of data for training.
            validation_ratio: Fraction of data for validation.
            test_ratio: Fraction of data for testing.
            random_seed: Seed for reproducible splitting.

        Returns:
            Dictionary with counts per split.

        Raises:
            FileNotFoundError: If raw data directory does not exist.
        """
        import random

        if not os.path.isdir(self.raw_dir):
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_dir}. "
                "Download the PlantVillage dataset first."
            )

        random.seed(random_seed)
        split_counts = {"train": 0, "val": 0, "test": 0}

        for class_name in sorted(os.listdir(self.raw_dir)):
            class_path = os.path.join(self.raw_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            random.shuffle(images)

            total = len(images)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * validation_ratio)

            splits = {
                "train": images[:train_end],
                "val": images[train_end:val_end],
                "test": images[val_end:],
            }

            for split_name, split_images in splits.items():
                output_dir = os.path.join(
                    self.processed_dir, split_name, class_name
                )
                os.makedirs(output_dir, exist_ok=True)
                for image_file in split_images:
                    src = os.path.join(class_path, image_file)
                    dst = os.path.join(output_dir, image_file)
                    shutil.copy2(src, dst)
                split_counts[split_name] += len(split_images)

        print(f"[DataCollector] Split complete: {split_counts}")
        return split_counts

    def get_class_mapping(self) -> dict[int, str]:
        """Return the integer-to-class-name mapping used by the model.

        Returns:
            Dictionary mapping class index to class name string.
        """
        mapping_path = os.path.join(PROJECT_ROOT, "models", "class_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as file:
                return json.load(file)

        # Fallback: generate from CLASS_NAMES
        return {str(i): name for i, name in enumerate(CLASS_NAMES)}


if __name__ == "__main__":
    collector = DataCollector()
    print("Class mapping:", json.dumps(collector.get_class_mapping(), indent=2))
    class_counts = collector.verify_raw_data()
    if class_counts:
        print(f"Found {len(class_counts)} classes, {sum(class_counts.values())} images")
    else:
        print("No raw data found locally (dataset is on Google Colab).")
    print("DataCollector smoke test passed.")
