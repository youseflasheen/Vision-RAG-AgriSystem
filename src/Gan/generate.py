"""Synthetic image generation using a trained GAN generator.

Loads a saved Generator checkpoint and produces N synthetic leaf-disease
images per class, saving them to data/synthetic/<class_name>/.
These images are available for Lab 5 retraining and Lab 3 augmentation.
"""

import os
import argparse
import torch
import torchvision.transforms.functional as transforms_functional
from PIL import Image

from generator import Generator, LATENT_DIM, NUM_CLASSES

# Default output location — mirrors the structure Lab 3 uses
DEFAULT_OUTPUT_DIR: str = "data/synthetic"
DEFAULT_MODEL_PATH: str = "models/gan_generator_v1.pth"
DEFAULT_IMAGES_PER_CLASS: int = 50

# PlantVillage class names (38 classes) — must match the order
# used when training the ResNet-50 model in Lab 5
CLASS_NAMES: list[str] = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def load_generator(model_path: str, device: torch.device) -> Generator:
    """Load a trained Generator from a checkpoint file.

    Args:
        model_path: Path to the saved .pth state dict file.
        device: Torch device to load the model onto.

    Returns:
        Generator model in eval mode.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Generator checkpoint not found: {model_path}. "
            "Run trainer.py first to train the GAN."
        )

    generator = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator


def generate_synthetic_images(
    generator: Generator,
    class_index: int,
    num_images: int,
    output_class_dir: str,
    device: torch.device,
) -> None:
    """Generate and save synthetic images for one disease class.

    Args:
        generator: Trained Generator in eval mode.
        class_index: Integer class index (0 to NUM_CLASSES - 1).
        num_images: Number of synthetic images to produce.
        output_class_dir: Directory path where images will be saved.
        device: Torch device.
    """
    os.makedirs(output_class_dir, exist_ok=True)

    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, device=device)
        labels = torch.full((num_images,), class_index, dtype=torch.long, device=device)
        fake_images = generator(noise, labels)          # (N, 3, 64, 64) in [-1, 1]

        # Rescale from [-1, 1] to [0, 255] for saving as PNG
        fake_images = ((fake_images + 1.0) / 2.0).clamp(0.0, 1.0)
        fake_images = (fake_images * 255).byte().cpu()

        for image_index in range(num_images):
            # Convert tensor (C, H, W) to PIL Image (H, W, C)
            image_tensor = fake_images[image_index].permute(1, 2, 0).numpy()
            pil_image = Image.fromarray(image_tensor)
            filename = os.path.join(output_class_dir, f"synthetic_{image_index:04d}.png")
            pil_image.save(filename)


def run_generation(
    model_path: str,
    output_dir: str,
    images_per_class: int,
    target_classes: list[int] | None = None,
) -> None:
    """Generate synthetic images for all (or selected) disease classes.

    Args:
        model_path: Path to the saved Generator checkpoint.
        output_dir: Root directory for synthetic output images.
        images_per_class: Number of images to generate per class.
        target_classes: Optional list of class indices to generate.
                        If None, generates for all NUM_CLASSES classes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = load_generator(model_path, device)
    print(f"Generator loaded from: {model_path}")

    classes_to_generate = target_classes if target_classes is not None else list(range(NUM_CLASSES))

    for class_index in classes_to_generate:
        class_name = CLASS_NAMES[class_index]
        output_class_dir = os.path.join(output_dir, class_name)
        print(f"  Generating {images_per_class} images for class: {class_name}")
        generate_synthetic_images(generator, class_index, images_per_class, output_class_dir, device)

    total_generated = len(classes_to_generate) * images_per_class
    print(f"\nDone. {total_generated} synthetic images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic pest images using a trained GAN.")
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to the saved Generator checkpoint (.pth file)."
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Root directory where synthetic images will be saved."
    )
    parser.add_argument(
        "--images_per_class", type=int, default=DEFAULT_IMAGES_PER_CLASS,
        help="Number of synthetic images to generate per disease class."
    )
    parser.add_argument(
        "--classes", type=int, nargs="*", default=None,
        help="Specific class indices to generate (default: all 38 classes)."
    )

    args = parser.parse_args()
    run_generation(
        model_path=args.model_path,
        output_dir=args.output_dir,
        images_per_class=args.images_per_class,
        target_classes=args.classes,
    )