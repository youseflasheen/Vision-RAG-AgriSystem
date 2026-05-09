"""GAN training loop for synthetic pest image generation.

Trains the Generator and Discriminator adversarially on the
preprocessed PlantVillage dataset from data/processed/.
Saves model checkpoints to models/ and sample grids to data/synthetic/.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Tuple

from generator import Generator, LATENT_DIM, NUM_CLASSES
from discriminator import Discriminator

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 64
NUM_EPOCHS: int = 50
LEARNING_RATE: float = 0.0002
ADAM_BETA_1: float = 0.5   # DCGAN paper recommends 0.5 (not default 0.9)
ADAM_BETA_2: float = 0.999

# Paths — relative to project root
PROCESSED_DATA_DIR: str = "data/processed"
SYNTHETIC_DATA_DIR: str = "data/synthetic"
MODELS_DIR: str = "models"

# Save a grid of generated images every N epochs
SAMPLE_INTERVAL: int = 5

# Number of fixed noise vectors used for consistent visual progress tracking
NUM_SAMPLE_IMAGES: int = 64

# Device selection
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader(data_dir: str, batch_size: int) -> Tuple[DataLoader, int]:
    """Build a DataLoader from a preprocessed ImageFolder directory.

    Args:
        data_dir: Path to directory with one subdirectory per class.
        batch_size: Mini-batch size.

    Returns:
        Tuple of (DataLoader, number_of_classes).

    Raises:
        FileNotFoundError: If data_dir does not exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Processed data directory not found: {data_dir}. "
            "Run Lab 3 preprocessing first."
        )

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,   # Ensures every batch is exactly batch_size
    )
    return dataloader, len(dataset.classes)


def save_sample_grid(
    generator: Generator,
    fixed_noise: torch.Tensor,
    fixed_labels: torch.Tensor,
    epoch: int,
    output_dir: str,
) -> None:
    """Generate and save a grid of synthetic images for visual inspection.

    Args:
        generator: Trained (or partially trained) Generator.
        fixed_noise: Constant noise tensor reused each epoch for comparison.
        fixed_labels: Constant labels tensor reused each epoch.
        epoch: Current training epoch (used in filename).
        output_dir: Directory where the image grid PNG is saved.
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise, fixed_labels)   # (N, 3, 64, 64)
        # Rescale from [-1, 1] to [0, 1] for saving
        fake_images = (fake_images + 1.0) / 2.0
        grid_path = os.path.join(output_dir, f"synthetic_epoch_{epoch:04d}.png")
        torchvision.utils.save_image(fake_images, grid_path, nrow=int(math.sqrt(NUM_SAMPLE_IMAGES)))
        print(f"  Saved sample grid: {grid_path}")
    generator.train()


def train_gan() -> None:
    """Run the full adversarial training loop.

    Trains Generator and Discriminator for NUM_EPOCHS, logging losses
    each epoch and saving checkpoints at the end.
    """
    os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # --- Data ---
    dataloader, detected_classes = build_dataloader(PROCESSED_DATA_DIR, BATCH_SIZE)
    print(f"Dataset loaded: {detected_classes} classes, {len(dataloader)} batches/epoch")

    # --- Models ---
    generator = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    discriminator = Discriminator(num_classes=NUM_CLASSES).to(DEVICE)

    # --- Loss & Optimisers ---
    adversarial_loss = nn.BCELoss()
    optimizer_generator = optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2)
    )
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2)
    )

    # Fixed noise for consistent progress visualisation
    fixed_noise = torch.randn(NUM_SAMPLE_IMAGES, LATENT_DIM, device=DEVICE)
    fixed_labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLE_IMAGES,), device=DEVICE)

    # Real / fake label values (with slight smoothing for stability)
    real_label_value: float = 0.9   # Label smoothing: use 0.9 instead of 1.0
    fake_label_value: float = 0.0

    # --- Training Loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_discriminator_loss: float = 0.0
        epoch_generator_loss: float = 0.0

        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            current_batch_size = batch_images.size(0)

            real_targets = torch.full((current_batch_size, 1, 1, 1), real_label_value, device=DEVICE)
            fake_targets = torch.full((current_batch_size, 1, 1, 1), fake_label_value, device=DEVICE)

            # ---------------------------------------------------------------
            # Step 1: Train Discriminator
            # Goal: correctly classify real images as real and fake as fake
            # ---------------------------------------------------------------
            optimizer_discriminator.zero_grad()

            # Real images loss
            real_predictions = discriminator(batch_images, batch_labels)
            discriminator_real_loss = adversarial_loss(real_predictions, real_targets)

            # Fake images loss
            noise = torch.randn(current_batch_size, LATENT_DIM, device=DEVICE)
            fake_labels = torch.randint(0, NUM_CLASSES, (current_batch_size,), device=DEVICE)
            fake_images = generator(noise, fake_labels).detach()  # detach: don't backprop into G
            fake_predictions = discriminator(fake_images, fake_labels)
            discriminator_fake_loss = adversarial_loss(fake_predictions, fake_targets)

            discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
            discriminator_loss.backward()
            optimizer_discriminator.step()

            # ---------------------------------------------------------------
            # Step 2: Train Generator
            # Goal: fool discriminator into classifying fake images as real
            # ---------------------------------------------------------------
            optimizer_generator.zero_grad()

            noise = torch.randn(current_batch_size, LATENT_DIM, device=DEVICE)
            fake_labels = torch.randint(0, NUM_CLASSES, (current_batch_size,), device=DEVICE)
            fake_images = generator(noise, fake_labels)
            generator_predictions = discriminator(fake_images, fake_labels)

            # Generator wants discriminator to output "real" (1.0) for its fakes
            generator_loss = adversarial_loss(generator_predictions, real_targets)
            generator_loss.backward()
            optimizer_generator.step()

            epoch_discriminator_loss += discriminator_loss.item()
            epoch_generator_loss += generator_loss.item()

        # --- End of epoch logging ---
        avg_discriminator_loss = epoch_discriminator_loss / len(dataloader)
        avg_generator_loss = epoch_generator_loss / len(dataloader)
        print(
            f"Epoch [{epoch:3d}/{NUM_EPOCHS}] "
            f"D_loss: {avg_discriminator_loss:.4f}  "
            f"G_loss: {avg_generator_loss:.4f}"
        )

        if epoch % SAMPLE_INTERVAL == 0:
            save_sample_grid(generator, fixed_noise, fixed_labels, epoch, SYNTHETIC_DATA_DIR)

    # --- Save final checkpoints ---
    generator_path = os.path.join(MODELS_DIR, "gan_generator_v1.pth")
    discriminator_path = os.path.join(MODELS_DIR, "gan_discriminator_v1.pth")
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print(f"\nTraining complete.")
    print(f"Generator saved  : {generator_path}")
    print(f"Discriminator saved: {discriminator_path}")


if __name__ == "__main__":
    train_gan()