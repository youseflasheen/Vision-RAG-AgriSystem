"""Discriminator network for the pest image GAN.

Classifies whether a 64x64 RGB leaf image is real or synthetic,
conditioned on the disease class label.
"""

import torch
import torch.nn as nn

# Must match the value in generator.py
NUM_CLASSES: int = 38

# Image dimensions
IMAGE_CHANNELS: int = 3
IMAGE_SIZE: int = 64


class Discriminator(nn.Module):
    """Conditional DCGAN discriminator.

    Takes a 64x64 RGB image and a class label and outputs a
    real/fake probability score.

    Args:
        num_classes: Number of disease classes to condition on.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # Class label is projected to a single-channel spatial map
        # that is concatenated with the input image -> 4 input channels
        self.label_embedding = nn.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)

        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> scalar
        self.conv_blocks = nn.Sequential(
            # Input: (3 + 1, 64, 64) = 4 channels (image + label map)
            nn.Conv2d(IMAGE_CHANNELS + 1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 4x4 -> scalar
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Apply DCGAN weight initialisation (mean=0, std=0.02)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.constant_(module.bias, val=0.0)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Score a batch of images as real or fake.

        Args:
            images: Image tensor of shape (batch_size, 3, 64, 64),
                    normalised to [-1, 1].
            labels: Integer class labels of shape (batch_size,).

        Returns:
            Probability scores of shape (batch_size, 1, 1, 1).
            Values close to 1 indicate "real"; close to 0 indicate "fake".
        """
        batch_size = images.size(0)

        # Build a spatial label map: (B, 1, 64, 64)
        label_map = self.label_embedding(labels)                     # (B, 64*64)
        label_map = label_map.view(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE)  # (B, 1, 64, 64)

        # Concatenate along channel dimension: (B, 4, 64, 64)
        discriminator_input = torch.cat([images, label_map], dim=1)

        return self.conv_blocks(discriminator_input)                 # (B, 1, 1, 1)


if __name__ == "__main__":
    # Quick smoke test — verify output shape
    discriminator = Discriminator()
    test_images = torch.randn(8, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    test_labels = torch.randint(0, NUM_CLASSES, (8,))
    output = discriminator(test_images, test_labels)
    print(f"Discriminator output shape: {output.shape}")  # Expected: (8, 1, 1, 1)
    assert output.shape == (8, 1, 1, 1), "Discriminator output shape mismatch"
    print("Discriminator smoke test passed.")