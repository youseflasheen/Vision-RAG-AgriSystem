"""Generator network for the pest image GAN.

Transforms a latent noise vector (concatenated with a class embedding)
into a synthetic 64x64 RGB leaf-disease image.
"""

import torch
import torch.nn as nn


# Latent noise vector size fed into the generator
LATENT_DIM: int = 100

# Number of disease classes (matches PlantVillage subset used in Lab 5)
NUM_CLASSES: int = 38

# Size of the class embedding vector
EMBEDDING_DIM: int = 50


class Generator(nn.Module):
    """Conditional DCGAN generator.

    Takes a noise vector and a class label and produces a
    64x64 RGB synthetic leaf image.

    Args:
        latent_dim: Size of the input noise vector.
        num_classes: Number of disease classes to condition on.
        embedding_dim: Dimensionality of the class embedding.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Input to the first linear layer: noise + class embedding
        input_dim: int = latent_dim + embedding_dim

        # Project and reshape into a spatial feature map: (512, 4, 4)
        self.project = nn.Sequential(
            nn.Linear(input_dim, 512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # Upsample: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Output in [-1, 1] to match normalised real images
        )

        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Apply DCGAN weight initialisation (mean=0, std=0.02)."""
        for module in self.modules():
            if isinstance(module, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.constant_(module.bias, val=0.0)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate a batch of synthetic images.

        Args:
            noise: Random noise tensor of shape (batch_size, latent_dim).
            labels: Integer class labels of shape (batch_size,).

        Returns:
            Synthetic images of shape (batch_size, 3, 64, 64).
        """
        class_embedding = self.label_embedding(labels)          # (B, embedding_dim)
        generator_input = torch.cat([noise, class_embedding], dim=1)  # (B, latent_dim + embedding_dim)

        feature_map = self.project(generator_input)             # (B, 512*4*4)
        feature_map = feature_map.view(-1, 512, 4, 4)           # (B, 512, 4, 4)

        return self.conv_blocks(feature_map)                     # (B, 3, 64, 64)


if __name__ == "__main__":
    # Quick smoke test — verify output shape
    generator = Generator()
    test_noise = torch.randn(8, LATENT_DIM)
    test_labels = torch.randint(0, NUM_CLASSES, (8,))
    output = generator(test_noise, test_labels)
    print(f"Generator output shape: {output.shape}")  # Expected: (8, 3, 64, 64)
    assert output.shape == (8, 3, 64, 64), "Generator output shape mismatch"
    print("Generator smoke test passed.")