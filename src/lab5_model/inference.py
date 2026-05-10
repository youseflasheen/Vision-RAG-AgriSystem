"""Vision model inference — ResNet-50 + RAG pipeline.

Loads the trained ResNet-50 model, runs inference on a leaf image,
and queries the RAG generator for expert treatment advice.

Dependencies:
    - torch, torchvision
    - Pillow
    - src.lab4_rag.generator (RAG pipeline)
"""

import torch
import json
import os
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

from src.lab4_rag.generator import RAGGenerator


class AgriVisionRAG:
    """End-to-end vision + RAG inference pipeline.

    Loads the fine-tuned ResNet-50, classifies a leaf image,
    and generates RAG-grounded expert advice.

    Args:
        model_path: Path to the saved ResNet-50 .pth weights.
        mapping_path: Path to the class index → name JSON mapping.
    """

    def __init__(
        self,
        model_path: str = "models/resnet50_hybrid_best.pth",
        mapping_path: str = "models/class_mapping.json",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load class mapping
        with open(mapping_path, "r", encoding="utf-8") as file_handle:
            self.class_mapping = json.load(file_handle)

        self.num_classes = len(self.class_mapping)

        # Reconstruct model architecture (must match training exactly)
        self.model = models.resnet50(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_classes),
        )

        # Load trained weights
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # Initialize RAG generator
        self.rag = RAGGenerator()

        # Inference transforms — match training normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict_and_advise(self, image_path: str) -> dict:
        """Run the full prediction + RAG advice pipeline.

        Args:
            image_path: Path to the leaf image file.

        Returns:
            Dictionary with keys:
                - disease_class: Predicted PlantVillage class name
                - confidence: Softmax probability (0.0–1.0)
                - expert_advice: Markdown treatment protocol from RAG
        """
        # 1. Process image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 2. Vision model prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_index = torch.max(probabilities, 0)

        predicted_class = self.class_mapping[str(predicted_index.item())]
        confidence_value = confidence.item()

        # 3. Get RAG advice
        advice = self.rag.generate_advice(predicted_class)

        return {
            "disease_class": predicted_class,
            "confidence": round(confidence_value, 4),
            "expert_advice": advice,
        }