import torch
import json
import os
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from src.rag.generator import RAGGenerator

class AgriVisionRAG:
    def __init__(self, model_path='models/resnet50_hybrid_best.pth', mapping_path='models/class_mapping.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Class Mapping
        with open(mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.num_classes = len(self.class_mapping)
        
        # Reconstruct Model Architecture (Must match training exactly)
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_classes)
        )
        
        # Load Weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize RAG Generator
        self.rag = RAGGenerator()
        
        # Inference Transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_and_advise(self, image_path):
        # 1. Process Image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 2. Vision Model Prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred_idx = torch.max(probs, 0)
            
        predicted_class = self.class_mapping[str(pred_idx.item())]
        confidence = conf.item()
        
        # 3. Get RAG Advice
        advice = self.rag.generate_advice(predicted_class)
        
        return {
            "disease_class": predicted_class,
            "confidence": round(confidence, 4),
            "expert_advice": advice
        }