"""Garment classifier inference (upper / lower / dress)."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CKPT_PATH = MODELS_DIR / "garment_classifier.pt"


class GarmentClassifier:
    def __init__(self):
        self.model = None
        self.classes = None
        self.img_size = 224
        self.transform = None

    def load(self):
        if self.model is not None:
            return
        if not CKPT_PATH.exists():
            raise FileNotFoundError(f"Classifier checkpoint missing: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        self.classes = ckpt["classes"]
        self.img_size = ckpt.get("img_size", 224)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image: Image.Image):
        self.load()
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu()
        idx = int(probs.argmax())
        return {
            "category": self.classes[idx],
            "confidence": round(float(probs[idx]), 3),
            "scores": {c: round(float(p), 3) for c, p in zip(self.classes, probs)},
        }


classifier = GarmentClassifier()
