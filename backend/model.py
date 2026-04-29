"""
Deep Learning Model for Deepfake Detection (PyTorch)
Architecture: ResNet50 / EfficientNet backbone + custom head
"""

import os
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


class DeepfakeClassifier:
    """
    Deepfake detection classifier using transfer learning (PyTorch).
    Supports ResNet50 and EfficientNetB0.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        backbone: str = "resnet50",
        weights_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.input_shape = input_shape
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model().to(self.device)

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _build_model(self):
        """Build model architecture"""

        if self.backbone == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = model.fc.in_features

            model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        elif self.backbone == "efficientnetb0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = model.classifier[1].in_features

            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        return model

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded weights from {path}")

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved weights to {path}")

    def predict_frame(self, frame: np.ndarray) -> float:
        """
        Predict probability (0–1) that frame is fake
        """
        try:
            img = self.transform(frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img)

            return float(output.item())

        except Exception:
            return 0.5

    def fine_tune(self, train_loader, val_loader, epochs: int = 10):
        """
        Fine-tune model (unfreeze top layers)
        """
        # Unfreeze last layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in list(self.model.parameters())[-20:]:
            param.requires_grad = True

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-5)
        criterion = nn.BCELoss()

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[Fine-tune] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        self.model.eval()


# =========================
# TRAINING FUNCTION
# =========================

def create_model_for_training(
    data_dir: str,
    backbone: str = "efficientnetb0",
    output_dir: str = "./models",
    batch_size: int = 32,
    epochs: int = 10
) -> DeepfakeClassifier:
    """
    Expected structure:
        data_dir/
            real/
            fake/
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(data_dir, transform=transform)

    # Train/Validation split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    classifier = DeepfakeClassifier(backbone=backbone, device=device)

    model = classifier.model
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    print("Phase 1: Training classification head...")

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    print("Phase 2: Fine-tuning...")
    classifier.fine_tune(train_loader, val_loader, epochs=10)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{backbone}_deepfake.pth")
    classifier.save_weights(save_path)

    return classifier