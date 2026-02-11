"""Pitcher classification using fine-tuned EfficientNet-B0.

Classifies person crops as 'pitcher' or 'not_pitcher'. Used to replace
spatial heuristics in the pitcher detection pipeline.

Follows the same pattern as camera_filter.py (CameraAngleClassifier).
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms, models

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/pitcher_classifier/best.pt"

# Inference transform — must match training
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PitcherClassifier:
    """EfficientNet-B0 binary classifier for pitcher identification."""

    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None
        self._initialized = False

    def initialize(self) -> None:
        """Load the trained model."""
        if self._initialized:
            return

        logger.info(f"Loading pitcher classifier from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.class_names = checkpoint["class_names"]
        n_classes = len(self.class_names)

        self.model = models.efficientnet_b0()
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, n_classes),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded pitcher classifier (classes: {self.class_names}, "
                     f"test_acc: {checkpoint.get('test_acc', '?')}, device: {self.device})")
        self._initialized = True

    def classify_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Classify a single BGR person crop.

        Args:
            crop_bgr: OpenCV BGR crop (numpy array)

        Returns:
            (label, confidence) — label is 'pitcher' or 'not_pitcher',
            confidence is the softmax probability of the predicted class.
        """
        if not self._initialized:
            self.initialize()

        # BGR -> RGB, apply transform
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = _transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0]

        pred_idx = probs.argmax().item()
        label = self.class_names[pred_idx]
        confidence = probs[pred_idx].item()

        return label, confidence

    def classify_crops_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Classify a batch of BGR person crops.

        Args:
            crops_bgr: List of OpenCV BGR crops

        Returns:
            List of (label, confidence) tuples
        """
        if not self._initialized:
            self.initialize()

        if not crops_bgr:
            return []

        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(_transform(rgb))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)

        results = []
        for i in range(len(crops_bgr)):
            pred_idx = probs[i].argmax().item()
            label = self.class_names[pred_idx]
            confidence = probs[i][pred_idx].item()
            results.append((label, confidence))

        return results
