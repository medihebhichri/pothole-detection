import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config


class PotholeDetector:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

        self.transform = A.Compose([
            A.Resize(Config.INPUT_SIZE[0], Config.INPUT_SIZE[1]),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2()
        ])

    def predict(self, image):
        """
        Predict pothole mask from image
        Args:
            image: numpy array (H, W, 3) in RGB format
        Returns:
            mask: numpy array (H, W) with 0=background, 1=pothole
            overlay: visualization with pothole highlighted
        """
        original_size = image.shape[:2]

        # Transform
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Resize to original size
        mask = cv2.resize(prediction.astype(np.uint8),
                          (original_size[1], original_size[0]),
                          interpolation=cv2.INTER_NEAREST)

        # Create overlay
        overlay = self.create_overlay(image, mask)

        return mask, overlay

    def create_overlay(self, image, mask, color=(255, 0, 0), alpha=0.4):
        """Create visualization with pothole highlighted"""
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
        return overlay