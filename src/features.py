import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class ECGFeatureExtractor:
    def __init__(self, target_size=(256, 256)):
        """
        Initializes the feature extractor using a pre-trained PyTorch MobileNetV2.
        """
        self.target_size = target_size

        print("Loading pre-trained PyTorch MobileNetV2 model...")
        # Load the pre-trained model
        weights = models.MobileNet_V2_Weights.DEFAULT
        base_model = models.mobilenet_v2(weights=weights)

        # MobileNetV2's features are extracted before the classifier.
        # We add an AdaptiveAvgPool2d to flatten the 2D feature maps into a 1D vector (1280 features)
        self.model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Set to evaluation mode (turns off training-specific layers like dropout)
        self.model.eval()
        print("Feature Extractor ready.")

    def extract(self, images, batch_size=32):
        """
        Takes the binarized images, converts them to 3-channel PyTorch tensors,
        applies ImageNet normalization, and extracts deep features in batches.
        """
        print(f"Extracting features for {len(images)} images...")

        features_list = []

        # Define normalization tensors once
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Extract features without tracking gradients
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]

                # Check if images are grayscale (3D array) or already RGB (4D array)
                if batch.ndim == 3:
                    batch_rgb = np.stack((batch,) * 3, axis=-1)
                else:
                    batch_rgb = batch

                # Convert to PyTorch Tensor format: (Batch, Channels, Height, Width)
                # PyTorch expects images in [0, 1] range rather than [0, 255]
                tensor_images = torch.tensor(batch_rgb).permute(
                    0, 3, 1, 2).float() / 255.0

                # Apply standard ImageNet normalization
                tensor_images = (tensor_images - mean) / std

                # Pass batch through the model
                batch_features = self.model(tensor_images)
                features_list.append(batch_features.numpy())

                # Print progress every 10 batches (320 images) so you know it's working
                if i % 320 == 0 and i > 0:
                    print(f"  Processed {i}/{len(images)} images...")

        return np.vstack(features_list)
