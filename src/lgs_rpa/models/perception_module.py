import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Tuple, Dict

class ConvBlock(nn.Module):
    """Basic convolution block for U-Net"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    """U-Net encoder (downsampling path)"""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Store skip connections
        skip1 = self.conv1(x)  # 64 channels
        x = self.pool(skip1)
        
        skip2 = self.conv2(x)  # 128 channels
        x = self.pool(skip2)
        
        skip3 = self.conv3(x)  # 256 channels  
        x = self.pool(skip3)
        
        x = self.conv4(x)      # 512 channels (bottleneck)
        
        return x, [skip1, skip2, skip3]

class UNetDecoder(nn.Module):
    """U-Net decoder (upsampling path)"""
    def __init__(self, num_classes: int = 21):  # 21 object classes for household
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(512, 256)  # 512 = 256 + 256 (skip connection)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(256, 128)  # 256 = 128 + 128
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 64)   # 128 = 64 + 64
        
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x, skip_connections):
        skip1, skip2, skip3 = skip_connections
        
        # Upsampling with skip connections
        x = self.upconv1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv1(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv3(x)
        
        return self.output(x)

class PerceptionModule(nn.Module):
    """Complete perception module for LGS-RPA"""
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.encoder = UNetEncoder(in_channels=3)
        self.decoder = UNetDecoder(num_classes=num_classes)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Object class names (household items)
        self.class_names = [
            "background", "table", "chair", "cup", "plate", "knife", "spoon",
            "refrigerator", "microwave", "sink", "stove", "teapot", "book",
            "television", "sofa", "bed", "door", "window", "wall", "floor", "ceiling"
        ]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net"""
        encoded, skip_connections = self.encoder(x)
        segmentation = self.decoder(encoded, skip_connections)
        return segmentation
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for U-Net input"""
        # Resize to 256x256 (U-Net standard input size)
        image = cv2.resize(image, (256, 256))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension and convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        
        return image_tensor.to(self.device)

# Test function
if __name__ == "__main__":
    # Initialize perception module
    perception = PerceptionModule(num_classes=21)
    perception.to(perception.device)
    
    print(f"Perception module initialized on {perception.device}")
    print(f"Number of parameters: {sum(p.numel() for p in perception.parameters())}")
    
    # Test with random image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = perception.preprocess_image(test_image)
    
    print(f"Input image shape: {test_image.shape}")
    print(f"Processed tensor shape: {processed.shape}")
    
    # Forward pass test
    with torch.no_grad():
        output = perception(processed)
        print(f"Segmentation output shape: {output.shape}")
        print(f"Predicted classes: {output.argmax(dim=1).unique()}")