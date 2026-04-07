import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path
import yaml

class SolarRoofModel(nn.Module):
    def __init__(self, num_classes=2, encoder_weights='imagenet'):
        super().__init__()
        self.num_classes = num_classes
        
        # Build U-Net with ResNet34 encoder
        self.unet = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1 if num_classes == 2 else num_classes,
            activation=None  # raw logits — loss handles activation
        )

    def forward(self, x):
        return self.unet(x)

    def get_prediction(self, x):
        '''
        Returns hard class predictions for inference.
        For binary (num_classes=2): applies sigmoid, 
            threshold at 0.5, returns LongTensor
        For multiclass: applies argmax over class dim
        '''
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 2:
                # Binary case: single output channel
                # squeeze from (B,1,H,W) to (B,H,W)
                probs = torch.sigmoid(logits.squeeze(1))
                return (probs > 0.5).long()
            else:
                return torch.argmax(logits, dim=1)

    def get_probabilities(self, x):
        '''
        Returns class probabilities for confidence thresholding.
        Binary: returns sigmoid probability of class 1
        Shape: (B, H, W) with values 0.0 to 1.0
        '''
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 2:
                return torch.sigmoid(logits.squeeze(1))
            else:
                return torch.softmax(logits, dim=1)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SolarRoofModel(num_classes=2).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    dummy = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 1, 512, 512), f"Wrong output shape: {out.shape}"
    
    # Test prediction
    pred = model.get_prediction(dummy)
    print(f"Prediction shape: {pred.shape}")
    print(f"Prediction unique values: {pred.unique()}")
    assert pred.shape == (2, 512, 512)
    
    # Test probabilities
    probs = model.get_probabilities(dummy)
    print(f"Probability range: {probs.min():.3f} to {probs.max():.3f}")
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
    
    print("All model tests passed.")
