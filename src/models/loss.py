import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    '''
    Differentiable Dice Loss for binary segmentation.
    Dice = 2*|A∩B| / (|A|+|B|)
    Loss = 1 - Dice
    Smooth factor prevents division by zero.
    '''
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        '''
        logits:  (B, 1, H, W) raw model output
        targets: (B, H, W) long tensor with values 0 or 1
        '''
        probs = torch.sigmoid(logits.squeeze(1))
        targets_float = targets.float()
        
        # Flatten spatial dims for dot product
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets_float.view(targets_float.shape[0], -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()

class CompoundLoss(nn.Module):
    '''
    Weighted combination of BCE Loss and Dice Loss.
    BCE handles per-pixel confidence.
    Dice directly optimises overlap metric.
    Total = bce_weight * BCE + dice_weight * Dice
    '''
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        '''
        pos_weight: tensor weight for positive class
        Use when roof pixels are less frequent than background.
        Compute as: (background_pixels / roof_pixels)
        '''
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        # pos_weight upweights the minority class
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        '''
        logits:  (B, 1, H, W)
        targets: (B, H, W) LongTensor values 0 or 1
        '''
        targets_float = targets.float().unsqueeze(1)
        bce_loss = self.bce(logits, targets_float)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def compute_pos_weight(train_csv_path):
    '''
    Scans train split masks and computes the ratio:
        background_pixels / roof_pixels
    This corrects for class imbalance.
    Returns a scalar tensor for use in BCEWithLogitsLoss.
    
    Only samples 200 random masks for speed.
    '''
    import pandas as pd
    import numpy as np
    from PIL import Image
    from pathlib import Path
    
    df = pd.read_csv(train_csv_path)
    sample = df.sample(min(200, len(df)), random_state=42)
    
    total_bg = 0
    total_roof = 0
    for _, row in sample.iterrows():
        mask = np.array(Image.open(row['mask_path']))
        total_bg += (mask == 0).sum()
        total_roof += (mask == 1).sum()
    
    if total_roof == 0:
        return torch.tensor(1.0)
    
    ratio = total_bg / total_roof
    print(f"Class balance — Background: {total_bg:,} px")
    print(f"Class balance — Roof:       {total_roof:,} px")
    print(f"Pos weight (bg/roof ratio): {ratio:.2f}")
    return torch.tensor(ratio, dtype=torch.float32)

if __name__ == '__main__':
    from pathlib import Path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test loss with random tensors
    logits = torch.randn(4, 1, 512, 512).to(device)
    targets = torch.randint(0, 2, (4, 512, 512)).to(device)
    
    dice = DiceLoss().to(device)
    compound = CompoundLoss().to(device)
    
    d_loss = dice(logits, targets)
    c_loss = compound(logits, targets)
    
    print(f"Dice loss value:     {d_loss.item():.4f}")
    print(f"Compound loss value: {c_loss.item():.4f}")
    assert 0.0 < d_loss.item() < 2.0
    assert 0.0 < c_loss.item() < 2.0
    
    # Test pos_weight computation
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    train_csv = PROJECT_ROOT / 'data/splits/train.csv'
    if train_csv.exists():
        pw = compute_pos_weight(train_csv)
        print(f"Computed pos_weight: {pw.item():.2f}")
        assert pw.item() > 1.0, "Roof should be minority class"
    
    print("All loss tests passed.")
