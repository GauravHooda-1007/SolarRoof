import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class RoofDataset(Dataset):
    def __init__(self, split_csv, imagery_dir, masks_dir, augment=False):
        self.df = pd.read_csv(split_csv)
        self.imagery_dir = Path(imagery_dir)
        self.masks_dir = Path(masks_dir)
        self.augment = augment
        
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tile_name = row['tile_name']
        
        imagery_path = self.imagery_dir / f"{tile_name}.png"
        mask_path = self.masks_dir / f"{tile_name}.png"
        
        img = Image.open(imagery_path).convert('RGB')
        img = np.array(img)
        
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.uint8)
        
        assert mask.max() <= 1, f"Bad mask values in {tile_name}"
        
        transformed = self.transform(image=img, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask.long(),
            'tile_name': tile_name
        }

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    train_ds = RoofDataset(
        split_csv=PROJECT_ROOT / 'data/splits/train.csv',
        imagery_dir=PROJECT_ROOT / 'data/raw/imagery',
        masks_dir=PROJECT_ROOT / 'data/processed/masks',
        augment=True
    )
    print(f"Train dataset size: {len(train_ds)}")
    
    # Test single item
    sample = train_ds[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Mask shape:  {sample['mask'].shape}")
    print(f"Mask dtype:  {sample['mask'].dtype}")
    print(f"Mask unique values: {sample['mask'].unique()}")
    print(f"Tile name: {sample['tile_name']}")
    
    # Test DataLoader with batch
    loader = DataLoader(train_ds, batch_size=4, 
                       shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch mask shape:  {batch['mask'].shape}")
    assert batch['image'].shape == (4, 3, 512, 512)
    assert batch['mask'].shape == (4, 512, 512)
    print("All dataset tests passed.")
