import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

# Internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset import RoofDataset
from src.models.model import SolarRoofModel
from src.models.loss import CompoundLoss, compute_pos_weight

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'configs' / 'config.yaml'

def compute_iou(preds, targets, num_classes=2):
    '''
    Computes per-class Intersection over Union.
    IoU = TP / (TP + FP + FN)
    Returns dict: {'iou_background': float, 
                   'iou_roof': float, 
                   'mean_iou': float}
    preds:   (B, H, W) LongTensor predicted classes
    targets: (B, H, W) LongTensor ground truth
    '''
    results = {}
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        tgt_cls = (targets == cls)
        intersection = (pred_cls & tgt_cls).sum().float()
        union = (pred_cls | tgt_cls).sum().float()
        iou = (intersection / (union + 1e-6)).item()
        name = 'iou_background' if cls == 0 else 'iou_roof'
        results[name] = iou
        ious.append(iou)
    results['mean_iou'] = np.mean(ious)
    return results

def run_epoch(model, loader, criterion, optimizer, device, is_train, sanity_check=False):
    '''
    Runs one full epoch of training or validation.
    Returns dict of metrics for this epoch.
    '''
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    all_ious = {'iou_background': [], 'iou_roof': [], 'mean_iou': []}
    
    context = torch.enable_grad if is_train else torch.no_grad
    
    batch_count = 0
    with context():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, masks)
            
            if is_train:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Compute IoU for this batch
            preds = model.get_prediction(images)
            batch_ious = compute_iou(preds, masks)
            for k in all_ious:
                all_ious[k].append(batch_ious[k])
                
            batch_count += 1
            if sanity_check and batch_count >= 2:
                break
    
    return {
        'loss': total_loss / batch_count,
        'iou_background': np.mean(all_ious['iou_background']),
        'iou_roof': np.mean(all_ious['iou_roof']),
        'mean_iou': np.mean(all_ious['mean_iou'])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(CONFIG_PATH), help='path to config file')
    parser.add_argument('--sanity-check', action='store_true', help='run only 2 batches to verify pipeline')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # STEP 1
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    PROJECT_ROOT = Path(args.config).resolve().parent.parent

    # STEP 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = PROJECT_ROOT / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    train_csv = PROJECT_ROOT / 'data/splits/train.csv'
    val_csv = PROJECT_ROOT / 'data/splits/val.csv'
    imagery_dir = PROJECT_ROOT / 'data/raw/imagery'
    masks_dir = PROJECT_ROOT / 'data/processed/masks'

    # STEP 3
    train_ds = RoofDataset(train_csv, imagery_dir, masks_dir, augment=True)
    val_ds = RoofDataset(val_csv, imagery_dir, masks_dir, augment=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True)
    
    logger.info(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    logger.info(f"Batches per epoch (train): {len(train_loader)}")

    # STEP 4
    model = SolarRoofModel(num_classes=config['model']['num_classes']).to(device)
    
    pos_weight = compute_pos_weight(train_csv).to(device)
    criterion = CompoundLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=pos_weight
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6)

    # STEP 5
    CHECKPOINT_PATH = checkpoint_dir / 'last.pth'
    best_iou = 0.0
    start_epoch = 0
    
    if CHECKPOINT_PATH.exists() and not args.sanity_check:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        logger.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")

    # STEP 6
    if args.sanity_check:
        wandb.init(mode='disabled')
    else:
        wandb.init(
            project='solarroof-ai',
            name='unet-resnet34-binary',
            config={
                'model': 'unet-resnet34',
                'num_classes': config['model']['num_classes'],
                'batch_size': config['training']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'pos_weight': pos_weight.item(),
                'train_tiles': len(train_ds),
                'val_tiles': len(val_ds),
                'gsd': config['geo']['gsd'],
                'city': config['geo']['city']
            },
            resume='allow'
        )

    # STEP 7
    patience_counter = 0
    
    epochs = 1 if args.sanity_check else config['training']['num_epochs']
    start_range = 0 if args.sanity_check else start_epoch

    for epoch in range(start_range, epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = run_epoch(
            model, train_loader, criterion,
            optimizer, device, is_train=True, sanity_check=args.sanity_check)
        
        # Validate
        val_metrics = run_epoch(
            model, val_loader, criterion,
            optimizer, device, is_train=False, sanity_check=args.sanity_check)
        
        if not args.sanity_check:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0] if not args.sanity_check else config['training']['learning_rate']
        
        # Log to console
        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IoU Roof: {val_metrics['iou_roof']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        if args.sanity_check:
            logger.info("Sanity check passed.")
            break
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_metrics['loss'],
            'train/iou_roof': train_metrics['iou_roof'],
            'train/mean_iou': train_metrics['mean_iou'],
            'val/loss': val_metrics['loss'],
            'val/iou_roof': val_metrics['iou_roof'],
            'val/mean_iou': val_metrics['mean_iou'],
            'learning_rate': current_lr
        })
        
        # Save last checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'val_iou_roof': val_metrics['iou_roof'],
            'config': config
        }, checkpoint_dir / 'last.pth')
        
        # Save best checkpoint
        if val_metrics['iou_roof'] > best_iou:
            best_iou = val_metrics['iou_roof']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'config': config
            }, checkpoint_dir / 'best.pth')
            logger.info(f"New best model saved — IoU Roof: {best_iou:.4f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{config['training']['early_stopping_patience']}"
            )
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    if not args.sanity_check:
        wandb.finish()
        logger.info(f"Training complete. Best Val IoU Roof: {best_iou:.4f}")
        logger.info("Best model saved to checkpoints/best.pth")

if __name__ == '__main__':
    main()
