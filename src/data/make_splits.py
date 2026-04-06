import json
import yaml
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.geo_utils import bbox_to_tiles

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    config_path = root_dir / 'configs' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    geo_config = config.get('geo', {})
    zoom = geo_config.get('zoom_level', 19)
    priority_zones = geo_config.get('priority_zones', [])
    
    # Pre-compute tiles for each priority zone for fast lookup
    zone_tiles_map = {}
    for z in priority_zones:
        z_bbox = {
            'north': z['north'],
            'south': z['south'],
            'east': z['east'],
            'west': z['west']
        }
        z_tiles = bbox_to_tiles(z_bbox, zoom)
        zone_tiles_map[z['name']] = set(z_tiles)
        
    meta_files = list(root_dir.glob('data/processed/metadata/*.json'))
    
    records = []
    for mf in meta_files:
        with open(mf, 'r') as f:
            meta = json.load(f)
            
        tile_name = mf.stem
        tx = meta['tile_x']
        ty = meta['tile_y']
        
        zone = "General"
        # Check priority zones logic
        for z_name, z_tiles in zone_tiles_map.items():
            if (tx, ty) in z_tiles:
                zone = z_name
                break
                
        imagery_path = f"data/raw/imagery/{tile_name}.png"
        mask_path = f"data/processed/masks/{tile_name}.png"
        
        records.append({
            'tile_name': tile_name,
            'zone': zone,
            'imagery_path': imagery_path,
            'mask_path': mask_path
        })
        
    df = pd.DataFrame(records)
    
    train_zones = ["Sector 22", "Sector 7-8", "General"]
    val_zones = ["Sector 17", "Sector 35"]
    test_zones = ["Sector 15"]
    
    train_df = df[df['zone'].isin(train_zones)].sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = df[df['zone'].isin(val_zones)].sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = df[df['zone'].isin(test_zones)].sample(frac=1, random_state=42).reset_index(drop=True)
    
    out_dir = root_dir / 'data' / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(out_dir / 'train.csv', index=False)
    val_df.to_csv(out_dir / 'val.csv', index=False)
    test_df.to_csv(out_dir / 'test.csv', index=False)
    
    print(f"Train tiles: {len(train_df)}")
    print(f"Val tiles:   {len(val_df)}")
    print(f"Test tiles:  {len(test_df)}")
    
    train_set = set(train_df['tile_name'])
    val_set = set(val_df['tile_name'])
    test_set = set(test_df['tile_name'])
    
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    print("No data leakage confirmed")

if __name__ == "__main__":
    main()
