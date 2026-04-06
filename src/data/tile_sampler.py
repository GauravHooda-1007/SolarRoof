import argparse
import yaml
import json
import math
import time
import logging
import sys
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# Fix Python Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.geo_utils import tile_to_latlon, compute_gsd, bbox_to_tiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'configs' / 'config.yaml'

URL_TEMPLATE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
)

def is_valid_tile(img: Image.Image) -> tuple[bool, float]:
    """
    Returns False if tile is a placeholder 'Map data not available' image.
    Detects by checking if the image is mostly a single grey/beige colour with low variance.
    """
    import numpy as np
    arr = np.array(img.convert('RGB'))
    # Valid satellite tiles have high colour variance
    # Placeholder tiles are flat grey/beige
    channel_stds = arr.reshape(-1, 3).std(axis=0)
    std_mean = float(channel_stds.mean())
    return std_mean > 15.0, std_mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-count', type=int, default=3000, help='Limit number of tiles to download')
    parser.add_argument('--output-imagery', type=str, default=str(Path('data/raw/imagery')), help='Path override for imagery folder')
    parser.add_argument('--config', type=str, default=str(CONFIG_PATH), help='Path override for config')
    args = parser.parse_args()

    # STEP 1: Load config and buildings
    logging.info("STEP 1: Load config and buildings")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error("Failed to load config: %s", e)
        sys.exit(1)

    geo_config = config.get('geo', {})
    zoom = geo_config.get('zoom_level', 20)
    bbox_dict = {
        'north': geo_config.get('bbox_north'),
        'south': geo_config.get('bbox_south'),
        'east': geo_config.get('bbox_east'),
        'west': geo_config.get('bbox_west')
    }

    buildings_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed' / 'buildings' / 'chandigarh_buildings_clean.gpkg'
    logging.info(f"Loading buildings from {buildings_path.as_posix()}")
    try:
        buildings_gdf = gpd.read_file(buildings_path)
    except Exception as e:
        logging.error("Failed to load buildings GeoPackage: %s", e)
        sys.exit(1)

    logging.info("Total buildings loaded: %d", len(buildings_gdf))

    # STEP 2: Find tiles that contain buildings
    logging.info("STEP 2: Find tiles that contain buildings")
    
    # Priority tiles collection
    priority_zones = geo_config.get('priority_zones', [])
    all_tile_records = []
    
    # First get tiles for priority zones
    for zone in priority_zones:
        z_bbox = {
            'north': zone['north'],
            'south': zone['south'],
            'east': zone['east'],
            'west': zone['west']
        }
        z_tiles = bbox_to_tiles(z_bbox, zoom)
        for (x, y) in z_tiles:
            tl_lat, tl_lon = tile_to_latlon(x, y, zoom)
            br_lat, br_lon = tile_to_latlon(x+1, y+1, zoom)
            b = box(tl_lon, br_lat, br_lon, tl_lat)
            all_tile_records.append({
                'x': x, 'y': y,
                'tl_lat': tl_lat, 'tl_lon': tl_lon,
                'br_lat': br_lat, 'br_lon': br_lon,
                'geometry': b,
                'zone': zone['name']
            })
            
    # Then get general tiles
    gen_tiles = bbox_to_tiles(bbox_dict, zoom)
    prioritized_xy = set((r['x'], r['y']) for r in all_tile_records)
    
    for (x, y) in gen_tiles:
        if (x, y) not in prioritized_xy:
            tl_lat, tl_lon = tile_to_latlon(x, y, zoom)
            br_lat, br_lon = tile_to_latlon(x+1, y+1, zoom)
            b = box(tl_lon, br_lat, br_lon, tl_lat)
            all_tile_records.append({
                'x': x, 'y': y,
                'tl_lat': tl_lat, 'tl_lon': tl_lon,
                'br_lat': br_lat, 'br_lon': br_lon,
                'geometry': b,
                'zone': 'General'
            })
            
    tiles_gdf = gpd.GeoDataFrame(all_tile_records, crs="EPSG:4326")
    tiles_gdf = tiles_gdf.drop_duplicates(subset=['x', 'y'])
    
    # Use geopandas spatial join to find tiles that intersect at least one building polygon
    sjoin_gdf = gpd.sjoin(tiles_gdf, buildings_gdf, how='inner', predicate='intersects')
    
    # Count buildings per tile
    tile_counts = sjoin_gdf.groupby(['x', 'y']).size().reset_index(name='building_count')
    
    # Merge back to get tile info
    active_tiles = tile_counts.merge(tiles_gdf.drop(columns=['geometry']), on=['x', 'y'])
    
    # Priority tiles first, then random
    df_priority = active_tiles[active_tiles['zone'] != 'General'].copy()
    df_general = active_tiles[active_tiles['zone'] == 'General'].copy()
    
    # Randomize both within their groups
    df_priority = df_priority.sample(frac=1, random_state=42)
    df_general = df_general.sample(frac=1, random_state=42)
    
    active_tiles = pd.concat([df_priority, df_general]).reset_index(drop=True)
    
    logging.info("Tiles containing buildings: %d", len(active_tiles))

    # Apply target count
    active_tiles = active_tiles.head(args.target_count)
    logging.info("Applying target-count: processing %d tiles", len(active_tiles))

    # STEP 3: Download tiles
    logging.info("STEP 3: Download tiles")
    img_dir = Path(args.output_imagery)
    if not img_dir.is_absolute():
        img_dir = Path(__file__).resolve().parent.parent.parent / img_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    
    meta_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed' / 'metadata'
    meta_dir.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    failed_count = 0

    already_done = {
        f.stem for f in img_dir.glob('*.png')
    }
    logging.info(f"Already downloaded: {len(already_done)} tiles")
    logging.info(f"Remaining: {max(0, args.target_count - len(already_done))} tiles")

    resample_filter = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

    start_time = time.time()
    total_to_process = len(active_tiles)

    for idx, (i, row) in enumerate(active_tiles.iterrows(), 1):
        x = row['x']
        y = row['y']
        tile_name = f"{zoom}_{x}_{y}"
        if tile_name in already_done:
            continue

        if idx > 1 and (idx - 1) % 50 == 0:
            elapsed_min = (time.time() - start_time) / 60.0
            print(f"Progress: {idx-1}/{total_to_process} — zone={row['zone']} — failed={failed_count} — elapsed={elapsed_min:.1f}min")
        
        url = URL_TEMPLATE.format(zoom=zoom, y=y, x=x)
        
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            
            img = Image.open(BytesIO(resp.content))
            
            valid, std_mean = is_valid_tile(img)
            if not valid:
                logging.warning(f"Tile {zoom}_{x}_{y} is a placeholder, skipping")
                failed_count += 1
                time.sleep(0.1)
                continue
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            if img.size != (512, 512):
                img = img.resize((512, 512), resample=resample_filter)
                
            img_path = img_dir / f"{zoom}_{x}_{y}.png"
            img.save(img_path, format="PNG")
            
            # Metadata
            centroid_lat = (row['tl_lat'] + row['br_lat']) / 2.0
            gsd_m_per_px = compute_gsd(centroid_lat, zoom)
            
            b_count = int(row['building_count'])
            meta = {
                "tile_x": int(x),
                "tile_y": int(y),
                "zoom": int(zoom),
                "lat_top_left": float(row['tl_lat']),
                "lon_top_left": float(row['tl_lon']),
                "lat_bottom_right": float(row['br_lat']),
                "lon_bottom_right": float(row['br_lon']),
                "gsd_m_per_px": float(gsd_m_per_px),
                "building_count": b_count
            }
            
            meta_path = meta_dir / f"{zoom}_{x}_{y}.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
                
            downloaded_count += 1
            print(f"Saved tile {zoom}_{x}_{y} — zone={row['zone']} — std={std_mean:.1f} — buildings={b_count}")
            
        except Exception as e:
            logging.warning("Failed to download or process tile %s/%s/%s: %s", zoom, y, x, e)
            failed_count += 1
            
        time.sleep(0.1)

    # STEP 4: Summary
    print(f"Total tiles downloaded: {downloaded_count}")
    print(f"Total tiles failed: {failed_count}")
    print(f"Imagery saved to: {img_dir.as_posix()}")
    print(f"Metadata saved to: {meta_dir.as_posix()}")

if __name__ == '__main__':
    main()
