import json
import logging
import sys
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import box
from pathlib import Path
import argparse
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'configs' / 'config.yaml'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--metadata', type=str, default=str(Path('data/processed/metadata')))
    parser.add_argument('--buildings', type=str, default=str(Path('data/processed/buildings/chandigarh_buildings_clean.gpkg')))
    parser.add_argument('--output-masks', type=str, default=str(Path('data/processed/masks')))
    parser.add_argument('--config', type=str, default=str(CONFIG_PATH))
    args = parser.parse_args()

    # STEP 1 — Load buildings once into memory
    buildings_path = Path(args.buildings)
    if not buildings_path.is_absolute():
        buildings_path = Path(__file__).resolve().parent.parent.parent / buildings_path
        
    logging.info("Loading buildings from %s", buildings_path)
    try:
        gdf = gpd.read_file(buildings_path)
    except Exception as e:
        logging.error("Failed to load buildings: %s", e)
        sys.exit(1)
        
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
        
    sindex = gdf.sindex
    logging.info("Total buildings loaded: %d", len(gdf))

    # STEP 2 — Process each tile
    meta_dir = Path(args.metadata)
    if not meta_dir.is_absolute():
        meta_dir = Path(__file__).resolve().parent.parent.parent / meta_dir
        
    masks_dir = Path(args.output_masks)
    if not masks_dir.is_absolute():
        masks_dir = Path(__file__).resolve().parent.parent.parent / masks_dir
    masks_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(meta_dir.glob("*.json"))
    if args.limit is not None:
        json_files = json_files[:args.limit]
        
    total_processed = 0
    total_saved = 0
    empty_masks = 0
    total_coverage = 0.0

    total_files = len(json_files)

    for idx, json_file in enumerate(json_files, 1):
        with open(json_file, 'r') as f:
            meta = json.load(f)
            
        tile_name = json_file.stem
        
        tile_box = box(
            meta['lon_top_left'],       # west
            meta['lat_bottom_right'],   # south
            meta['lon_bottom_right'],   # east
            meta['lat_top_left']        # north
        )
        
        candidates = list(sindex.intersection(tile_box.bounds))
        matches = gdf.iloc[candidates]
        matches = matches[matches.intersects(tile_box)]
        
        if len(matches) == 0:
            logging.warning("No buildings intersect tile %s, skipping", tile_name)
            empty_masks += 1
            total_processed += 1
            continue
            
        # from_bounds expects: west, south, east, north
        # west  = lon_top_left    (leftmost longitude)
        # south = lat_bottom_right (lowest latitude)
        # east  = lon_bottom_right (rightmost longitude)  
        # north = lat_top_left    (highest latitude)
        transform = from_bounds(
            meta['lon_top_left'],       # west
            meta['lat_bottom_right'],   # south
            meta['lon_bottom_right'],   # east
            meta['lat_top_left'],       # north
            512,                        # width
            512                         # height
        )
        
        clipped = matches.copy()
        clipped['geometry'] = matches.geometry.intersection(tile_box)
        
        shapes = [(geom, 1) for geom in clipped.geometry if not geom.is_empty]
        if not shapes:
             logging.warning("Empty mask after clipping for %s", tile_name)
             empty_masks += 1
             total_processed += 1
             continue
             
        mask = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(512, 512),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        output_path = masks_dir / f"{tile_name}.png"
        Image.fromarray(mask, mode='L').save(output_path)
        
        # After saving each mask, save an overlay visualization:
        imagery_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'raw' / 'imagery' / f"{tile_name}.png"
        sat = np.array(Image.open(imagery_path).convert('RGB'))
        mask_arr = mask.copy()
        overlay = sat.copy()
        overlay[mask_arr == 1] = [0, 255, 0]
        blended = (0.6 * sat + 0.4 * overlay).astype(np.uint8)
        viz_dir = masks_dir / 'viz'
        viz_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(blended).save(
            viz_dir / f"{tile_name}_overlay.png"
        )
        
        roof_pixel_pct = 100 * mask.sum() / (512 * 512)
        roof_pixel_pct = 100 * mask.sum() / (512 * 512)
        if roof_pixel_pct == 0.0:
            empty_masks += 1
        else:
            total_saved += 1
            
        total_coverage += roof_pixel_pct
        total_processed += 1

        if idx % 100 == 0:
            avg_cov_so_far = total_coverage / total_processed
            print(f"Masks: {idx}/{total_files} — avg coverage {avg_cov_so_far:.1f}%")

    # STEP 3 — Summary
    avg_cov = total_coverage / total_processed if total_processed > 0 else 0.0
    print(f"Total tiles processed: {total_processed}")
    print(f"Total masks saved: {total_saved}")
    print(f"Empty masks (0% coverage): {empty_masks}")
    print(f"Average roof coverage: {avg_cov:.1f}%")

if __name__ == '__main__':
    main()
