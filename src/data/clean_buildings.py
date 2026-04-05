import argparse
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import shapely.ops
from shapely.ops import unary_union
from shapely.strtree import STRtree
import scipy.sparse
import scipy.sparse.csgraph
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'configs' / 'config.yaml'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to input CSV')
    parser.add_argument('--output', required=True, help='path to output GPKG')
    parser.add_argument('--config', default=str(CONFIG_PATH), help='path to config.yaml')
    args = parser.parse_args()

    try:
        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        c_conf = config['data_cleaning']
        confidence_threshold = c_conf['confidence_threshold']
        min_area_m2 = c_conf['min_area_m2']
        centroid_merge_distance_m = c_conf['centroid_merge_distance_m']
        large_complex_threshold_m2 = c_conf['large_complex_threshold_m2']

        # STEP 1 - Load data
        logging.info("STEP 1: Loading data from %s", args.input)
        df = pd.read_csv(args.input)
        geoms = shapely.from_wkt(df['geometry'].values, on_invalid='ignore')
        gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
        logging.info("Total rows loaded: %d", len(gdf))

        # STEP 2 - Confidence filter
        logging.info("STEP 2: Applying confidence filter (threshold=%.2f)", confidence_threshold)
        initial_len = len(gdf)
        gdf = gdf[gdf['confidence'] >= confidence_threshold].copy()
        logging.info("Rows removed: %d", initial_len - len(gdf))
        logging.info("Rows remaining: %d", len(gdf))

        # STEP 3 - Area filter
        logging.info("STEP 3: Applying area filter (min_area_m2=%.2f)", min_area_m2)
        initial_len = len(gdf)
        gdf = gdf[gdf['area_in_meters'] >= min_area_m2].copy()
        logging.info("Rows removed: %d", initial_len - len(gdf))
        logging.info("Rows remaining: %d", len(gdf))

        # STEP 4 - Geometry repair
        logging.info("STEP 4: Repairing and filtering geometries")
        initial_len = len(gdf)
        gdf['geometry'] = shapely.make_valid(gdf['geometry'].values)
        
        # Keep only Polygon and MultiPolygon, valid, not empty
        valid_types = ['Polygon', 'MultiPolygon']
        is_poly = gdf['geometry'].geom_type.isin(valid_types)
        is_valid_geom = is_poly & ~gdf['geometry'].is_empty & gdf['geometry'].notna()
        gdf = gdf[is_valid_geom].copy()
        logging.info("Rows removed (invalid/empty/non-polygon): %d", initial_len - len(gdf))

        # STEP 5 - Spatial dissolve using connected components
        logging.info("STEP 5: Spatial dissolve using connected components")
        logging.info("Reprojecting to EPSG:3857")
        gdf_3857 = gdf.to_crs("EPSG:3857").reset_index(drop=True)
        geoms = gdf_3857['geometry'].values

        logging.info("Building STRtree with buffered geometries")
        geoms_orig = gdf_3857['geometry'].values
        
        logging.info("Buffering each centroid by %.2f m", centroid_merge_distance_m)
        centroids = shapely.centroid(geoms_orig)
        buffered_centroids = shapely.buffer(centroids, centroid_merge_distance_m)

        # To merge polygons that physically touch OR have their centroid near the other polygon,
        # we unite the polygon with its buffered centroid. This creates a symmetric relationship
        # so l < r filtering drops exactly half without losing valid unidirectional connections.
        geoms = shapely.union(geoms_orig, buffered_centroids)

        logging.info("Building STRtree on all geometries")
        tree = STRtree(geoms)

        logging.info("Querying STRtree with predicate=None to get candidate pairs")
        l_idx, r_idx = tree.query(geoms, predicate=None)

        logging.info("Filtering pairs where left_idx < right_idx")
        valid_pairs = l_idx < r_idx
        l_sub = l_idx[valid_pairs]
        r_sub = r_idx[valid_pairs]

        logging.info("Testing actual intersection: shapely.intersects(geoms[l_sub], geoms[r_sub])")
        intersects = shapely.intersects(geoms[l_sub], geoms[r_sub])
        
        final_l = l_sub[intersects]
        final_r = r_sub[intersects]

        logging.info("Building scipy sparse adjacency matrix")
        n = len(gdf_3857)
        # to build sparse adjacency, use COO format
        adj = scipy.sparse.coo_matrix((np.ones(len(final_l)), (final_l, final_r)), shape=(n, n))

        logging.info("Running connected_components")
        n_components, labels = scipy.sparse.csgraph.connected_components(adj, directed=False)
        
        logging.info("Dissolving %d components", n_components)
        gdf_3857['cluster'] = labels

        dissolved_rows = []
        for cluster_id, group in gdf_3857.groupby('cluster'):
            merged_geom = unary_union(group['geometry'].values)
            area_in_meters = merged_geom.area
            confidence = group['confidence'].mean()
            best_idx = group['confidence'].idxmax()
            full_plus_code = group.loc[best_idx, 'full_plus_code']
            
            dissolved_rows.append({
                'geometry': merged_geom,
                'area_in_meters': area_in_meters,
                'confidence': confidence,
                'full_plus_code': full_plus_code
            })
            
        dissolved_gdf = gpd.GeoDataFrame(dissolved_rows, crs="EPSG:3857")
        
        logging.info("Polygons before dissolve: %d", len(gdf_3857))
        logging.info("Polygons after dissolve: %d", len(dissolved_gdf))
        pct_reduction = (len(gdf_3857) - len(dissolved_gdf)) / len(gdf_3857) * 100
        logging.info("Reduction: %.2f%%", pct_reduction)

        # STEP 6 - Flag large complexes
        logging.info("STEP 6: Flagging large complexes")
        dissolved_gdf = dissolved_gdf.to_crs("EPSG:4326")
        dissolved_gdf['is_large_complex'] = dissolved_gdf['area_in_meters'] > large_complex_threshold_m2
        num_large = dissolved_gdf['is_large_complex'].sum()
        logging.info("Large complexes flagged: %d", num_large)

        # STEP 7 - Save and validate
        logging.info("STEP 7: Save and validate")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        cols_to_save = ['geometry', 'area_in_meters', 'confidence', 'full_plus_code', 'is_large_complex']
        dissolved_gdf = dissolved_gdf[cols_to_save]
        
        logging.info("Saving to %s", out_path)
        dissolved_gdf.to_file(out_path, driver='GPKG')
        
        logging.info("Running validation checks")
        result = gpd.read_file(out_path)
        assert 40000 < len(result) < 150000, f"Unexpected count: {len(result)}"
        assert result.crs.to_epsg() == 4326, "CRS mismatch"
        geom_types = result.geometry.geom_type.unique()
        assert 'GeometryCollection' not in geom_types, "Bad geometry type found"
        
        print("\nVALIDATION PASSED")
        print(f"Final polygon count: {len(result)}")
        print(f"Mean area: {result['area_in_meters'].mean():.2f} m2")
        print(f"Median area: {result['area_in_meters'].median():.2f} m2")
        print(f"Large complexes flagged: {num_large}")

    except Exception as e:
        logging.error("Pipeline failed", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
