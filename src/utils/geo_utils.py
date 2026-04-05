import math
import logging
import pyproj
import shapely.ops
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_gsd(lat: float, zoom: int) -> float:
    """
    Compute Ground Sample Distance in metres per pixel.
    Uses Web Mercator formula.
    Args:
        lat: latitude of tile centroid in decimal degrees
        zoom: map zoom level (integer)
    Returns:
        GSD in metres per pixel
    """
    return (40075016.686 * math.cos(math.radians(lat))) / (2**zoom * 256)

def tile_to_latlon(tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert XYZ tile coordinates to lat/lon of tile's TOP-LEFT corner.
    Args:
        tile_x, tile_y: XYZ tile indices
        zoom: map zoom level
    Returns:
        (latitude, longitude) in decimal degrees
    """
    n = 2.0 ** zoom
    lon = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon

def bbox_to_tiles(bbox: Dict[str, float], zoom: int) -> List[Tuple[int, int]]:
    """
    Return list of all XYZ tile (x, y) pairs covering a bounding box.
    Args:
        bbox: dict with keys north, south, east, west (decimal degrees)
        zoom: map zoom level
    Returns:
        List of (tile_x, tile_y) tuples
    """
    n = 2.0 ** zoom
    
    x_min = math.floor((bbox['west'] + 180.0) / 360.0 * n)
    x_max = math.floor((bbox['east'] + 180.0) / 360.0 * n)
    
    def lat_to_y(lat):
        return math.floor((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
        
    y_min = lat_to_y(bbox['north'])
    y_max = lat_to_y(bbox['south'])
    
    start_x = min(x_min, x_max)
    end_x = max(x_min, x_max)
    start_y = min(y_min, y_max)
    end_y = max(y_min, y_max)
    
    tiles = []
    for x in range(start_x, end_x + 1):
        for y in range(start_y, end_y + 1):
            tiles.append((x, y))
            
    logging.info("Total tile count after computing: %d", len(tiles))
    return tiles

def reproject_polygon(geometry, src_crs: str, dst_crs: str):
    """
    Reproject a single Shapely geometry between coordinate systems.
    Args:
        geometry: Shapely geometry object
        src_crs: source CRS string e.g. 'EPSG:4326'
        dst_crs: destination CRS string e.g. 'EPSG:3857'
    Returns:
        Reprojected Shapely geometry
    """
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return shapely.ops.transform(project, geometry)


if __name__ == '__main__':
    # Test compute_gsd
    gsd = compute_gsd(lat=30.7, zoom=20)
    print(f"GSD at Chandigarh zoom 20: {gsd:.5f} m/px")
    assert 0.12 < gsd < 0.14, f"GSD out of expected range: {gsd}"
    print(f"At 512x512 tile, coverage = {512*gsd:.1f}m x {512*gsd:.1f}m")

    # Chandigarh Sector 22 is approximately at lat=30.73, lon=76.78
    # Compute what tile that falls on at zoom 20
    import math
    zoom = 20
    lat_test, lon_test = 30.73, 76.78
    n = 2 ** zoom
    x_test = int(math.floor((lon_test + 180.0) / 360.0 * n))
    y_test = int(math.floor((1.0 - math.log(
        math.tan(math.radians(lat_test)) + 
        1.0 / math.cos(math.radians(lat_test))
    ) / math.pi) / 2.0 * n))
    print(f"Chandigarh Sector 22 tile at zoom 20: x={x_test}, y={y_test}")
    
    # Now reverse it — tile corner should be close to our test point
    lat_out, lon_out = tile_to_latlon(x_test, y_test, zoom)
    print(f"Tile top-left corner: lat={lat_out:.4f}, lon={lon_out:.4f}")
    assert 30.60 < lat_out < 30.85, f"Latitude out of Chandigarh range: {lat_out}"
    assert 76.60 < lon_out < 77.00, f"Longitude out of Chandigarh range: {lon_out}"
    print("tile_to_latlon test PASSED — coordinates are in Chandigarh")

    # Test bbox_to_tiles
    chandigarh_bbox = {
        'north': 30.7900, 'south': 30.6500,
        'east': 76.8800, 'west': 76.7000
    }
    tiles = bbox_to_tiles(chandigarh_bbox, zoom=20)
    print(f"Total tiles covering Chandigarh at zoom 20: {len(tiles)}")
    assert len(tiles) > 1000, "Too few tiles — check bbox formula"

    # Test reproject_polygon
    from shapely.geometry import Point
    p = Point(76.78, 30.73)
    p_3857 = reproject_polygon(p, 'EPSG:4326', 'EPSG:3857')
    print(f"Reprojected point: {p_3857.x:.1f}, {p_3857.y:.1f}")
    assert p_3857.x > 1000000, "Reprojection failed — x too small for EPSG:3857"

    print("All tests passed.")
