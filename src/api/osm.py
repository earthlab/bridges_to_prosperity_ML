import os
import osmnx as ox
import rasterio
from rasterio import features
from src.utilities.coords import tiff_to_bbox
from src.utilities.imaging import get_utm_epsg, mgrs_to_bbox

from file_types import OpticalComposite

TAGS_WATER = {
    'water': True,
    'waterway': True
}
TAGS_BOUNDARY = {
    'boundary': True
}

"""
    Given a tiff from from sentinel2 get the corresponding waterways and governmental boundaries from OSM, then combine the two into a hypercube
"""
def getOsm(s2_tiff: str, dst_tiff: str, debug: bool = False):
    assert os.path.isfile(s2_tiff), f'{s2_tiff} DNE'
    # the bounding box that shapely uses is a set of 4 (x,y) pairs, ox wants xmin, xmax, ymin, ymax
    (tl,tr,br,bl) = tiff_to_bbox(s2_tiff)
    print((tl,tr,br,bl))
    mgrs_bbox = mgrs_to_bbox(OpticalComposite.create(s2_tiff).mgrs)
    bbox = [tl[0], br[0], tl[1], br[1]]
    bbox = [mgrs_bbox[3], mgrs_bbox[0], mgrs_bbox[1], mgrs_bbox[2]]
    print(bbox)
    print('bbox', bbox)
    # Call to ox api to get geometries for specific tags
    if debug: print('Getting water from osm')
    water = ox.geometries.geometries_from_bbox(*bbox, TAGS_WATER)
    if debug: print('Getting boundary from osm')
    boundary = ox.geometries.geometries_from_bbox(*bbox, TAGS_BOUNDARY)
    # Remove anything with nan
    water_trim = water.loc[:,['geometry', 'waterway']].dropna()
    boundary_trim = boundary.loc[:,['geometry', 'boundary']].dropna()
    # convert to crs that matches the sentinel2
    epsg_code = get_utm_epsg(tl[0], tl[1])
    print(mgrs_to_bbox(OpticalComposite.create(s2_tiff).mgrs))
    water_trim = water_trim.to_crs(f'epsg:{epsg_code}')
    boundary_trim = boundary_trim.to_crs(f'epsg:{epsg_code}')
    # Turn this into an iterable that will be used later by features 
    water_shapes = ((geom,1) for j,geom in enumerate(water_trim['geometry']) )
    boundary_shapes = ((geom,1) for j,geom in enumerate(boundary_trim['geometry']) )

    with rasterio.open(s2_tiff, 'r') as src:
        meta = src.meta.copy()
        meta['count'] = 2

        with rasterio.open(dst_tiff, 'w+', **meta) as dst:
            # Rasterize water shapes
            if debug: print('Rasterizing water')
            water_arr = features.rasterize(
                shapes=water_shapes,
                fill=0,
                out=dst.read(1),
                transform=src.transform
            )
            if debug: print('Writing water to hypercube tiff')
            dst.write_band(1, water_arr)

            # Rasterize boundary shapes
            boundary_arr = features.rasterize(
                shapes=boundary_shapes,
                fill=0,
                out=dst.read(2),
                transform=src.transform
            )
            dst.write_band(2, boundary_arr)
