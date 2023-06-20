"""
Given a composite tiff file get the corresponding waterways and governmental boundaries from OSM, 
then write the bands to output OSM file
"""

import osmnx as ox
import rasterio
from rasterio import features
from src.utilities.coords import tiff_to_bbox
from osgeo import gdal
from file_types import OpticalComposite, MultiVariateComposite, OSM
from typing import Union

TAGS_WATER = {
    'water': True,
    'waterway': True
}
TAGS_BOUNDARY = {
    'boundary': "administrative",
    'admin_level': True
}


def create_osm_composite(input_composite_file: Union[OpticalComposite, MultiVariateComposite]) -> OSM:
    """
    Given a composite tiff file get the corresponding waterways and governmental boundaries from OSM, 
    then write the bands to output OSM file
    Args:
        input_composite_file (OpticalComposite, MultiVariateComposite): Composite file object for the file thats bounds will be used
            to download the OSM data
    Returns:
        osm_file (OSM): Output OSM file object 
    """  
    # The bounding box that shapely uses is a set of 4 (x,y) pairs, ox wants ymax, ymin, xmax, xmin
    tl, _, br, _ = tiff_to_bbox(input_composite_file.archive_path)
    bbox = [tl[0], br[0], br[1], tl[1]]
    input_tiff_file = gdal.Open(input_composite_file.archive_path)
    epsg_code = rasterio.CRS.from_wkt(input_tiff_file.GetProjection()).to_epsg()

    # Call to ox api to get geometries for specific tags
    water = ox.geometries.geometries_from_bbox(*bbox, TAGS_WATER)
    boundary = ox.geometries.geometries_from_bbox(*bbox, TAGS_BOUNDARY)
    
    # Remove anything with nan
    water_trim = water.loc[:,['geometry', 'waterway']].dropna()
    boundary_trim = boundary.loc[:,['geometry', 'boundary']].dropna()
    boundary_trim = boundary_trim[boundary_trim['geometry'].geom_type == 'LineString']
   
    # Convert to crs that matches the sentinel2
    water_trim = water_trim.to_crs(f'epsg:{epsg_code}')
    boundary_trim = boundary_trim.to_crs(f'epsg:{epsg_code}')
    
    # Turn this into an iterable that will be used later by features 
    water_shapes = ((geom, 1) for geom in water_trim['geometry'])
    boundary_shapes = ((geom, 1) for geom in boundary_trim['geometry'])

    osm_file = OSM(input_composite_file.region, input_composite_file.district, input_composite_file.mgrs)
    osm_file.create_archive_dir()
    with rasterio.open(input_composite_file.archive_path, 'r') as src:
        meta = src.meta.copy()
        meta['count'] = 2
        meta.update(
            compress='lzw'
        )

        with rasterio.open(osm_file.archive_path, 'w+', **meta) as dst:
            # Rasterize water shapes
            water_arr = features.rasterize(
                shapes=water_shapes,
                fill=0,
                out=dst.read(1),
                transform=src.transform
            )

            dst.write_band(1, water_arr)

            # Rasterize boundary shapes
            boundary_arr = features.rasterize(
                shapes=boundary_shapes,
                fill=0,
                out=dst.read(2),
                transform=src.transform
            )
            dst.write_band(2, boundary_arr)

    return osm_file
