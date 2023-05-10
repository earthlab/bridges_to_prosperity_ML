import os
import osmnx as ox
import rasterio
from rasterio import features
from src.utilities.coords import tiff_to_bbox
from src.utilities.imaging import get_utm_epsg, mgrs_to_bbox

from file_types import OpticalComposite
from affine import Affine
from gdal import osr, ogr

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
    # the bounding box that shapely uses is a set of 4 (x,y) pairs, ox wants ymax, ymin, xmax, xmin
    #(tl,tr,br,bl) = tiff_to_bbox(s2_tiff)
    #bbox = [tl[1], br[1], tl[0], br[0]]
    optical_composite = OpticalComposite.create(s2_tiff)
    mgrs_bbox = mgrs_to_bbox(optical_composite.mgrs)
    bbox = [mgrs_bbox[3], mgrs_bbox[1], mgrs_bbox[2], mgrs_bbox[0]]

    epsg_code = get_utm_epsg(mgrs_bbox[3], mgrs_bbox[0])

    src_crs = osr.SpatialReference()
    src_crs.ImportFromEPSG(4326)  # Lat / lon

    dst_crs = osr.SpatialReference()
    dst_crs.ImportFromEPSG(epsg_code)

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(mgrs_bbox[1], mgrs_bbox[0])
    transform = osr.CoordinateTransformation(src_crs, dst_crs)
    point.Transform(transform)

    top_left_lat = point.GetY() + (10980 * 10)
    top_left_lon = point.GetX()

    new_transform = Affine(10, 0, top_left_lon, 0, -10, top_left_lat)

    print(bbox)
    # Call to ox api to get geometries for specific tags
    if debug: print('Getting water from osm')
    water = ox.geometries.geometries_from_bbox(*bbox, TAGS_WATER)
    if debug: print('Getting boundary from osm')
    boundary = ox.geometries.geometries_from_bbox(*bbox, TAGS_BOUNDARY)
    # Remove anything with nan
    water_trim = water.loc[:,['geometry', 'waterway']].dropna()
    boundary_trim = boundary.loc[:,['geometry', 'boundary']].dropna()
    # convert to crs that matches the sentinel2

    water_trim = water_trim.to_crs(f'epsg:{epsg_code}')
    boundary_trim = boundary_trim.to_crs(f'epsg:{epsg_code}')
    # Turn this into an iterable that will be used later by features 
    water_shapes = ((geom,1) for j,geom in enumerate(water_trim['geometry']) )
    boundary_shapes = ((geom,1) for j,geom in enumerate(boundary_trim['geometry']) )

    with rasterio.open(s2_tiff, 'r') as src:
        meta = src.meta.copy()
        meta['count'] = 2
        meta.update(
            compress='lzw'
        )

        with rasterio.open(dst_tiff, 'w+', **meta) as dst:
            # Rasterize water shapes
            if debug: print('Rasterizing water')
            water_arr = features.rasterize(
                shapes=water_shapes,
                fill=0,
                out=dst.read(1),
                transform=new_transform
            )
            print(water_arr)
            if debug: print('Writing water to hypercube tiff')
            dst.write_band(1, water_arr)

            # Rasterize boundary shapes
            boundary_arr = features.rasterize(
                shapes=boundary_shapes,
                fill=0,
                out=dst.read(2),
                transform=new_transform
            )
            dst.write_band(2, boundary_arr)
