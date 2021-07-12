import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
import rasterio as rio
from fiona.crs import from_epsg
import geopandas as gpd
from shapely.geometry import Point, box

# set BING API key
BING_API_KEY = 'Apj6VafMa1ng5CQZKVkTBAGEN8pbGtCJcgB4h2T7VtKVZQrbxqe6orSfUuJ2jXjN'

# Import helper modules
import json
import mercantile
import requests

from math import degrees, atan, sinh
import gdal


def get_aoi_merc_tiles(aoi_bounds, tsize=17):
    ''' tiles will be requested at level tsize-1, so as to generate the center points
    
    params: 
        aoi_bounds (list, or similar): an iterable containing the bounds of the AOI for which we request the tile geometries
        tsize (int): the zoom level we are requesting. 
    
    the result can be overlaid with original aoi_poly to use only intersecting tiles.
    e.g., poly_df_4326[poly_df_4326.intersects(aoi_poly.unary_union)]
    '''
    
    # create geometries for image tiles, should be non-overlapping, but pixel requests will be slightly.
    zm = tsize-1
    
    #pix_size16 = 2.387 #zm=16
    #pix_size17 = 1.196 #zm=17
    
    # return the tile geometries and center points
    all_tiles_aoi = list(mercantile.tiles(*aoi_bounds, zooms=[zm]))
    center_pts = [box(*list(mercantile.bounds(t))).centroid for t in all_tiles_aoi]
    tile_polys = [box(*list(mercantile.bounds(t))) for t in all_tiles_aoi]

    # how far apart are the center_pts?
    pts_df_4326 = gpd.GeoDataFrame({'geometry':center_pts}, crs=from_epsg(4326))
    poly_df_4326 = gpd.GeoDataFrame({'geometry':tile_polys}, crs=from_epsg(4326))

    
    return pts_df_4326, poly_df_4326


def x_to_lon_edges(x, z):
    tile_count = pow(2, z)
    unit = 360 / tile_count
    lon1 = -180 + x * unit
    lon2 = lon1 + unit
    return(lon1, lon2)

def mercatorToLat(mercatorY):
    return(degrees(atan(sinh(mercatorY))))


def y_to_lat_edges(y, z):
    tile_count = pow(2, z)
    unit = 1 / tile_count
    relative_y1 = y * unit
    relative_y2 = relative_y1 + unit
    lat1 = mercatorToLat(np.pi * (1 - 2 * relative_y1))
    lat2 = mercatorToLat(np.pi * (1 - 2 * relative_y2))
    return(lat1, lat2)

def tile_edges(x, y, z):
    lat1, lat2 = y_to_lat_edges(y, z)
    lon1, lon2 = x_to_lon_edges(x, z)
    return[lon1, lat1, lon2, lat2]

def georeference_raster_tile(x, y, z, path, epsg=4326):
    bounds = tile_edges(x, y, z)
    filename, extension = os.path.splitext(path)
    gdal.Translate(filename + '.tif',
                   path,
                   outputSRS=f'EPSG:{epsg}',
                   outputBounds=bounds)
    
def web_merc_tile_edges(x,y,z):
    bounds = tile_edges(x, y, z)
    pt_ul = Point(bounds[0], bounds[1])
    
    pt_ul_df = gpd.GeoDataFrame({'geometry':[pt_ul]}, crs=from_epsg(4326))
    
    return pt_ul_df.to_crs(epsg=3857)