import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
import rasterio as rio
from fiona.crs import from_epsg
import geopandas as gpd
from shapely.geometry import Point

# set PLANET API Key
PLANET_API_KEY = 'f2429704c3ab4d95989e7b889ca4cb5d'

# Import helper modules
import json
import mercantile
import requests

from math import degrees, atan, sinh
import gdal

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