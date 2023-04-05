import os
import time
from glob import glob

import geopandas as gpd
import pandas as pd
from dateutil import parser
from osgeo import osr, gdal
from shapely.geometry import polygon


def get_transform(tiff):
    src = gdal.Open(tiff)
    # Setup the source projection - you can also import from epsg, proj4...
    srcRef = osr.SpatialReference()
    srcRef.ImportFromWkt(src.GetProjection())

    # The target projection
    tgtRef = osr.SpatialReference()
    tgtRef.ImportFromEPSG(4326) # this code gives us lat, long

    # Create the transform - this can be used repeatedly
    return osr.CoordinateTransformation(srcRef, tgtRef)

def lat_long_bbox(bbox, transform):
    tl = transform.TransformPoint(*bbox[0])[:2] 
    tr = transform.TransformPoint(*bbox[1])[:2] 
    br = transform.TransformPoint(*bbox[2])[:2]
    bl = transform.TransformPoint(*bbox[3])[:2] 
    return (tl,tr,br,bl)

def tiff_to_bbox(tiff:str, debug:bool=False):
    src = gdal.Open(tiff)
    lx, xres, xskew, ty, yskew, yres  = src.GetGeoTransform()
    rx = lx + (src.RasterXSize * xres)
    by = ty + (src.RasterYSize * yres)

    tform = get_transform(tiff)

    # Transform the point. You can also create an ogr geometry and use the more generic `point.Transform()`
    # return transform.TransformPoint(ulx, uly)
    tl = (lx, ty)
    bl = (lx, by)
    tr = (rx, ty)
    br = (rx, by)
    bbox = lat_long_bbox((tl,tr,br,bl), tform)
    
    if debug: 
        print(f'ul: {bbox[0]}')
        print(f'll: {bbox[1]}')
        print(f'ur: {bbox[2]}')
        print(f'lr: {bbox[3]}')
    return bbox

def bridge_in_bbox(bbox, bridge_locations):
    p = polygon.Polygon(bbox)
    is_bridge  = False 
    bridge_loc = None
    ix = None
    for i, loc in enumerate(bridge_locations): 
        # check if the current tiff tile contains the current verified bridge location
        if p.contains(loc):
            is_bridge  = True 
            bridge_loc = loc
            ix = i 
            break
    return is_bridge, bridge_loc, ix

def get_bridge_locations(truth_dir):
    csv_files = glob(os.path.join(truth_dir, "*csv"))
    dates = [parser.parse(os.path.basename(csv), fuzzy=True) for csv in csv_files]
    max_ix = dates.index(max(dates))
    print(csv_files[max_ix])
    tDf = pd.read_csv(csv_files[max_ix])
    tDf = tDf.dropna(subset=['Latitude', 'Longitude'])
    bridge_locations = gpd.points_from_xy(tDf['Latitude'], tDf['Longitude'])
    return bridge_locations

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('\tElapsed: %s' % (time.time() - self.tstart))
        print('')