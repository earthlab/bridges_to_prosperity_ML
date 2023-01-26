from osgeo import osr, gdal
import time 
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
    ul = transform.TransformPoint(*bbox[0])[:2] 
    ll = transform.TransformPoint(*bbox[1])[:2] 
    ur = transform.TransformPoint(*bbox[2])[:2] 
    lr = transform.TransformPoint(*bbox[3])[:2]
    return (ul,ll,ur,lr)

def tiff_to_bbox(tiff:str, debug:bool=False):
    src = gdal.Open(tiff)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)

    tform = get_transform(tiff)

    # Transform the point. You can also create an ogr geometry and use the more generic `point.Transform()`
    # return transform.TransformPoint(ulx, uly)
    ul = (ulx, uly)
    ll = (ulx, lry)
    ur = (lrx, uly)
    lr = (lrx, lry)
    bbox = lat_long_bbox((ul,ll,ur,lr), tform)
    
    if debug: 
        print(f'ul: {bbox[0]}')
        print(f'll: {bbox[1]}')
        print(f'ur: {bbox[2]}')
        print(f'lr: {bbox[3]}')
    return bbox

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