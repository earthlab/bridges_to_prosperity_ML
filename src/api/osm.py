import os
import osmnx as ox
import rasterio
from rasterio import features
from src.utilities.coords import tiff_to_bbox

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
    bbox = [tl[0], br[0], tl[1], br[1]] 
    # Call to ox api to get geometries for specific tags
    if debug: print('Getting water from osm')
    water = ox.geometries.geometries_from_bbox(*bbox, TAGS_WATER)
    if debug: print('Getting boundary from osm')
    boundary = ox.geometries.geometries_from_bbox(*bbox, TAGS_BOUNDARY)
    # Remove anything with nan
    water_trim = water.loc[:,['geometry', 'waterway']].dropna()
    boundary_trim = boundary.loc[:,['geometry', 'waterway']].dropna()
    # convert to crs that matches the sentinel2
    water_trim = water_trim.to_crs('epsg:32735') 
    boundary_trim = boundary_trim.to_crs('epsg:32735') 
    # Turn this into an iterable that will be used later by features 
    water_shapes = ((geom,1) for j,geom in enumerate(water_trim['geometry']) )
    boundary_shapes = ((geom,1) for j,geom in enumerate(boundary_trim['geometry']) )

    with rasterio.open(s2_tiff, 'r') as src:
        # copy and update the metadata from the input raster for the output
        meta = src.meta.copy()
        d = meta['count'] # should be 3 or 4 to include RGB and maybe IR (near wave)
        meta.update(
            compress='lzw',
            count=d+2
        )
        with rasterio.open(dst_tiff, 'w+', **meta) as dst:
            if debug: print('Copying S2 data to hypecube tiff')
            for i in range(d): 
                dst.write_band(i+1, src.read(i+1))
            # Rasterize water shapes
            if debug: print('Rasterizing water')
            water_arr = features.rasterize(
                shapes=water_shapes, 
                fill=0, 
                out=dst.read(d+1), 
                transform=dst.transform 
            )
            assert (water_arr > 0).any(), 'Water array is empty'
            if debug: print('Writing water to hypecube tiff')
            dst.write_band(d+1, water_arr)

            # Rasterize boundary shapes
            if debug: print('Rasterizing water')
            boundary_arr = features.rasterize(
                shapes=boundary_shapes, 
                fill=0, 
                out=dst.read(d+2), 
                transform=dst.transform 
            )
            # I think it is ok if boundary array is empty because there may be areas where district line are not present for example
            # assert (boundary_arr > 0).any(), 'boundary array is empty'
            if debug: print('Writing water to hypecube tiff')
            dst.write_band(d+2, boundary_arr)
    return None