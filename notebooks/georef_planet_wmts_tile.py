import mercantile
import urllib
from math import degrees, atan, sinh
import gdal

# some functions from https://jimmyutterstrom.com/blog/2019/06/05/map-tiles-to-geotiff/
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

def georeference_raster_tile(x, y, z, path):
    bounds = tile_edges(x, y, z)
    filename, extension = os.path.splitext(path)
    gdal.Translate(filename + '.tif',
                   path,
                   outputSRS='EPSG:4326',
                   outputBounds=bounds)
    
    
# get a tile coordinate for lon lat
lon=29.620480
lat=-2.153859
zm=15
tile_mapping = mercantile.tile(lon, lat, zm)

# retrieve the tile at lat lon and zoom=15 and specifying Planet API key
z=tile_mapping.z
x=tile_mapping.x
y=tile_mapping.y
t_url = f'https://tiles.planet.com/basemaps/v1/planet-tiles/global_monthly_2016_01_mosaic/gmap/{z}/{x}/{y}.png?api_key={PLANET_API_KEY}'
resp_tiles = session.get(t_url)

# save tile png
path = f'{x}_{y}_{z}.png'
urllib.request.urlretrieve(t_url, path)

# georeference the tile as a TIF. It will be UINT8
georeference_raster_tile(x,y,z,path)
