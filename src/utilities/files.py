import os

from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import pandas as pd
import rasterio
from osgeo import gdal, osr


def filter_non_unique_bridge_locations(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes any non-unique bridge locations from the dataframe
    Args:
        matched_df (pd.DataFrame): Dataframe with potentially non-unique bridge locations
    Returns:
        (pd.DataFrame): Original dataframe with any non-unique bridge locations filtered out
    """
    rows_to_delete = []
    bridge_dup = []
    for i, t in enumerate(matched_df['is_bridge']):
        if t:
            if matched_df['bridge_loc'][i] in bridge_dup:
                rows_to_delete.append(i)
            else:
                bridge_dup.append(matched_df['bridge_loc'][i])

    return matched_df.drop(rows_to_delete)

def list_of_vertices(in_list):
    return [[in_list[0], in_list[1]], [in_list[2], in_list[3]], [in_list[4], in_list[5]], [in_list[6], in_list[7]]]


def results_csv_to_shapefile(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    geometry = []
    for bbox in df['bbox']:
        points = list_of_vertices([float(i) for i in bbox.replace('(', '').replace(')', '').split(',')])
        try:
            geometry.append(Polygon(points))
        except:
            print(points)

    gdf = GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
    gdf.to_file(out_path)


def fix_lookup(indir: str):
    fix = {}
    for dir in os.listdir(indir):
        mgrs = dir
        for date_dir in os.listdir(os.path.join(indir, dir)):
            for file in os.listdir(os.path.join(indir, dir, date_dir)):
                if file.endswith('.jp2'):
                    with rasterio.open(os.path.join(indir, dir, date_dir, file), 'r', driver='JP2OpenJPEG') as rf:
                        crs = rf.crs
                        transform = rf.transform
                        fix[mgrs] = {'tran': [transform[2], transform[0], transform[1],
                                              transform[5], transform[3], transform[4]],
                                     'crs': int(str(crs).split(':')[1])}
    return fix
