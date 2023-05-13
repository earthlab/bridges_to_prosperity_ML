import os
from typing import List

from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import pandas as pd
from src.utilities.imaging import fix_crs
from file_types import OpticalComposite, TileMatch
import shutil
from src.api.sentinel2 import initialize_s3_bucket


def find_directories(root_dir: str, file_extension: str) -> List[str]:
    """
    Recursively searches for directories containing files with the given file extension in the given root directory and
     its subdirectories.
    Args:
        root_dir (str): The root directory to start searching from.
        file_extension (str): The file extension to search for (e.g., ".txt").
    Returns:
         A list of absolute paths to the directories containing files with the given file extension.
    """
    found_directories = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(file_extension):
                found_directories.append(os.path.abspath(root))
                break
    return found_directories


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


def fix_file_name_and_crs(in_dir: str, region: str, district: str):
    for file in os.listdir(in_dir):
        if file.endswith('multiband.tiff'):
            mgrs = file[:5]
            optical_composite = OpticalComposite(region=region, district=district, military_grid=mgrs,
                                                 bands=['B02', 'B03', 'B04'])
            shutil.copy(os.path.join(in_dir, file), optical_composite.archive_path)

    fix_crs(in_dir)
    bucket = initialize_s3_bucket()
    for file in os.listdir(in_dir):
        optical_composite = OpticalComposite.create(file)
        if optical_composite is not None:
            bucket.upload_file(Key=os.path.join('composites', optical_composite.region, optical_composite.district,
                                                optical_composite.name),
                               Filename=os.path.join(in_dir, file))


def filter_and_write_csv(input_file, output_file, column_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Filter the DataFrame to remove duplicate values in the specified column
    filtered_df = df.drop_duplicates(subset=column_name)

    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)


def concat_geoloc(indir):
    dfs = []
    for mgrs_dir in os.listdir(indir):
        df = pd.read_csv(os.path.join(indir, mgrs_dir, 'multivariate_geoloc.csv'))
        dfs.append(dfs)
    dfss = pd.concat(dfs, ignore_index=True)
    filtered_df = dfss.drop_duplicates(subset='bbox')
    dfss.to_csv(os.path.join(indir, TileMatch().name))
