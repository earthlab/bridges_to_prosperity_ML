import os
from typing import List

from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import pandas as pd


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
