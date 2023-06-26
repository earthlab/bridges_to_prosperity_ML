"""
Creates tiff tile files for the specified region's composites. By default, this will crawl the data/composites directory and make
tiles for all composites and write them to data/tiles. 
"""
import argparse
import yaml

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

from definitions import REGION_FILE_PATH
from src.utilities.coords import get_bridge_locations
from src.utilities.imaging import composite_to_tiles
from src.utilities.files import filter_non_unique_bridge_locations
from file_types import MultiVariateComposite, SingleRegionTileMatch


def create_tiles(args) -> pd.DataFrame:
    """
    Task to run in parallel for creating tiles from composites.
    Args:
        args.composite_files (str): Composite paths to create tiles out of
        args.bridge_locs (gpd.array.GeometryArray): Geometry array containing locations of bridges in ground truth data files
        args.pos (int): pointer for tqdm progress 
        tile_size (int): Height and width to make the tiles into in meters 
    Returns:
        df (pd.DataFrame): Dataframe containing path to each tile, whether it is a bridge or not, and its bounding box
    """
    composite_files, bridge_locs, pos, tile_size = args
    df = []
    for composite_path in tqdm(composite_files, position=0, leave=True):
        df_i = composite_to_tiles(composite_path, bridge_locs, pos, tile_size=tile_size)
        df.append(df_i)
    df = pd.concat(df, ignore_index=True)
    return df


def tiles_from_composites(no_truth: bool, cores: int, region: str, tile_size: int) -> None:
    """
    Creates tiles for the input region's composite files. The height and width of the tiles is
    specified with the tile_size parameter
    Args:
        no_truth (bool): If set, ground truth data will not be searched for the input region. This should
                         be set if there is no ground truth data for the input region
        cores (int): The number of cores to use for parallel creation of tiles
        region (str): Name of the region to create tiles for. Must be in the region_info.yaml file
        tile_size (int): The width and height to make the tiles in meters
    """
    bridge_locations = None if no_truth else get_bridge_locations()

    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)
        districts = list(region_info[region]['districts'].keys())

    dfs = []
    for district in districts:
        composites = MultiVariateComposite.find_files(region, district)

        if cores == 1:
            matched_df = create_tiles((composites, bridge_locations, 1))
        else:
            inputs = [
                (
                    cs,
                    bridge_locations,
                    n,
                    tile_size
                )
                for n, cs in enumerate(np.array_split(composites, cores))]
            matched_df = process_map(
                create_tiles,
                inputs,
                max_workers=cores
            )

        tile_match_file = SingleRegionTileMatch(region, tile_size, district)
        tile_match_file.create_archive_dir()

        unique_bridge_locations = filter_non_unique_bridge_locations(matched_df)
        dfs.append(unique_bridge_locations)
        unique_bridge_locations.to_csv(tile_match_file.archive_path)

    regional_matched_df = pd.concat(dfs, ignore_index=True)
    unique_bridge_locations = filter_non_unique_bridge_locations(regional_matched_df)
    regional_tile_match_file = SingleRegionTileMatch(region, tile_size)
    regional_tile_match_file.create_archive_dir()
    unique_bridge_locations.to_csv(regional_tile_match_file.archive_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', required=True, type=str, help='Name of region to make tiles for. Must be in the '
                                                                  'region_info.yaml file')
    parser.add_argument('--no_truth', action='store_true', help='Must be set if none of the districts in the specified'
                                                                ' region has ground truth data')
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='The number of cores to use to create tiles in parallel. Default is cpu_count - 1')
    parser.add_argument('--tile_size', required=False, type=int, default=300,
                        help='The size to make the tiles in meters. Ex. 300 will split the composite into 300x300m'
                             ' tiles')
    args = parser.parse_args()
    tiles_from_composites(args.no_truth, args.cores, args.region, args.tile_size)
