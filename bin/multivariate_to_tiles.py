"""
Creates tiff tile files for the specified composites. By default, this will crawl the data/composites directory and make
tiles for all composites and write them to data/tiles. The input composite directory, output tile directory, and ground
truth directory paths can be overriden so process is only completed for specified region.
"""
import yaml
import argparse
import multiprocessing as mp
import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from definitions import B2P_DIR, REGION_FILE_PATH, MULTIVARIATE_DIR
from src.utilities.coords import get_bridge_locations
from src.utilities.imaging import composite_to_tiles
from file_types import MultiVariateComposite, TileMatch


def create_tiles(args):
    multivariate_files, bridge_locs, pos = args
    df = []
    for multivariate_path in tqdm(multivariate_files, position=0, leave=True):
        multivariate_file = MultiVariateComposite.create(multivariate_path)
        if multivariate_file is None:
            continue
        df_i = composite_to_tiles(multivariate_file, None, bridge_locs, pos)
        df.append(df_i)
    df = pd.concat(df, ignore_index=True)
    return df


def multivariate_to_tiles(no_truth: bool, cores: int, region: str, districts: List[str] = None, mgrs: str = None):
    bridge_locations = None if no_truth else get_bridge_locations()
    if districts is None:
        with open(REGION_FILE_PATH, 'r') as f:
            region_info = yaml.safe_load(f)
            districts = list(region_info[region]['districts'].keys())

    for district in districts:
        multivariate_dir = os.path.join(MULTIVARIATE_DIR, region, district)
        if mgrs is not None:
            multivariate_dir = os.path.join(multivariate_dir, mgrs)
        composites = MultiVariateComposite.find_files(multivariate_dir, recursive=True)
        if args.cores == 1:
            create_tiles((composites, bridge_locations, 1))
        else:
            inputs = [
                (
                    cs,
                    bridge_locations,
                    n
                )
                for n, cs in enumerate(np.array_split(composites, cores))]
            matched_df = process_map(
                create_tiles,
                inputs,
                max_workers=cores
            )

            tile_match_file = TileMatch(bands=None)
            tile_match_path = tile_match_file.archive_path(region, district)

            matched_df.to_csv(tile_match_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--region',
        '-r',
        type=str,
        required=True,
        help='Name of the composite region (Ex Uganda)'
    )
    parser.add_argument(
        '--districts',
        '-d',
        type=str,
        nargs='+',
        required=False,
        help='Name of the composite district (Ex. Uganda/Ibanda)'
    )
    parser.add_argument(
        '--mgrs',
        '-m',
        type=str,
        required=False,
        help='Name of the mgrs tiles to make tiles for'
    )
    parser.add_argument(
        '--no_truth',
        '-nt',
        action='store_true',
        help='If set then no truth data will be used to create output dataframe'
    )
    parser.add_argument(
        '--truth_dir',
        '-t',
        type=str,
        required=False,
        default=os.path.join(B2P_DIR, "data", "ground_truth"),
        help='Path to directory where csv bridge locations'
    )
    parser.add_argument(
        '--cores',
        '-c',
        type=int,
        default=mp.cpu_count() - 1,
        required=False,
        help='Number of cores to use in parallel for tiling'
    )

    args = parser.parse_args()

    multivariate_to_tiles(no_truth=args.no_truth, cores=args.cores, region=args.region, districts=args.districts,
                          mgrs=args.mgrs)
