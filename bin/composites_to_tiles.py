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

from definitions import B2P_DIR, REGION_FILE_PATH, COMPOSITE_DIR
from src.utilities.coords import get_bridge_locations
from src.utilities.imaging import composite_to_tiles
from file_types import OpticalComposite, TileMatch


def create_tiles(args):
    composite_files, bridge_locs, pos = args
    df = []
    for optical_composite_path in tqdm(composite_files, position=0, leave=True):
        optical_composite_file = OpticalComposite.create(optical_composite_path)
        if optical_composite_file is None:
            continue
        df_i = composite_to_tiles(optical_composite_path, optical_composite_file.bands, bridge_locs, pos)
        df.append(df_i)
    df = pd.concat(df, ignore_index=True)
    return df


def composites_to_tiles(no_truth: bool, cores: int, bands: List[str], region: str, districts: List[str] = None):
    bridge_locations = None if no_truth else get_bridge_locations()
    if districts is None:
        with open(REGION_FILE_PATH, 'r') as f:
            region_info = yaml.safe_load(f)
            districts = list(region_info[region]['districts'].keys())

    for district in districts:
        composite_dir = os.path.join(COMPOSITE_DIR, region, district)
        composites = OpticalComposite.find_files(composite_dir, bands, recursive=True)
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

            tile_match_file = TileMatch(bands)
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
    parser.add_argument(
        '--bands',
        '-b',
        type=str,
        nargs='+',
        required=True
    )

    args = parser.parse_args()

    composites_to_tiles(no_truth=args.no_truth, cores=args.cores, bands=args.bands, region=args.region,
                        districts=args.districts)
