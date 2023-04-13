"""
Creates tiff tile files for the specified composites. By default, this will crawl the data/composites directory and make
tiles for all composites and write them to data/tiles. The input composite directory, output tile directory, and ground
truth directory paths can be overriden so process is only completed for specified region.
"""
import argparse
import multiprocessing as mp
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from definitions import B2P_DIR
from src.utilities.coords import get_bridge_locations
from src.utilities.imaging import tiff_to_tiles


def create_tiles(arg):
    composite_files, tile_dir, bridge_locs, pos = arg
   
    df = []
    for multiband_tiff in tqdm(composite_files, position=0, leave=True):
        df_i = tiff_to_tiles(multiband_tiff, tile_dir, bridge_locs, pos)
        df.append(
            df_i
        )
    df = pd.concat(df, ignore_index=True)
    return df


def composites_to_tiles(in_dir: str, out_dir: str, truth_dir: str, cores: int):
    os.makedirs(out_dir, exist_ok=True)

    bridge_locations = get_bridge_locations(truth_dir)
    composites = glob(os.path.join(in_dir, "**/*multiband.tiff"), recursive=True)
    if args.cores == 1:
        create_tiles(
            (
                composites,
                out_dir,
                bridge_locations,
                1
            )
        )
    else:
        inputs = [
            (
                cs,
                out_dir,
                bridge_locations,
                n
            )
            for n, cs in enumerate(np.array_split(composites, cores))]
        matched_df = process_map(
            create_tiles,
            inputs,
            max_workers=cores
        )
        matched_df = pd.concat(matched_df, ignore_index=True)
        matched_df.to_csv(os.path.join(out_dir, 'matched.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--in_dir', 
        '-i', 
        type=str, 
        required=False, 
        default=os.path.join(B2P_DIR, "data", "composites"),
        help='Path to input directory where s2 path is'
    )
    parser.add_argument(
        '--out_dir', 
        '-o', 
        type=str, 
        required=False, 
        default=os.path.join(B2P_DIR, "data", "tiles"),
        help='Path to directory where output files will be written'
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

    composites_to_tiles(in_dir=args.in_dir, out_dir=args.out_dir, truth_dir=args.truth_dir, cores=args.cores)
