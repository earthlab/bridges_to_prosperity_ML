import argparse
import multiprocessing as mp
import os
import random
from glob import glob
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from bin.optical.tiles_from_composites import create_tiles
from bin.optical.download_composites import download_composites
from definitions import COMPOSITE_DIR, TILE_DIR, TRUTH_DIR
from src.utilities.config_reader import CONFIG
from src.utilities.coords import get_bridge_locations

CORES = mp.cpu_count() - 1

def prepare_inputs(requested_locations: List[str] = None, composites_dir: str = COMPOSITE_DIR,
                           tiles_dir: str = TILE_DIR, s3_bucket_name: str = CONFIG.AWS.BUCKET,
                           bucket_composite_dir: str = 'composites', truth_file_path: Union[None, str] = None,
                           train_to_test_ratio: float = 0.7, cores: int = mp.cpu_count() - 1, no_truth: bool = False):

    # Load composites from s3
    download_composites(requested_locations, composites_dir, s3_bucket_name, bucket_composite_dir, cores)

    # call composites to tiles for each requested location
    print('Making tiles...')
    bridge_locations = None if no_truth else get_bridge_locations(truth_file_path)

    all_composites = glob(os.path.join(composites_dir, "**/*multiband.tiff"), recursive=True)
    districts = set([os.path.dirname(composite) for composite in all_composites])
    for district in districts:
        district_composites = [composite for composite in all_composites if os.path.dirname(composite) == district]

        # Resolve the region and district from the composite path to create the tile directory
        composite_path_split = os.path.normpath(district).split(os.sep)
        print(tiles_dir, composite_path_split[-2], composite_path_split[-1])
        district_tile_dir = os.path.join(tiles_dir, composite_path_split[-2], composite_path_split[-1])
        print('TILE_DIR', district_tile_dir)
        inputs = [
            (
                cs,
                district_tile_dir,
                bridge_locations,
                n + 1
            )
            for n, cs in enumerate(np.array_split(district_composites, cores))]
        matched_df = process_map(
            create_tiles,
            inputs,
            max_workers=cores
        )

        print(f"Creating data set at {district_tile_dir}")
        create_dset_csv(matched_df, train_to_test_ratio, os.path.join(district_tile_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--locations', required=False, default=None, nargs='+', type=str,
                        help="List of locations to pull composites from s3 for. If not specified, composites for all"
                             " locations in data/region_info.yaml will be processed."
                             " Specific districts can be specified d bypassing in district along with region. If "
                             " the entire regions composites are desired then only pass in region. Ex. --locations "\
                             " Zambia/Chibombo,Uganda will pull in just Chibombo composites and the composites for "\
                             "all districts in Uganda")
    parser.add_argument('--composites_dir', '-c', required=False, default=COMPOSITE_DIR, type=str,
                        help=f'Directory where composites will be written to. Default is {COMPOSITE_DIR}')
    parser.add_argument('--tiles_dir', '-t', required=False, default=TILE_DIR, type=str,
                        help=f'Directory where tiles will be written to. Default is {TILE_DIR}')
    parser.add_argument('--s3_bucket_name', '-b', required=False, default=CONFIG.AWS.BUCKET, type=str,
                        help='Name of s3 bucket to search for composites in. Default is from project config, which is'
                             f' currently set to {CONFIG.AWS.BUCKET}')
    parser.add_argument('--bucket_composite_dir', required=False, default='composites', type=str,
                        help="Name of the composite root directory in the s3 bucket. Default is  'composites'")
    parser.add_argument('--truth_file_path', required=False, type=str,
                        help='Path to the ground truth csv file used to make tiles. If not specified the most recent '
                             f'truth file in {TRUTH_DIR} is used')
    parser.add_argument('--train_to_test_ratio', '-ttr', required=False, type=float, default=0.7,
                        help='Percentage of total data used for training. Default is 0.7')
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='Number of cores to use when making tiles in parallel. Default is cpu_count - 1')
    parser.add_argument('--no_truth', '-nt', action='store_true',
                        help='If set then no truth data will be used to create output dataframe')
    args = parser.parse_args()

    prepare_inputs(requested_locations=args.locations, composites_dir=args.composites_dir,
                           tiles_dir=args.tiles_dir, s3_bucket_name=args.s3_bucket_name,
                           bucket_composite_dir=args.bucket_composite_dir, truth_file_path=args.truth_file_path,
                           train_to_test_ratio=args.train_to_test_ratio, cores=args.cores, no_truth=args.no_truth)
