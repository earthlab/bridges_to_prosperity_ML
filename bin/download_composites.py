import argparse
import multiprocessing as mp
import os
import random
from glob import glob
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from bin.composites_to_tiles import create_tiles
from definitions import REGION_FILE_PATH, COMPOSITE_DIR, TILE_DIR, TRUTH_DIR
from src.utilities.aws import initialize_s3
from src.utilities.config_reader import CONFIG
from src.utilities.coords import get_bridge_locations

CORES = mp.cpu_count() - 1


def this_download(location_request_info: Tuple[str, int, int, str]) -> None:
    for location_path, composite_size, destination, position in location_request_info:
        s3 = initialize_s3(CONFIG.AWS.BUCKET)
        bucket = s3.Bucket(CONFIG.AWS.BUCKET)
        if os.path.isfile(destination):
            return None
        dst_root = os.path.split(destination)[0]
        os.makedirs(dst_root, exist_ok=True)
        print('Downloading')
        with tqdm(total=int(composite_size), unit='B', unit_scale=True, desc=location_path, leave=False,
                  position=int(position)) as pbar:
            print(destination)
            bucket.download_file(location_path, destination,
                                 Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    return None


def download_composites(requested_locations: List[str] = None, composites_dir: str = COMPOSITE_DIR,
                        s3_bucket_name: str = CONFIG.AWS.BUCKET, bucket_composite_dir: str = 'composites',
                        cores: int = mp.cpu_count() - 1):
    s3 = initialize_s3(s3_bucket_name)
    s3_bucket = s3.Bucket(s3_bucket_name)

    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    # If user didn't specify regions then get all the region / districts in the regions.yaml file
    if requested_locations is None:
        requested_locations = []
        for region in region_info:
            for district in region_info[region]['districts']:
                requested_locations.append(os.path.join(bucket_composite_dir, region, district))

    location_info = []
    for location in requested_locations:
        for obj in s3_bucket.objects.filter(Prefix=location):
            if obj.key.endswith('.tiff'):
                destination = os.path.join(composites_dir, obj.key.strip(f'{bucket_composite_dir}/'))
                location_info.append((obj.key, obj.size, destination))

    parallel_inputs = []
    for i, location in enumerate(np.array_split(location_info, cores)):
        parallel_inputs.append([])
        for info_tuple in location:
            print(info_tuple)
            parallel_inputs[i].append((info_tuple[0], info_tuple[1], info_tuple[2], str(i + 1)))
    process_map(
        this_download,
        parallel_inputs,
        max_workers=cores
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--locations', required=False, default=None, nargs='+', type=str,
                        help="List of locations to pull composites from s3 for. If not specified, composites for all"
                             " locations in data/region_info.yaml will be processed."
                             " Specific districts can be specified d bypassing in district along with region. If "
                             " the entire regions composites are desired then only pass in region. Ex. --locations " \
                             " Zambia/Chibombo,Uganda will pull in just Chibombo composites and the composites for " \
                             "all districts in Uganda")
    parser.add_argument('--composites_dir', '-c', required=False, default=COMPOSITE_DIR, type=str,
                        help=f'Directory where composites will be written to. Default is {COMPOSITE_DIR}')
    parser.add_argument('--s3_bucket_name', '-b', required=False, default=CONFIG.AWS.BUCKET, type=str,
                        help='Name of s3 bucket to search for composites in. Default is from project config, which is'
                             f' currently set to {CONFIG.AWS.BUCKET}')
    parser.add_argument('--bucket_composite_dir', required=False, default='composites', type=str,
                        help="Name of the composite root directory in the s3 bucket. Default is  'composites'")
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='Number of cores to use when making tiles in parallel. Default is cpu_count - 1')
    args = parser.parse_args()

    download_composites(requested_locations=args.locations, composites_dir=args.composites_dir,
                        s3_bucket_name=args.s3_bucket_name, bucket_composite_dir=args.bucket_composite_dir,
                        cores=args.cores)