import argparse
import multiprocessing as mp
import os
import botocore
import random
from glob import glob
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from bin.composites_to_tiles import create_tiles
from definitions import REGION_FILE_PATH, COMPOSITE_DIR, TILE_DIR, TRUTH_DIR, S3_COMPOSITE_DIR
from src.api.sentinel2 import initialize_s3
from src.utilities.config_reader import CONFIG
from src.utilities.coords import get_bridge_locations
from file_types import OpticalComposite

CORES = mp.cpu_count() - 1


def this_download(location_request_info: Tuple[str, int, int, str]) -> None:
    for location_path, composite_size, destination, position, bucket_name in location_request_info:
        s3, client = initialize_s3(bucket_name)
        if not client:
            bucket = s3.Bucket(bucket_name)
        else:
            bucket = s3
        if os.path.isfile(destination):
            return None
        dst_root = os.path.split(destination)[0]
        os.makedirs(dst_root, exist_ok=True)
        with tqdm(total=int(composite_size), unit='B', unit_scale=True, desc=location_path, leave=False,
                  position=int(position)) as pbar:
            bucket.download_file(Key=location_path, Filename=destination, Bucket=bucket_name,
                                 Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    return None


def get_requested_locations(region: str, districts: List[str]) -> List[str]:
    requested_locations = []
    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    if region is None:
        for region in region_info:
            for district in region_info[region]['districts']:
                requested_locations.append(os.path.join(S3_COMPOSITE_DIR, region, district))

    elif districts is None:
        for district in region_info[region]['districts']:
            requested_locations.append(os.path.join(S3_COMPOSITE_DIR, region, district))

    else:
        for district in districts:
            requested_locations.append(os.path.join(S3_COMPOSITE_DIR, region, district))

    return requested_locations


def download_composites(region: str = None, districts: List[str] = None, s3_bucket_name: str = CONFIG.AWS.BUCKET,
                        cores: int = mp.cpu_count() - 1):
    s3, client = initialize_s3(s3_bucket_name)
    if not client:
        s3_bucket = s3.Bucket(s3_bucket_name)
    else:
        s3_bucket = s3

    requested_locations = get_requested_locations(region, districts)

    location_info = []
    for location in requested_locations:
        for obj in s3_bucket.objects.filter(Prefix=location):
            s3_composite = OpticalComposite.create(obj.key)
            if s3_composite:
                destination = s3_composite.archive_path
                location_info.append((obj.key, obj.size, destination))

    parallel_inputs = []
    for i, location in enumerate(np.array_split(location_info, cores)):
        for info_tuple in location:
            parallel_inputs.append([(info_tuple[0], info_tuple[1], info_tuple[2], str(i + 1), s3_bucket_name)])
    process_map(
        this_download,
        parallel_inputs,
        max_workers=cores
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--region',
        '-r',
        type=str,
        required=False,
        help='Name of the composite region (Ex Uganda)'
    )
    parser.add_argument(
        '--districts',
        '-d',
        type=str,
        nargs='+',
        required=False,
        help='Name of the composite district (Ex. Fafan Ibanda)'
    )
    parser.add_argument('--s3_bucket_name', '-b', required=False, default=CONFIG.AWS.BUCKET, type=str,
                        help='Name of s3 bucket to search for composites in. Default is from project config, which is'
                             f' currently set to {CONFIG.AWS.BUCKET}')
    parser.add_argument('--bucket_composite_dir', required=False, default='composites', type=str,
                        help="Name of the composite root directory in the s3 bucket. Default is  'composites'")
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='Number of cores to use when making tiles in parallel. Default is cpu_count - 1')
    args = parser.parse_args()

    download_composites(region=args.region, districts=args.districts,
                        s3_bucket_name=args.s3_bucket_name, bucket_composite_dir=args.bucket_composite_dir,
                        cores=args.cores)
