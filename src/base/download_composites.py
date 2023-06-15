import argparse
import multiprocessing as mp
import os
from typing import List, Tuple

import numpy as np
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from definitions import REGION_FILE_PATH, S3_COMPOSITE_DIR
from src.api.sentinel2 import initialize_s3_bucket
from src.utilities.config_reader import CONFIG
from file_types import OpticalComposite, MultiVariateComposite, File


def this_download(location_request_info: Tuple[str, int, int, str]) -> None:
    for location_path, composite_size, destination, position, bucket_name in location_request_info:
        bucket = initialize_s3_bucket(bucket_name)
        if os.path.isfile(destination):
            return None
        dst_root = os.path.split(destination)[0]
        os.makedirs(dst_root, exist_ok=True)
        with tqdm(total=int(composite_size), unit='B', unit_scale=True, desc=location_path, leave=False,
                  position=int(position)) as pbar:
            bucket.download_file(Key=location_path, Filename=destination,
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


def download_composites(region: str = None, districts: List[str] = None,
                        s3_bucket_name: str = CONFIG.AWS.BUCKET, cores: int = mp.cpu_count() - 1,
                        bands: List[str] = None, mgrs: List[str] = None):
    s3 = initialize_s3_bucket(s3_bucket_name)
    requested_locations = get_requested_locations(region, districts)

    location_info = []
    for location in requested_locations:
        for obj in s3.objects.filter(Prefix=location):
            s3_composite = File.create(obj.key)
            if not isinstance(s3_composite, (OpticalComposite, MultiVariateComposite)):
                continue
            if s3_composite:
                if bands is not None:
                    if s3_composite.bands != bands:
                        continue
                if mgrs is not None:
                    if s3_composite.mgrs not in mgrs:
                        continue
                destination = s3_composite.archive_path()
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
