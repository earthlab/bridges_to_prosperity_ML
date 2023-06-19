"""
Downloads any multivariate or optical composites from s3 storage. The root s3 bucket name can be specified but defaults to project configuration value.
Regions, districts, and mgrs tiles can be specified to only get specific location's composites. Downloads are done in parallel by default but can be changed with 
cores parameter.
"""
import argparse
import multiprocessing as mp
import os
from typing import List, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from definitions import REGION_FILE_PATH, S3_COMPOSITE_DIR
from src.api.sentinel2 import initialize_s3_bucket
from src.utilities.config_reader import CONFIG
from file_types import OpticalComposite, MultiVariateComposite, File


def this_download(location_request_info: Tuple[str, int, int, str]) -> None:
    """
    Downloads file from s3 storage given s3 path and local destination path.
    Args:
        location_request_info (tuple):
            location_path (str): Path to composite object in s3 storage relative to the s3 bucket name
            composite_size (int): Size of the composite object in bytes
            destination (str): Local path where the downloaded composite object will be written to
            position (int): Position in the download queue
            bucket_name (str): Name of the s3 bucket that the composite object is in
    """
    for location_path, composite_size, destination, position, bucket_name in location_request_info:
        bucket = initialize_s3_bucket(bucket_name)
        if os.path.exists(destination):
            return 
        with tqdm(total=int(composite_size), unit='B', unit_scale=True, desc=location_path, leave=False,
                  position=int(position)) as pbar:
            bucket.download_file(Key=location_path, Filename=destination,
                                 Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    return


def get_requested_locations(regions: Union[List[str], None], districts: Union[List[str], None]) -> List[str]:
    """
    Resolves the relative s3 locations of the requested objects based off of the requested regions and districts
    Args:
        regions (list): List of names of the regions to download composites from s3 for. Must be in the region_info.yaml file
        districts (list): List of names of the districts to download composites from s3 for. Must be in region_info.yaml file
    Returns:
    requested_locations (list): List of region / district path combinations in s3 
    """
    requested_locations = []
    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    if regions is None:
        regions = []
        for region in region_info:
            regions.append(region)

    for region in regions:
        for district in region_info[region]['districts']:
            if districts is None or district in districts:
                requested_locations.append(os.path.join(S3_COMPOSITE_DIR, region, district))

    return requested_locations


def download_composites(regions: List[str] = None, districts: List[str] = None,
                        s3_bucket_name: str = CONFIG.AWS.BUCKET, cores: int = mp.cpu_count() - 1,
                        bands: List[str] = None, mgrs: List[str] = None) -> None:
    """
    Downloads any multivariate or optical composites from s3 storage. A list of regions, districts, and mgrs tiles can 
    be specified to narrow down the search for composites.
    Args:
        regions (list): List of names of the regions to download composites from s3 for. Must be in the region_info.yaml file
        districts (list): List of names of the districts to download composites from s3 for. Must be in region_info.yaml file
        s3_bucket_name (str): Name of the s3 bucket to look in for the composite objects
        cores (int): Number of cores to use for parallel downloading
        bands (list): List of optical composites to download. If specified only optical composites with these bands will be downloaded. 
                      Thus, all multivariate composites will be excluded. 
        mgrs (list): List of military grid tiles to filter composites download by.  
    """
    s3 = initialize_s3_bucket(s3_bucket_name)
    requested_locations = get_requested_locations(regions, districts)

    location_info = []
    for location in requested_locations:
        for obj in s3.objects.filter(Prefix=location):
            s3_composite = File.create(obj.key)
            if not isinstance(s3_composite, (OpticalComposite, MultiVariateComposite)):
                continue
            if s3_composite:
                if bands is not None:
                    bands.sort()
                    if not isinstance(s3_composite, OpticalComposite):
                        continue
                    if s3_composite.bands != bands:
                        continue
                if mgrs is not None:
                    if s3_composite.mgrs not in mgrs:
                        continue
                destination = s3_composite.archive_path(create_dir=True)
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
    parser.add_argument('--region', '-r', type=str, required=False, help='Name of the region(s) to download composites for. Defualt'
                        'is all regions in regions_info.yaml file')
    parser.add_argument('--districts', '-d', type=str, nargs='+', required=False,
                        help='Name of the district(s) to download the composites for. Defualt is all districts per input region.')
    parser.add_argument('--mgrs', '-m', type=str, nargs='+', required=False,
                        help='Name of the mgrs tile(s) to download for regions and districts. Default is all tiles')
    parser.add_argument('--s3_bucket_name', '-b', required=False, default=CONFIG.AWS.BUCKET, type=str,
                        help='Name of s3 bucket to search for composites in. Default is from project config, which is'
                             f' currently set to {CONFIG.AWS.BUCKET}')
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='Number of cores to use when making tiles in parallel. Default is cpu_count - 1')
    args = parser.parse_args()

    download_composites(region=args.region, districts=args.districts,
                        s3_bucket_name=args.s3_bucket_name, cores=args.cores, mgrs=args.mgrs)
