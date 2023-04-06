import argparse
from typing import List, Tuple, Union
import multiprocessing as mp
import os
import random
from glob import glob

import boto3
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from bin.composites_to_tiles import create_tiles
from src.utilities.coords import get_bridge_locations
from definitions import REGION_FILE_PATH, COMPOSITE_DIR, TILE_DIR, TRUTH_DIR
from src.utilities.config_reader import CONFIG
from src.utilities.aws import initialize_s3


CORES = mp.cpu_count() - 1


# TODO: Will B2P always be running the code from an ec2 instance with a correctly configured IAM profile? If not we
#  will have to instantiate S3 sessions in a way they can provide credentials.


def this_download(location_request_info: Tuple[str, int, int, str]) -> None:
    for location_path, composite_size, position, destination in location_request_info:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(CONFIG.AWS.BUCKET)
        if os.path.isfile(destination):
            return None
        dst_root = os.path.split(destination)[0]
        os.makedirs(dst_root, exist_ok=True)

        with tqdm(total=composite_size, unit='B', unit_scale=True, desc=location_path, leave=False,
                  position=position) as pbar:
            bucket.download_file(location_path, destination,
                                 Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    return None


def create_dset_csv(matched_df: pd.DataFrame, ratio: float, tile_dir: str = TILE_DIR) -> None:
    assert 1 > ratio > 0

    # create training sets and validation sets
    # Separate the training and validation into separate files
    b_ix = matched_df.index[matched_df['is_bridge']].tolist()
    nb_ix = matched_df.index[not matched_df['is_bridge']].tolist()
    b_train_ix = random.sample(b_ix, int(round(ratio * len(b_ix))))
    nb_train_ix = random.sample(nb_ix, int(round(ratio * len(nb_ix))))
    b_val_ix = np.setdiff1d(b_ix, b_train_ix)
    nb_val_ix = np.setdiff1d(nb_ix, nb_train_ix)

    train_csv = os.path.join(tile_dir, 'train_df.csv')
    val_csv = os.path.join(tile_dir, 'val_df.csv')
    train_df = pd.concat(
        [
            matched_df.iloc[b_train_ix],
            matched_df.iloc[nb_train_ix]
        ],
        ignore_index=True
    )
    val_df = pd.concat(
        [
            matched_df.iloc[b_val_ix],
            matched_df.iloc[nb_val_ix]
        ],
        ignore_index=True
    )
    train_df.to_csv(train_csv)
    val_df.to_csv(val_csv)
    print(f'Saving to {train_csv} and {val_csv}')


def prepare_optical_inputs(requested_locations: List[str] = None, composites_dir: str = COMPOSITE_DIR,
                           tiles_dir: str = TILE_DIR, s3_bucket_name: str = CONFIG.AWS.BUCKET,
                           bucket_composite_dir: str = 'composites', truth_file_path: Union[None, str] = None,
                           train_to_test_ratio: float = 0.7, cores: int = mp.cpu_count() - 1):

    # Load composites from s3
    s3 = initialize_s3(s3_bucket_name)
    s3_bucket = s3.Bucket(s3_bucket_name)

    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    # If user didn't specify regions then get all the region / districts in the regions.yaml file
    if requested_locations is None:
        requested_locations = []
        for region in region_info:
            for district in region['districts']:
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
            parallel_inputs[i].append((info_tuple[0], info_tuple[1], info_tuple[2], i + 1))
    process_map(
        this_download,
        parallel_inputs,
        max_workers=cores
    )

    # call composites to tiles for each requested location
    print('Making tiles...')
    bridge_locations = get_bridge_locations(truth_file_path)

    all_composites = glob(os.path.join(composites_dir, "**/*multiband.tiff"), recursive=True)
    districts = set([os.path.dirname(composite) for composite in all_composites])
    for district in districts:
        district_composites = [composite for composite in all_composites if os.path.dirname(composite) == district]

        # Resolve the region and district from the composite path to create the tile directory
        composite_path_split = os.path.split(district)
        district_tile_dir = os.path.join(tiles_dir, composite_path_split[-2], composite_path_split[-1])
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
        matched_df = pd.concat(matched_df, ignore_index=True)

        print(f"Creating data set at {district_tile_dir}")
        create_dset_csv(matched_df, train_to_test_ratio, os.path.join(district_tile_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--locations', required=False, default=None, nargs='+', type=str,
                        help="List of locations to pull composites from s3 for. If not specified, composites for all"
                             " locations in data/region_info.yaml will be processed."
                             " Specific districts can be specified d bypassing in district along with region. If "
                             " the entire regions composites are desired thenonly pass in region. Ex. --locations "\
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
    args = parser.parse_args()

    prepare_optical_inputs(requested_locations=args.locations, composites_dir=args.composites_dir,
                           tiles_dir=args.tiles_dir, s3_bucket_name=args.s3_bucket_name,
                           bucket_composite_dir=args.bucket_composite_dir, truth_file_path=args.truth_file_path,
                           train_to_test_ratio=args.train_to_test_ratio, cores=args.cores)
