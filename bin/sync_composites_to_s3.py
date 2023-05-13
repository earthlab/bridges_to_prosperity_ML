"""
Uploads all the composite files found in the specified input directory to s3 storage
"""
import argparse
import glob
import os
from glob import glob
from typing import List

from tqdm import tqdm
import boto3

from definitions import COMPOSITE_DIR, S3_COMPOSITE_DIR
from src.api.sentinel2 import initialize_s3_bucket
from src.utilities.config_reader import CONFIG

from file_types import OpticalComposite, MultiVariateComposite


def sync_s3(root_composite_dir: str, s3_bucket_name: str, s3_directory: str, bands: List[str] = None):
    comp_files = OpticalComposite.find_files(root_composite_dir, bands=bands, recursive=True)
    comp_files += MultiVariateComposite.find_files(root_composite_dir, recursive=True)
    s3 = initialize_s3_bucket(s3_bucket_name)

    for filename in tqdm(comp_files, leave=True, position=0):
        file_size = os.stat(filename).st_size
        key = os.path.join(s3_directory, os.path.basename(filename))
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, leave=False, position=1) as pbar:
            s3.upload_file(
                Filename=filename,
                Key=key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--composite_dir', '-c', type=str, required=False,
                        default=COMPOSITE_DIR,
                        help='Path to the root composites directory that will be crawled for files to upload. Default '
                             'is root project composites directory i.e. all locations will be uploaded. Can also '
                             'specify a specific region and district i.e. data/composites/Uganda/all ')
    parser.add_argument('--s3_bucket_name', '-b', type=str, required=False, default=CONFIG.AWS.BUCKET,
                        help='Name of the s3 bucket to upload the tiles to. Default is bucket specified in config.yaml')
    parser.add_argument('--s3_directory', '-s3', type=str, required=False, default=S3_COMPOSITE_DIR,
                        help='Root directory where composites will be stored in s3')
    parser.add_argument('--bands', type=str, required=False, default=None, nargs='+',
                        help='The bands to upload. Default is None for all bands')
    args = parser.parse_args()

    sync_s3(root_composite_dir=args.composite_dir, s3_bucket_name=args.s3_bucket_name, s3_directory=args.s3_directory,
            bands=args.bands)
