"""
Uploads all the composite files found in the specified input directory to s3 storage
"""
import argparse

from definitions import COMPOSITE_DIR, S3_COMPOSITE_DIR
from src.utilities.config_reader import CONFIG

import os
from typing import List

from tqdm import tqdm

from src.api.sentinel2 import initialize_s3_bucket
from file_types import File, MultiVariateComposite

def upload_composites(comp_files: List[str], s3_bucket_name: str):
    s3 = initialize_s3_bucket(s3_bucket_name)

    for filename in tqdm(comp_files, leave=True, position=0):
        file = File.create(filename)

        if file is None:
            continue

        file_size = os.stat(filename).st_size
        key = file.s3_archive_path()
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, leave=False, position=1) as pbar:
            s3.upload_file(
                Filename=filename,
                Key=key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )


def upload_composites(s3_bucket_name: str, region: str = None, district: str = None, mgrs: List[str] = None):
    comp_files = MultiVariateComposite.find_files(region, district, mgrs)
    upload_composites(comp_files, s3_bucket_name, S3_COMPOSITE_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', '-r', type=str, required=False, help='Name of the region to upload composites for. Defualt'
                        'is all regions in the archive')
    parser.add_argument('--district', '-d', type=str, required=False,
                        help='Name of the district to upload the composites for. Defualt is all districts per input region.')
    parser.add_argument('--mgrs', '-m', type=str, nargs='+', required=False,
                        help='Name of the mgrs tile(s) to download for regions and districts. Default is all tiles')
    parser.add_argument('--s3_bucket_name', '-b', type=str, required=False, default=CONFIG.AWS.BUCKET,
                        help='Name of the s3 bucket to upload the tiles to. Default is bucket specified in config.yaml')

    args = parser.parse_args()

    upload_composites(s3_bucket_name=args.s3_bucket_name, region=args.region, district=args.district, mgrs=args.mgrs)
