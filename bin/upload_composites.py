"""
Uploads all the multivariate composite files found to s3 storage. Region, district, and military grid can be specified to narrow the search
"""
import argparse

from definitions import S3_COMPOSITE_DIR
from src.utilities.config_reader import CONFIG

import os
from typing import List

from tqdm import tqdm

from src.api.sentinel2 import initialize_s3_bucket
from file_types import MultiVariateComposite


def upload_composites(s3_bucket_name: str, region: str = None, district: str = None, mgrs: List[str] = None):
    comp_files = MultiVariateComposite.find_files(region, district, mgrs)
    s3 = initialize_s3_bucket(s3_bucket_name)
    for composite_file_path in tqdm(comp_files, leave=True, position=0):
        composite_file = MultiVariateComposite.create(composite_file_path)
        file_size = os.stat(composite_file_path).st_size
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=composite_file_path, leave=False, position=1) as pbar:
            s3.upload_file(
                Filename=composite_file_path,
                Key=composite_file.s3_archive_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )


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
