"""
Uploads all the composite files found in the specified input directory to s3 storage
"""
import argparse
import glob
import os
from glob import glob

from tqdm import tqdm
import boto3

from definitions import COMPOSITE_DIR, S3_COMPOSITE_DIR
from src.utilities.aws import initialize_s3
from src.utilities.config_reader import CONFIG


def sync_s3(root_composite_dir: str, s3_bucket_name: str, s3_directory: str):
    comp_files = glob(os.path.join(root_composite_dir, '**', '*_multiband.tiff'), recursive=True)
    s3 = initialize_s3(s3_bucket_name)
    s3 = boto3.client('s3')

    for filename in tqdm(comp_files, leave=True, position=0):
        file_size = os.stat(filename).st_size
        key = os.path.join(s3_directory, filename.strip(root_composite_dir))
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, leave=False, position=1) as pbar:
            s3.upload_file(
                Filename=filename,
                Bucket=s3_bucket_name,
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
    args = parser.parse_args()

    sync_s3(root_composite_dir=args.composite_dir, s3_bucket_name=args.s3_bucket_name, s3_directory=args.s3_directory)
