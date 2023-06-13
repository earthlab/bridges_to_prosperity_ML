"""
Uploads all the composite files found in the specified input directory to s3 storage
"""
import argparse

from definitions import COMPOSITE_DIR, S3_COMPOSITE_DIR
from src.utilities.config_reader import CONFIG

from file_types import MultiVariateComposite
from src.base.upload_composites import upload_composites as base_upload_composites


def upload_composites(root_composite_dir: str, s3_bucket_name: str, s3_directory: str):
    comp_files = MultiVariateComposite.find_files(root_composite_dir, recursive=True)
    base_upload_composites(comp_files, s3_bucket_name, s3_directory)


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

    upload_composites(root_composite_dir=args.composite_dir, s3_bucket_name=args.s3_bucket_name,
                      s3_directory=args.s3_directory)
