"""
Uploads all the composite files found in the specified input directory to s3 storage
"""
import argparse
from typing import List

from definitions import COMPOSITE_DIR
from src.utilities.config_reader import CONFIG

from file_types import OpticalComposite
from src.base.upload_composites import upload_composites as base_upload_composites


def upload_composites(root_composite_dir: str, s3_bucket_name: str, bands: List[str] = None, mgrs: List[str] = None):
    comp_files = OpticalComposite.find_files(root_composite_dir, bands=bands, mgrs=mgrs, recursive=True)
    base_upload_composites(comp_files, s3_bucket_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--composite_dir', '-c', type=str, required=False,
                        default=COMPOSITE_DIR,
                        help='Path to the root composites directory that will be crawled for files to upload. Default '
                             'is root project composites directory i.e. all locations will be uploaded. Can also '
                             'specify a specific region and district i.e. data/composites/Uganda/all ')
    parser.add_argument('--s3_bucket_name', '-b', type=str, required=False, default=CONFIG.AWS.BUCKET,
                        help='Name of the s3 bucket to upload the tiles to. Default is bucket specified in config.yaml')
    parser.add_argument('--bands', type=str, required=False, default=None, nargs='+',
                        help='The bands to upload. Default is None for all bands')
    parser.add_argument('--mgrs', '-m', type=str, nargs='+', required=False,
                        help='Name of the mgrs tiles to make tiles for'
    )
    args = parser.parse_args()

    upload_composites(root_composite_dir=args.composite_dir, s3_bucket_name=args.s3_bucket_name,
                      bands=args.bands, mgrs=args.mgrs)
