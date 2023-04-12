"""
Downloads sentinel2 data using Sinergise API and creates composites for each specified region. Composites are kept
locally and uploaded to s3 along with the Sentinel 2 files used to create the composite.
"""
import argparse
import multiprocessing
import os
import tqdm
import subprocess
from typing import List

import numpy as np
import yaml

from definitions import B2P_DIR, REGION_FILE_PATH
from src.api.sentinel2 import SinergiseSentinelAPI
from src.utilities import imaging
from src.utilities.aws import upload_to_s3, initialize_s3
from src.utilities.config_reader import CONFIG


# TODO: Add checks for existing data in s3 for requested regions so as to not duplicate data unnecessarily,
#  with override option

# TODO: Start thinking about file structure in s3 so we can hash each request configuration

def _composite_task(task_args: argparse.Namespace):
    multi_band_file_path = imaging.create_composite(
        task_args.district_s2_dir,
        task_args.district_composite_dir,
        task_args.coord,
        ['B02', 'B03', 'B04'],
        np.float32,
        task_args.slices,
        task_args.n_cores > 1)

    s3_session = initialize_s3(CONFIG.AWS.BUCKET)

    upload_to_s3(s3_session, multi_band_file_path,
                 os.path.join('composites', task_args.region, task_args.district,
                              os.path.basename(multi_band_file_path)),
                 bucket=task_args.s3_bucket)

    return None


def get_optical_data(sentinel_2_dir: str, composite_dir: str, bands: List[str], buffer: float, slices: int,
                     s3_bucket: str, requested_regions: List[str] = None, requested_districts: List[str] = None,
                     upload_s2_dir: bool = False):
    s3_session = initialize_s3(CONFIG.AWS.BUCKET)

    with open(REGION_FILE_PATH, 'r') as file:
        region_info = yaml.safe_load(file)

    api = SinergiseSentinelAPI()

    # Check to see if each requested region is in region_info.yml file
    if requested_regions is not None:
        for requested_region in requested_regions:
            if requested_region not in region_info:
                raise LookupError(f'Requested {requested_region} not in {REGION_FILE_PATH}. Please add this region and '
                                  f'its information to this file before requesting its optical data.')

    print('======================================================')
    print('Starting script')
    print(requested_districts, requested_regions)
    for region, info in region_info.items():
        if requested_regions is not None and region not in requested_regions:
            continue

        dates = info['dates']

        for district, district_info in info['districts'].items():
            if requested_districts is not None and district not in requested_districts:
                continue
            bbox = district_info['bbox']
            print(f'{region}/{district}\n'
                  f'\tdates : {dates}\n'
                  f'\tbbox : {bbox}')

            # Define src and dst dirs
            district_s2_dir = os.path.join(sentinel_2_dir, region, district)
            district_composite_dir = os.path.join(composite_dir, region, district)
            print(district_s2_dir, district_composite_dir)
            os.makedirs(district_s2_dir, exist_ok=True)
            os.makedirs(district_composite_dir, exist_ok=True)

            # Download data from s2 api
            print('--------------------------------')
            print('downloading sentinel2')
            for date in dates:
                start_date, end_date = date
                print(start_date, end_date)
                api.download(bbox, buffer, district_s2_dir, start_date, end_date)

            # create composite
            print('--------------------------------')
            print('creating composites')
            n_cores = multiprocessing.cpu_count() - 1  # TODO: Make this a parameter
            args = []
            for coord in os.listdir(district_s2_dir):
                if coord.startswith("."):
                    continue
                args.append(
                    argparse.Namespace(
                        district_s2_dir=district_s2_dir,
                        district_composite_dir=district_composite_dir,
                        coord=coord,
                        slices=slices,
                        n_cores=n_cores,
                        s3_bucket=s3_bucket,
                        region=region,
                        district=district
                    )
                )

            print('Building composites...')

            # if n_cores == 1:
            #     print('\tNot using multiprocessing...')
            #     for arg in tqdm(args, total=len(args), desc="Sequential...", leave=True):
            #         _composite_task(arg)
            # else:
            print('\tUsing multiprocessing...')
            with multiprocessing.Pool(n_cores) as pool:
                for _ in tqdm(pool.imap_unordered(_composite_task, args)):
                    pass

            # Compress sentinel2
            if upload_s2_dir:
                print('-----------------------------')
                print('Compressing raw s2')
                tar_file = os.path.join(B2P_DIR, 'data', 'sentinel2', f's2_{region}_{district}.tar.gz')
                tar_cmd = f'tar -czvf {tar_file}'

                tar_cmd += f' --remove-files {district_s2_dir}'
                process = subprocess.Popen(tar_cmd.split(), shell=False)
                process.communicate()

                # Upload raw s2 to s3
                upload_to_s3(s3_session, tar_file, os.path.join('sentinel2_raw', os.path.basename(tar_file)),
                             bucket=s3_bucket)

                # remove raw data from this vm
                os.remove(tar_file)
                print('============================================\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', '-r', required=False, nargs='+', type=str,
                        help='The regions to download data and create composites for. If not specified, all regions '
                             'in the data/region_info.yaml file will be prepared. Multiple regions are separated by'
                             ' spaces from the command line. Ex. zambia ethiopa')
    parser.add_argument('--districts', '-d', required=False, nargs='+', type=str,
                        help="The districts to download data and create composites for. If not specified, "
                             "all districts for each region file will be prepared. Multiple districts are separated "\
                             "by spaces from the command line. Ex. zambia 'cote divoire' ethiopa")
    parser.add_argument('--s2_dir', '-s', required=False, type=str,
                        default=os.path.join(B2P_DIR, 'data', 'sentinel2'),
                        help='Path to the base directory where sentinel 2 files will be written to. Set to '
                             'data/sentinel2 by default')
    parser.add_argument('--composite_dir', '-c', required=False, type=str,
                        default=os.path.join(B2P_DIR, 'data', 'composites'),
                        help='Path to the base directory where composites will be written to. Set to data/sentinel2 by'
                             ' default')
    parser.add_argument('--upload_s2', '-s2', action='store_true',
                        help='If specified then the s2 images will not be removed after being uploaded to s3 and will '
                             'be kept locally')
    parser.add_argument('--bands', '-b', required=False, nargs='+', type=str, default=['B02', 'B03', 'B04'],
                        help='The Sentinel 2 bands to download for the requested regions. Multiple regions are '
                             'separated by a comma fro the command line. Ex. B02,B03 .')
    parser.add_argument('--buffer', type=float, required=False, default=1000,
                        help='Buffer in meters to apply to bounding box area when making data request')
    parser.add_argument('--slices', type=int, required=False, default=12,
                        help='The amount of slices to split composites up to when being created. The more slices the '
                             'less RAM will be used, but will be slower.')
    parser.add_argument('--s3_bucket', type=str, required=False, default=CONFIG.AWS.BUCKET,
                        help='Name of the s3 bucket that the composites and Sentinel2 files will be uploaded to')
    args = parser.parse_args()

    get_optical_data(requested_regions=args.regions, requested_districts=args.districts, sentinel_2_dir=args.s2_dir,
                     bands=args.bands, buffer=args.buffer, slices=args.slices, s3_bucket=args.s3_bucket,
                     composite_dir=args.composite_dir, upload_s2_dir=args.upload_s2)
