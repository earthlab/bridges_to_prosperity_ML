from src.api.sentinel2 import SinergiseSentinelAPI
import multiprocessing as mp
import os
import yaml
from argparse import Namespace, ArgumentParser
from typing import List

import numpy as np
from tqdm import tqdm

from definitions import SENTINEL_2_DIR, REGION_FILE_PATH
from src.utilities import imaging


# TODO: Add bands parameter
def download_sentinel2(output_dir, bounds, start_date, end_date, buffer, bands: List[str]):
    api = SinergiseSentinelAPI()
    api.download(bounds, buffer, output_dir, start_date, end_date, bands)

    if not os.listdir(output_dir):
        raise FileNotFoundError('No files returned from the query parameters')


def _composite_task(task_args: Namespace):
    imaging.create_composite(
        task_args.region,
        task_args.district,
        task_args.coord,
        task_args.bands,
        np.float32,
        task_args.slices,
        task_args.n_cores > 1
    )
    return None


def split_list(lst, n):
    # create a list of sublists of size n
    sublists = [lst[i:i + n] for i in range(0, len(lst), n)]

    # handle the remainder if the final sublist is smaller than n
    if len(sublists[-1]) < n:
        last = sublists.pop()
        sublists[-1].extend(last)

    return sublists


def sentinel2_to_composite(s2_dir, district, slices, n_cores: int, bands: List[str], region: str,
                           mgrs: List[str] = None):
    if mgrs is not None:
        mgrs = [c.lower() for c in mgrs]

    args = []
    for coord in os.listdir(s2_dir):
        if mgrs is not None and coord.lower() not in mgrs:
            continue

        args.append(
            Namespace(
                region=region,
                district=district,
                coord=coord,
                bands=bands,
                slices=slices,
                n_cores=n_cores
            )
        )
    print('Building composites...')

    if n_cores == 1:
        print('\tNot using multiprocessing...')
        for arg in tqdm(args, total=len(args), desc="Sequential...", leave=True):
            _composite_task(arg)
    else:
        with mp.Pool(n_cores) as pool:
            parallel_batches = split_list(args, n_cores) if n_cores > len(args) else args
            print(parallel_batches)
            print(n_cores)
            print('\tUsing multiprocessing...')
            results = []
            for group in parallel_batches:
                print('Processing group')
                results.append(pool.imap_unordered(_composite_task, group))
            for res in tqdm(results):
                for _ in res:
                    pass


def create_composites(region: str, bands: List[str], buffer: int, slices: int, n_cores: int, mgrs: List[str],
                      districts: List[str] = None):
    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    if districts is None:
        districts = list(region_info[region]['districts'].keys())

    dates = region_info[region]['dates']

    for district in districts:
        s2_dir = os.path.join(SENTINEL_2_DIR, region, district)
        bounds = region_info[region]['districts'][district]['bbox']
        print('Downloading Sentinel2 data')
        for date in dates:
            download_sentinel2(s2_dir, bounds, date[0], date[1], buffer, bands)

        sentinel2_to_composite(s2_dir, district, slices, n_cores, bands, region, mgrs=mgrs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--region',
        '-r',
        type=str,
        required=True,
        help='Name of the composite region (Ex Uganda)'
    )
    parser.add_argument(
        '--districts',
        '-d',
        type=str,
        nargs='+',
        required=False,
        help='Name of the composite district (Ex. Uganda/Ibanda)'
    )
    parser.add_argument(
        '--slices',
        '-s',
        type=int,
        required=False,
        default=6,
        help='The number of slices to break the sentinel2 tiles up into before cloud-correcting (Default is 1)'
    )
    parser.add_argument(
        '--buffer',
        required=False,
        default=100,
        type=float,
        help='Buffer for bounding box query in meters'
    )
    parser.add_argument(
        '--n_cores',
        '-n',
        type=int,
        required=False,
        default=mp.cpu_count() - 1,
        help='number of cores to be used to paralellize these tasks)'
    )
    parser.add_argument(
        '--mgrs',
        '-m',
        type=str,
        nargs='+',
        required=False,
        help='Name of the mgrs tiles to make tiles for'
    )
    parser.add_argument(
        '--bands',
        '-b',
        type=str,
        nargs='+',
        required=False,
        default=['B02', 'B03', 'B04', 'B08']
    )
    args = parser.parse_args()

    create_composites(args.region, bands=args.bands, buffer=args.buffer, slices=args.slices, n_cores=args.n_cores,
                      districts=args.districts, mgrs=args.mgrs)
