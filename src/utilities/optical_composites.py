from src.api.sentinel2 import SinergiseSentinelAPI
import multiprocessing as mp
import os
import yaml
from argparse import Namespace
from typing import List

import numpy as np
from tqdm import tqdm

from definitions import REGION_FILE_PATH
from src.utilities.imaging import create_optical_composite_from_s2
from file_types import Sentinel2Tile


def download_sentinel2(region, district, bounds, start_date, end_date, buffer, bands: List[str]):
    api = SinergiseSentinelAPI()
    api.download(bounds, buffer, region, district, start_date, end_date, bands)


def _composite_task(task_args: Namespace):
    create_optical_composite_from_s2(
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


def sentinel2_to_composite(region: str, district: str, slices: int, n_cores: int, bands: List[str], 
                           mgrs: List[str] = None):
    if mgrs is not None:
        mgrs = [c.lower() for c in mgrs]

    mgrs_coords = Sentinel2Tile.get_mgrs_dirs(region, district)

    args = []
    for coord in mgrs_coords:
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
        bounds = region_info[region]['districts'][district]['bbox']
        print('Downloading Sentinel2 data')
        for date in dates:
            download_sentinel2(region, district, bounds, date[0], date[1], buffer, bands)

        sentinel2_to_composite(region, district, slices, n_cores, bands, region, mgrs=mgrs)
