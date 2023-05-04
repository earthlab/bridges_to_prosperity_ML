import multiprocessing
import multiprocessing as mp
import os
import yaml
from argparse import ArgumentParser
from argparse import Namespace
from typing import List

import numpy as np
from tqdm import tqdm

from definitions import COMPOSITE_DIR, SENTINEL_2_DIR, REGION_FILE_PATH
from src.utilities import imaging


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
    sublists = [lst[i:i+n] for i in range(0, len(lst), n)]

    # handle the remainder if the final sublist is smaller than n
    if len(sublists[-1]) < n:
        last = sublists.pop()
        sublists[-1].extend(last)

    return sublists


def sentinel2_to_composite(slices, n_cores: int, bands: List[str], region: str, districts: List[str] = None):
    if districts is None:
        with open(REGION_FILE_PATH, 'r') as f:
            region_info = yaml.safe_load(f)
            districts = list(region_info[region]['districts'].keys())

    for district in districts:
        args = []
        # TODO: Centralize
        s2_dir = os.path.join(SENTINEL_2_DIR, region, district)
        for coord in os.listdir(s2_dir):
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
            parallel_batches = split_list(args, n_cores)
            print('\tUsing multiprocessing...')
            with mp.Pool(n_cores) as pool:
                results = []
                for group in parallel_batches:
                    results.append(pool.imap_unordered(_composite_task, group))
                for res in tqdm(results):
                    for _ in res:
                        pass


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
        '--n_cores',
        '-n',
        type=int, 
        required=False, 
        default=multiprocessing.cpu_count() - 1,
        help='number of cores to be used to paralellize these tasks)'
    )
    parser.add_argument(
        '--bands',
        '-b',
        type=str,
        nargs='+',
        required=False,
        default=[]  # TODO: Make required with no default?
    )
    args = parser.parse_args()

    sentinel2_to_composite(
        args.slices,
        args.n_cores,
        args.bands,
        args.region,
        args.districts
    )
