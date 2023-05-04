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
        task_args.s2_dir,
        task_args.composite_dir,
        task_args.coord,
        task_args.bands,
        np.float32,
        task_args.slices,
        task_args.n_cores > 1
    )
    return None


def sentinel2_to_composite(slices, n_cores, bands: List[str], region: str, districts: List[str] = None):
    if districts is None:
        with open(REGION_FILE_PATH, 'r') as f:
            region_info = yaml.safe_load(f)
            districts = list(region_info[region]['districts'].keys())

    for district in districts:
        args = []
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
            print('\tUsing multiprocessing...')
            with mp.Pool(n_cores) as pool:
                for _ in tqdm(pool.imap_unordered(_composite_task, args)):
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
        default=[]
    )
    args = parser.parse_args()

    if args.district is None:
        # TODO: Get all districts for region
        pass

    s2_dir = os.path.join(SENTINEL_2_DIR, args.region, args.district) if args.s2_dir is None else args.s2_dir
    composite_dir = os.path.join(COMPOSITE_DIR, args.region, args.district) if args.composite_dir is None else\
        args.composite_dir

    sentinel2_to_composite(
        s2_dir,
        composite_dir,
        args.slices,
        args.n_cores,
        args.bands
    )
