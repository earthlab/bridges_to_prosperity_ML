import multiprocessing as mp
import os
from argparse import ArgumentParser
from argparse import Namespace

import numpy as np
from tqdm import tqdm

from src.utilities import imaging
from definitions import COMPOSITE_DIR, SENTINEL_2_DIR


def _composite_task(args):
    imaging.create_composite(
        args.s2_dir, 
        args.composite_dir, 
        args.coord, 
        ['B02', 'B03', 'B04'],
        np.float32,
        args.slices,
        args.n_cores > 1
    )
    return None


def sentinel2_to_composite(s2_dir, composite_dir, slices, n_cores):
    args = []
    for coord in os.listdir(s2_dir):
        args.append(
            Namespace(
                s2_dir=s2_dir, 
                composite_dir=composite_dir, 
                coord=coord, 
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
        '--s2_dir',
        '-i',
        type=str,
        required=False,
        help="Path to local sentinel2 data"
    )
    parser.add_argument(
        '--composite_dir',
        '-o',
        type=str,
        required=False,
        help="Path to where the composites will be written"
    )
    parser.add_argument(
        '--region',
        '-r',
        type=str, 
        required=False,
        help='Name of the composite region (Ex Uganda)'
        )
    parser.add_argument(
        '--district',
        '-d',
        type=str, 
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
        default=1,
        help='number of cores to be used to paralellize these tasks)'
    )
    args = parser.parse_args()

    if args.s2_dir is None or args.composite_dir is None and (args.region is None or args.district is None):
        raise ValueError('Must specify --region and --district if --s2_dir or --composite dir is not specified')

    s2_dir = os.path.join(SENTINEL_2_DIR, args.region, args.district) if args.s2_dir is None else args.s2_dir
    composite_dir = os.path.join(COMPOSITE_DIR, args.region, args.district) if args.composite_dir is None else\
        args.composite_dir

    sentinel2_to_composite(
        s2_dir,
        composite_dir,
        args.slices,
        args.n_cores
    )
