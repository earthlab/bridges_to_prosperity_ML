import multiprocessing as mp
import os
from argparse import ArgumentParser
from argparse import Namespace

import numpy as np
from tqdm import tqdm

from src.utilities import imaging


BANDS = ['B02', 'B03', 'B04']
DTYPE = np.float32
def _composite_task(args):
    imaging.create_composite(
        args.s2_dir, 
        args.composite_dir, 
        args.coord, 
        BANDS, 
        DTYPE,
        args.slices,
        args.n_cores>1
    )
    return None

def sentinel2_to_composite(s2_dir, composite_dir, region, district, slices, n_cores):
    args = []
    s2_dir = os.path.join(s2_dir, region, district)
    composite_dir = os.path.join(composite_dir, region, district)
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
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)
            ), 
            '..'
        )
    )
    parser = ArgumentParser()
    parser.add_argument(
        '--s2_dir',
        '-i',
        type=str,
        required=False,
        default=os.path.join(base_dir, "data", "sentinel2"),
        help="Path to local sentinel2 data"
    )
    parser.add_argument(
        '--composite_dir',
        '-o',
        type=str,
        required=False,
        default=os.path.join(base_dir, "data", "composite"),
        help="Path to local sentinel2 data"
    )
    parser.add_argument(
        '--region',
        '-r',
        type=str, 
        required=True, 
        help='Name of the composite region (Ex Uganda)'
        )
    parser.add_argument(
        '--district',
        '-d',
        type=str, 
        required=True, 
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
    sentinel2_to_composite(
        args.s2_dir, 
        args.composite_dir,
        args.region, 
        args.district, 
        args.slices,
        args.n_cores
    )
