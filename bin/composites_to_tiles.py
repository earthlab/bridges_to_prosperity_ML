import argparse
import multiprocessing as mp
import os
from glob import glob
from copy import copy
import pandas as pd
from tqdm import tqdm
import numpy as np
from tqdm.contrib.concurrent import process_map
from src.utilities.imaging import tiff_to_tiles
from src.utilities.coords import get_bridge_locations

def create_tiles(arg):
    composites, tile_dir, bridge_locations, pos = arg
   
    df = []
    for multiband_tiff in tqdm(composites, position=0, leave=True):
        df_i = tiff_to_tiles(multiband_tiff, tile_dir, bridge_locations, pos)
        df.append(
            df_i
        )
    df = pd.concat(df, ignore_index=True)
    return df

if __name__ == '__main__':
    
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)
            ), 
            '..'
        )
    )
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--in_dir', 
        '-i', 
        type=str, 
        required=False, 
        default=os.path.join(base_dir, 
        "data", 
        "composites"), 
        help='path to inpt directory where s2 path is'
    )
    parser.add_argument(
        '--out_dir', 
        '-o', 
        type=str, 
        required=False, 
        default=os.path.join(base_dir, "data", "tiles"),
         help='Path to directory where output files will be written'
    )
    parser.add_argument(
        '--truth-dir', 
        '-t', 
        type=str, 
        required=False, 
        default=os.path.join(base_dir, "data", "ground_truth"),
         help='Path to directory where csv bridge locs'
    )
    parser.add_argument(
        '--cores', 
        '-c', 
        type=int, 
        default=mp.cpu_count() - 1,
        required=False, 
        help='Number of cores to use in parallel for tiling'
    )
    
    args = parser.parse_args()
   
    # call function 
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    bridge_locations = get_bridge_locations(args.truth_dir)  
    composites = glob(os.path.join(args.in_dir, "**/*multiband.tiff"), recursive=True)
    if args.cores == 1:
        create_tiles(
            (
                composites,
                args.out_dir,
                bridge_locations, 
                1
            )
        )
    else:
        inputs = [
            (
                cs,
                args.out_dir,
                bridge_locations, 
                n 
            )
        for n, cs in enumerate(np.array_split(composites, args.cores))]
        process_map(
            create_tiles, 
            inputs, 
            max_workers=args.cores
        )
