import argparse
import multiprocessing as mp
import os
from glob import glob
from copy import copy
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.utilities.imaging import tiff_to_tiles
from src.utilities.coords import get_bridge_locations

def create_tiles(
    composite_dir: str, 
    tile_dir: str,
    truth_dir: str,
    cores: int):

    bridge_locations = get_bridge_locations(truth_dir)  
    composites = glob(os.path.join(composite_dir, "**/*multiband.tiff"), recursive=True)
   
    df = None
    if cores > 1:
        with mp.Pool(CORES) as p:
            items = [
                (multiband_tiff, TILE_DIR, copy(bridge_locations), n%cores+1)
            for n, multiband_tiff in enumerate(composites)]
            df = list(p.starmap(tiff_to_tiles,items))
    else:
        df = []
        print(len(composites))
        for multiband_tiff in tqdm(composites, position=0, leave=True):
            df_i = tiff_to_tiles(multiband_tiff, tile_dir, bridge_locations,1)
            print(type(df_i))
            print(df_i.shape)
            df.append(
                df_i
            )
    print(type(df))
    print(len(df))
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

    create_tiles(
        args.in_dir,
        args.out_dir,
        args.truth_dir, 
        args.cores,
    )
