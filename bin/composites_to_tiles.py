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
    composites = []
    for multiband_tiff in glob(os.path.join(composite_dir, "**/*multiband.tiff"), recursive=True):
        root, military_grid = os.path.split(multiband_tiff) 
        military_grid = military_grid[:5]
        root, region = os.path.split(root)
        root, country = os.path.split(root)
        this_tile_dir = os.path.join(tile_dir, country, region)

        grid_geoloc_file = os.path.join(this_tile_dir, military_grid+'_geoloc.csv')
        if not os.path.isfile(grid_geoloc_file):
            composites.append(multiband_tiff)
    
    if cores > 1:
        with mp.Pool(cores) as p:
            items = [
                (multiband_tiff, tile_dir, copy(bridge_locations), n%cores+1)
            for n, multiband_tiff in enumerate(composites)]
            p.starmap(tiff_to_tiles,items)
    else:
        for multiband_tiff in tqdm(composites, position=0, leave=True):
            tiff_to_tiles(multiband_tiff, tile_dir, bridge_locations,1)

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
