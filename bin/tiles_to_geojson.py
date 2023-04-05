import json
import multiprocessing as mp
import os
from argparse import Namespace
from glob import glob

import tqdm

from src.utilities.coords import *

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

def task(args:Namespace):
    geom_lookup = {}
    for tif in args.tiles:
        bbox = tiff_to_bbox(tif)
        geom_lookup[tif] = bbox
    with open(args.geojson, 'w+') as f:
        json.dump(geom_lookup, f)

def main():
    tile_dir = os.path.join(BASE_DIR, 'data', 'tiles')
    basedirs = glob(os.path.join(tile_dir, '**', '**', '*'))
    mp.set_start_method('spawn')

    pbar = tqdm.trange(len(basedirs))
    n = mp.cpu_count() - 1
    for k in pbar:
        dir = basedirs[k]
        pbar.set_description(f'{dir}')
        pbar.refresh()
        with mp.Pool(n) as p:
            args = []
            all_tiles = glob(os.path.join(dir, '*.tif'))
            m = len(all_tiles)
            step = int(round(m/n))
            for j, i in enumerate(range(0, m, step)):
                args.append(
                    Namespace(
                        tiles=all_tiles[i:i+step], 
                        geojson = os.path.join(dir, f'geom_lookup_{j}.json')
                    )
                )

            for _ in p.map(task, args):
                pass 
        geom_lookup = {}
        for file in glob(os.path.join(dir, f'geom_lookup_*.json')):
            with open(file, 'r') as f:
                geoms = json.load(f)
                for geom in geoms:
                    geom_lookup[geom] = geoms[geom]
        with open(os.path.join(dir, f'geom_lookup.json'), 'w+') as f:
            json.dump(geom_lookup, f)
    return None

if __name__ == "__main__":
    main()