import argparse
import json
import multiprocessing as mp
import os
from glob import glob

import tqdm

from definitions import TILE_DIR
from src.utilities.coords import tiff_to_bbox
from src.utilities.files import find_directories


def task(task_args: argparse.Namespace):
    geom_lookup = {}
    for tif in task_args.tiles:
        bbox = tiff_to_bbox(tif)
        geom_lookup[tif] = bbox
    with open(args.geojson, 'w+') as f:
        json.dump(geom_lookup, f)


def tiles_to_geojson(base_tile_dir: str = TILE_DIR, cores: int = mp.cpu_count() - 1):
    tile_directories = find_directories(base_tile_dir, '.tif')
    mp.set_start_method('spawn')

    pbar = tqdm.trange(len(tile_directories))
    for k in pbar:
        tile_directory = tile_directories[k]
        pbar.set_description(f'{tile_directory}')
        pbar.refresh()
        with mp.Pool(cores) as p:
            parallel_args = []
            all_tiles = glob(os.path.join(tile_directory, '*.tif'))
            m = len(all_tiles)
            step = int(round(m/cores))
            for j, i in enumerate(range(0, m, step)):
                parallel_args.append(
                    argparse.Namespace(
                        tiles=all_tiles[i:i+step], 
                        geojson=os.path.join(tile_directory, f'geom_lookup_{j}.json')
                    )
                )

            for _ in p.map(task, parallel_args):
                pass 
        geom_lookup = {}
        for file in glob(os.path.join(tile_directory, f'geom_lookup_*.json')):
            with open(file, 'r') as f:
                geoms = json.load(f)
                for geom in geoms:
                    geom_lookup[geom] = geoms[geom]
        with open(os.path.join(tile_directory, f'geom_lookup.json'), 'w+') as f:
            json.dump(geom_lookup, f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_tile_dir', '-t', required=False, type=str, default=TILE_DIR,
                        help='Directory that will be crawled for tif tile files. This directory will be indexed '
                             'recursively and a geom_lookup.json file will be created for each directory that contains '
                             'a .tif file. A single directory can be specified as well, i.e. '
                             'data/tiles/Uganda/Ibanda/35NRA. Default is data/tiles')
    parser.add_argument('--cores', '-c', required=False, default=mp.cpu_count() - 1, type=int,
                        help='The amount of cores to use when making each geom_lookup file in parallel. Default is '
                             'cpu_count - 1')
    args = parser.parse_args()

    tiles_to_geojson(base_tile_dir=args.base_tile_dir, cores=args.cores)
