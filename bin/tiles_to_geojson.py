import json
import multiprocessing as mp
import argparse

import tqdm

from src.utilities.coords import *


def task(args: argparse.Namespace):
    geom_lookup = {}
    for tif in args.tiles:
        bbox = tiff_to_bbox(tif)
        geom_lookup[tif] = bbox
    with open(args.geojson, 'w+') as f:
        json.dump(geom_lookup, f)


def find_files(root_dir: str, file_extension: str):
    """
    Recursively searches for files with the given file extension in the given root directory and its subdirectories.
    :param root_dir: The root directory to start searching from.
    :param file_extension: The file extension to search for (e.g., ".txt").
    :return: A list of absolute paths to the files with the given file extension.
    """
    found_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(file_extension):
                found_files.append(os.path.abspath(os.path.join(root, file)))
    return found_files


def main(tile_dir: str, ):
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
                    argparse.Namespace(
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
    parser = argparse.ArgumentParser()
