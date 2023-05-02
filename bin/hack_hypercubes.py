import argparse
import multiprocessing as mp
import os
import random
from glob import glob
from typing import List, Tuple, Union
from argparse import Namespace

import numpy as np
import pandas as pd
import rasterio
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from bin.composites_to_tiles import create_tiles
from definitions import REGION_FILE_PATH, COMPOSITE_DIR, TILE_DIR, TRUTH_DIR, SENTINEL_2_DIR
from src.utilities.aws import initialize_s3
from src.utilities.imaging import create_composite
from src.utilities.config_reader import CONFIG
from src.utilities.coords import get_bridge_locations
from bin.download_sentinel2 import download_sentinel2
from bin.sentinel2_to_composite import sentinel2_to_composite
CORES = mp.cpu_count() - 1

def this_download(location_request_info: Tuple[str, int, int, str]) -> None:
    for location_path, composite_size, destination, position in location_request_info:
        s3 = initialize_s3(CONFIG.AWS.BUCKET)
        bucket = s3.Bucket(CONFIG.AWS.BUCKET)
        if os.path.isfile(destination):
            return None
        dst_root = os.path.split(destination)[0]
        os.makedirs(dst_root, exist_ok=True)
        print('Downloading')
        with tqdm(total=int(composite_size), unit='B', unit_scale=True, desc=location_path, leave=False,
                  position=int(position)) as pbar:
            print(destination)
            bucket.download_file(location_path, destination,
                                 Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    return None


def _composite_task(task_args: Namespace):
    create_composite(
        task_args.s2_dir,
        task_args.composite_dir,
        task_args.coord,
        ['B08'],
        np.float32,
        task_args.slices,
        task_args.n_cores > 1
    )
    return None


def IR_only_composite(s2_dir, composite_dir):
    args = []
    for coord in os.listdir(s2_dir):
        args.append(
            Namespace(
                s2_dir=s2_dir, 
                composite_dir=composite_dir, 
                coord=coord, 
                slices=6, # default from other file
                n_cores=CORES
            )
        )
    print('Building composites...')

    if CORES == 1: 
        print('\tNot using multiprocessing...')
        for arg in tqdm(args, total=len(args), desc="Sequential...", leave=True):
            _composite_task(arg)
    else: 
        print('\tUsing multiprocessing...')
        with mp.Pool(CORES) as pool:
            for _ in tqdm(pool.imap_unordered(_composite_task, args)): 
                pass

def hack_hypecubes(requested_locations: List[str], composites_dir: str = COMPOSITE_DIR,
                   tiles_dir: str = TILE_DIR, s3_bucket_name: str = CONFIG.AWS.BUCKET,
                   bucket_composite_dir: str = 'composites', truth_file_path: Union[None, str] = None,
                   train_to_test_ratio: float = 0.7, cores: int = CORES):
    debug = True
    ## Load existing opitcal only composites from S3
    # Load composites from s3
    s3 = initialize_s3(s3_bucket_name)
    s3_bucket = s3.Bucket(s3_bucket_name)

    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    location_info = []
    for location in requested_locations:
        for obj in s3_bucket.objects.filter(Prefix=location):
            if obj.key.endswith('.tiff'):
                destination = os.path.join(composites_dir, obj.key.strip(f'{bucket_composite_dir}/'))
                location_info.append((obj.key, obj.size, destination))

    parallel_inputs = []
    for i, location in enumerate(np.array_split(location_info, cores)):
        parallel_inputs.append([])
        for info_tuple in location:
            print(info_tuple)
            parallel_inputs[i].append((info_tuple[0], info_tuple[1], info_tuple[2], str(i + 1)))
    print('Downloading composites from S3...')
    process_map(
        this_download,
        parallel_inputs,
        max_workers=cores
    )

    for loc in requested_locations:
        if '/' not in loc:
            region, district = os.path.split(loc)
            bbox = region_info[region][district]['bbox']
            dates = region_info[region]['dates']
            s2_dir = os.path.join(SENTINEL_2_DIR, region, district)
            ir_composite_dir = os.path.join(SENTINEL_2_DIR, '..', 'ir', region, district)
            os.makedirs(ir_composite_dir, exist_ok=True)
            for date in dates: 
                download_sentinel2(s2_dir, bbox, date[0], date[1], 100)
            print('Making ir composites')
            IR_only_composite(s2_dir, ir_composite_dir)
            RGB_composites =  glob(os.path.join(composites_dir, region, district, "**/*multiband.tiff"), recursive=True)
            for rgb_comp in RGB_composites:
                base_name = os.path.basename(rgb_comp)
                ir_comp = os.path.join(composites_dir, region, district, base_name)
                assert os.path.isfile(ir_comp), f'IR composite should already exist for {ir_comp}'
                dst_tiff = rgb_comp.split('multiband.tiff')[0]+'RGBplusIR.tiff'

                with rasterio.open(rgb_comp, 'r') as src_rgb:
                    with rasterio.open(ir_comp, 'r') as src_ir:
                        # copy and update the metadata from the input raster for the output
                        meta = src_rgb.meta.copy()
                        d = meta['count'] # should be 3 or 4 to include RGB and maybe IR (near wave)
                        meta.update(
                            count=d+1
                        )
                        with rasterio.open(dst_tiff, 'w+', **meta) as dst:
                            if debug: print('Copying RGB data to hypecube tiff')
                            for i in range(d): 
                                dst.write_band(i+1, src_rgb.read(i+1))
                            dst.write_band(d+1, src_ir.read(0))
        else :
            assert False, "not implemented"

if __name__ == "__main__":
    hack_hypecubes(
        ['Ethiopia/Fafan']
    )
