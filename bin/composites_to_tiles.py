import argparse
import multiprocessing as mp
import json
import logging
import os
import time
from glob import glob
from argparse import Namespace
from multiprocessing import Pool
import pandas as pd
from src.utilities.imaging import scale
from src.utilities.coords import tiff_to_bbox
from shapely.geometry import polygon
import numpy as np
import rasterio
from osgeo import gdal
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
import geopandas as gpd

def scale_multiband_composite(multiband_tiff:str):
    assert os.path.isfile(multiband_tiff)
    scaled_tiff = multiband_tiff.split('.')[0] + '_scaled.tiff'

    with gdal.Open(multiband_tiff, 'r') as rf:
        with rasterio.open(
            str(scaled_tiff), 
            'w', 
            driver='Gtiff',
            width=rf.width, height=rf.height,
            count=3,
            crs=rf.crs,
            transform=rf.transform,
            dtype='uint8'
        ) as wf:
            wf.write(scale(rf.read(0)).astype('uint8'), 0)
            wf.write(scale(rf.read(1)).astype('uint8'), 1)
            wf.write(scale(rf.read(2)).astype('uint8'), 2)
    return scaled_tiff

def bridge_in_bbox(bbox, bridge_locations):
    p = polygon.Polygon(bbox)
    is_bridge  = False 
    bridge_loc = None
    for loc in bridge_locations: 
        # check if the current tiff tile contains the current verified bridge location
        if p.contains(loc):
            is_bridge  = True 
            bridge_loc = loc
            break
    return is_bridge, bridge_loc
def _batch_task(arg:Namespace):
    rf = gdal.Open(arg.multiband_tiff)
    torchTformer = ToTensor()
    df = pd.DataFrame(
        columns=['tile', 'bbox', 'is_bridge', 'bridge_loc'],
        index=range((len(arg.xsteps)-1)*(len(arg.ysteps)-1))
    )
    k = 0
    for i in range(len(arg.xsteps)-1):
        for j in range(len(arg.ysteps)-1):
            xmin = arg.xsteps[i]
            xmax = arg.xsteps[i + 1] - 1
            ymax = arg.ysteps[j]
            ymin = arg.ysteps[j + 1] - 1
            print(f'xmin {xmin}')
            print(f'xmax {xmax}')
            print(f'ymax {ymax}')
            print(f'ymin {ymin}')
            tile_basename = str(xmin) + '_' + str(ymin) + '.tif'
            tile_tiff = os.path.join(arg.grid_dir, tile_basename)
            gdal.Warp(
                tile_tiff, 
                rf,
                outputBounds=(xmin, ymin, xmax, ymax),
                dstNodata=-999
            )
            bbox = tiff_to_bbox(tile_tiff)
            df.at[k, 'tile']  = tile_tiff
            df.at[k, 'bbox'] = bbox
            df.at[k, 'is_bridge'], df.at[k, 'bridge_loc'] = bridge_in_bbox(bbox, arg.bridge_locations)
            
            with rasterio.open(tile_tiff, 'r') as tmp:
                pt_file = tile_tiff.split('.')[0]+'.pt'
                scale_img = scale(tmp.read()).astype(np.uint8)
                print(scale_img.shape)
                scale_img = np.moveaxis(scale_img, 0, -1) # make dims be c, w, h
                tensor = torchTformer(scale_img)
                torch.save(tensor, pt_file)
                df.at[k, 'tile']  = pt_file
            os.remove(tile_tiff)  
            if k > 5: 
                assert False, 'stop'
            k += 1 
    return df

def make_tiles(
    multiband_tiff,
    tile_dir, 
    bridge_locations,
    cores,
    div:int=300 # in meters
):
    root, military_grid = os.path.split(multiband_tiff) 
    military_grid = military_grid[:5]
    root, region = os.path.split(root)
    root, country = os.path.split(root)
    this_tile_dir = os.path.join(tile_dir, country, region)

    grid_geoloc_file = os.path.join(this_tile_dir, military_grid+'_geoloc.csv')
    if os.path.isfile(grid_geoloc_file):
        return pd.read_csv(grid_geoloc_file)
    
    grid_dir = os.path.join(this_tile_dir, military_grid)
    os.makedirs(grid_dir, exist_ok=True)

    rf = gdal.Open(multiband_tiff)
    _, xres, _, _, _, yres = rf.GetGeoTransform()

    xsteps = np.arange(0, abs(xres)*rf.RasterXSize+1, div).astype(np.int64).tolist()
    ysteps = np.arange(0, abs(yres)*rf.RasterYSize+1, div).astype(np.int64).tolist()
   
    df = None
    with Pool(cores) as p:
        args = [
            Namespace(
                xsteps=xx,
                ysteps=ysteps,
                multiband_tiff=multiband_tiff,
                bridge_locations=bridge_locations,
                grid_dir=grid_dir
            )
            for xx in np.array_split(xsteps, cores)
        ]
        # dfs = p.map(_batch_task, args)
        dfs = []
        for arg in args:
            dfs.append(_batch_task(arg))
        df = pd.concat(dfs, ignore_index=True)
    df.to_csv(grid_geoloc_file)
    return df

def get_bridge_locations(truth_dir):
    bridge_locations = []
    for csv in glob(os.path.join(truth_dir, "*csv")):
        tDf = pd.read_csv(csv)
        bridge_locations += gpd.points_from_xy(tDf['Latitude'], tDf['Longitude'])
    return bridge_locations

def create_tiles(
    composite_dir: str, 
    tile_dir: str,
    truth_dir: str,
    cores: int):

    bridge_locations = get_bridge_locations(truth_dir)  
    composites = glob(os.path.join(composite_dir, "**/*multiband.tiff"), recursive=True)
    dfs = [
        make_tiles(
            multiband_tiff, 
            tile_dir, 
            bridge_locations,
            cores
        ) 
    for multiband_tiff in tqdm(composites)]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(tile_dir,'geoloc.csv'))
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
