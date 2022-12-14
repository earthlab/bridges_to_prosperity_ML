#!/usr/bin/env python
import os
import json
import shutil
import argparse

import shapely
import pandas as pd
import geopandas as gpd
import shapely.geometry
from raster2xyz.raster2xyz import Raster2xyz
from rasterio.crs import CRS
import random


def main(input_file: str, shape_dir: str, crs: str, outfile: str):
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    tiling_files = input_data['tiling_files']
    file_bounds = []
    for tiff_file_path in tiling_files:
        tiff_file_name = os.path.basename(tiff_file_path)
        out_csv = os.path.join(shape_dir, os.path.basename(tiff_file_path.replace('.tiff', '.csv')))
        rtxyz = Raster2xyz(verbose=False)
        rtxyz.translate(tiff_file_path, out_csv)
        my_raster_df = pd.read_csv(out_csv)
        my_raster_df = gpd.GeoDataFrame(geometry=[shapely.geometry.Point(d.x, d.y) for s, d in my_raster_df.iterrows()],
                                        crs=CRS.from_string(crs))

        shutil.copyfile(tiff_file_path, os.path.join(shape_dir, tiff_file_name))
        shape_file_path = os.path.join(shape_dir, tiff_file_name[:-4] + '.shp')
        my_raster_df.to_file(shape_file_path)
        if os.path.exists(out_csv):
            os.remove(out_csv)
        del out_csv, rtxyz, my_raster_df

        trial = gpd.read_file(shape_file_path)
        bound = trial.unary_union.bounds
        bound = ((bound[0], bound[1]), (bound[0], bound[3]), (bound[2], bound[3]), (bound[2], bound[1]))
        file_bounds.append((shape_file_path, bound))

        del trial, bound

    with open(outfile, 'w+') as f:
        json.dump({'file_bounds': file_bounds}, f, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to json file with tiling file paths')
    parser.add_argument('--shape_dir', type=str, required=True, help='Path to output shape file directory')
    parser.add_argument('--crs', type=str, required=True, help='Coordinate reference system for GeoDataFrame')
    parser.add_argument('--outpath', type=str, required=True, help='Path to the output json file')
    args = parser.parse_args()
    main(args.input_file, args.shape_dir, args.crs, args.outpath)
