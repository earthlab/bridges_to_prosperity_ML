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
from osgeo import gdal

from bin.composites_to_tiles import create_tiles
from definitions import REGION_FILE_PATH, COMPOSITE_DIR, TILE_DIR, TRUTH_DIR, SENTINEL_2_DIR, ELEVATION_DIR, SLOPE_DIR
from src.utilities.aws import initialize_s3
from src.utilities.imaging import numpy_array_to_raster
from src.utilities.config_reader import CONFIG
from src.utilities.imaging import elevation_to_slope, subsample_geo_tiff
from src.utilities.coords import get_bridge_locations
from bin.download_sentinel2 import download_sentinel2
from bin.sentinel2_to_composite import sentinel2_to_composite
from bin.prepare_optical_inputs import download_composites
from src.api.sentinel2 import SinergiseSentinelAPI
from src.api.lp_daac import Elevation
from src.api.osm import getOsm

CORES = mp.cpu_count() - 1


def combine_bands(source_file: str, target_file: str):
    with rasterio.open(source_file, 'r') as src_file:
        # copy and update the metadata from the input raster for the output
        meta = src_file.meta.copy()
        d = meta['count']
        meta.update(
            count=d + 1
        )
        with rasterio.open(target_file, 'w+', **meta) as dst:
            for i in range(d):
                dst.write_band(i + 1, src_file.read(i + 1))


def hack_hypercubes(requested_locations: List[str], composites_dir: str = COMPOSITE_DIR,
                    tiles_dir: str = TILE_DIR, s3_bucket_name: str = CONFIG.AWS.BUCKET,
                    bucket_composite_dir: str = 'composites', truth_file_path: Union[None, str] = None,
                    train_to_test_ratio: float = 0.7, cores: int = CORES):
    debug = True
    # Load composites from S3
    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    download_composites(requested_locations, composites_dir, s3_bucket_name, bucket_composite_dir, cores)

    sentinel2_api = SinergiseSentinelAPI()
    elevation_api = Elevation()

    for loc in requested_locations:
        if '/' in loc:
            region, district = os.path.split(loc)
            bbox = region_info[region][district]['bbox']
            dates = region_info[region]['dates']
            s2_dir = os.path.join(SENTINEL_2_DIR, region, district)
            ir_composite_dir = os.path.join(SENTINEL_2_DIR, '..', 'ir', region, district)
            os.makedirs(ir_composite_dir, exist_ok=True)
            for date in dates:
                sentinel2_api.download(bbox, 100, s2_dir, date[0], date[1], bands=['B08'])
            print('Making ir composites')
            sentinel2_to_composite(s2_dir, ir_composite_dir, 6, cores, bands=['B08'])  # TODO: Make slices configurable
            RGB_composites = glob(os.path.join(composites_dir, region, district, "**/*multiband.tiff"), recursive=True)
            for rgb_comp in RGB_composites:
                base_name = os.path.basename(rgb_comp)
                ir_comp = os.path.join(ir_composite_dir, region, district, base_name)
                assert os.path.isfile(ir_comp), f'IR composite should already exist for {ir_comp}'
                dst_tiff = rgb_comp.replace('multiband.tiff', 'RGBplusIR.tiff')

                # return
                # Get elevation data for requested region
                # TODO: Make file naming classes
                # TODO: Check if files already exist before making them
                elevation_outfile = os.path.join(ELEVATION_DIR, region, district, base_name.replace('multiband.tiff',
                                                                                                    'elevation.tiff'))
                elevation_api.download_district(elevation_outfile, region, district, buffer=100)

                slope_outfile = os.path.join(SLOPE_DIR, region, district, base_name.replace('multiband.tiff',
                                                                                            'slope.tiff'))
                elevation_to_slope(elevation_outfile, slope_outfile)

                high_res_elevation = subsample_geo_tiff(elevation_outfile, dst_tiff)
                high_res_slope = subsample_geo_tiff(slope_outfile, dst_tiff)

                dst_tiff_file = gdal.Open(dst_tiff)
                high_res_geo_reference = dst_tiff_file.GetGeoTransform()

                numpy_array_to_raster(elevation_outfile, high_res_elevation, high_res_geo_reference, 'wgs84')
                numpy_array_to_raster(slope_outfile, high_res_slope, high_res_geo_reference, 'wgs84')

                osm_outfile = os.path.join(SLOPE_DIR, region, district, base_name.replace('multiband.tiff',
                                                                                          'osm.tiff'))

                getOsm(dst_tiff, osm_outfile)

                combine_bands(rgb_comp, dst_tiff)
                combine_bands(ir_comp, dst_tiff)
                combine_bands(osm_outfile, dst_tiff)
                combine_bands(elevation_outfile, dst_tiff)
                combine_bands(slope_outfile, dst_tiff)

        else:
            assert False, "not implemented"


if __name__ == "__main__":
    hack_hypercubes(
        ['Ethiopia/Fafan']
    )
