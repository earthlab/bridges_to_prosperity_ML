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
from osgeo import gdal, osr
from PIL import Image

from bin.composites_to_tiles import create_tiles
from definitions import REGION_FILE_PATH, COMPOSITE_DIR, TILE_DIR, TRUTH_DIR, SENTINEL_2_DIR, ELEVATION_DIR, SLOPE_DIR,\
    S3_COMPOSITE_DIR
from src.api.sentinel2 import initialize_s3_bucket
from src.utilities.imaging import numpy_array_to_raster, mgrs_to_bbox, get_utm_epsg
from src.utilities.config_reader import CONFIG
from src.utilities.imaging import elevation_to_slope, subsample_geo_tiff
from src.utilities.coords import get_bridge_locations
from bin.download_sentinel2 import download_sentinel2
from bin.sentinel2_to_composite import sentinel2_to_composite
from bin.download_composites import download_composites, get_requested_locations
from src.api.sentinel2 import SinergiseSentinelAPI
from src.api.lp_daac import Elevation as ElevationAPI
from src.api.osm import getOsm

from file_types import OpticalComposite, Elevation as ElevationFile, Slope as SlopeFile, OSM as OSMFile,\
    MultiVariateComposite as MultiVariateCompositeFile

CORES = mp.cpu_count() - 1
Image.MAX_IMAGE_PIXELS = None


def combine_bands(source_file: str, target_file: str, new_bands: int):
    if os.path.exists(target_file):
        with rasterio.open(target_file, 'r') as target_f:
            existing_band_count = target_f.meta['count']
            with rasterio.open(source_file, 'r') as src_file:
                # copy and update the metadata from the input raster for the output
                meta = src_file.meta.copy()
                meta.update(
                    count=existing_band_count + new_bands
                )
                with rasterio.open(target_file, 'w+', **meta) as dst:
                    for i in range(existing_band_count + new_bands):
                        if i + 1 > existing_band_count:
                            dst.write_band(i + 1, src_file.read(i + 1 - existing_band_count))
                        else:
                            dst.write_band(i + 1, target_f.read(i + 1))
    else:
        existing_band_count = 0
        with rasterio.open(source_file, 'r') as src_file:
            # copy and update the metadata from the input raster for the output
            meta = src_file.meta.copy()
            meta.update(
                count=existing_band_count + new_bands
            )
            with rasterio.open(target_file, 'w+', **meta) as dst:
                for i in range(new_bands):
                    dst.write_band(i + 1, src_file.read(i+1))


def create_date_cubes(s3_bucket_name: str = CONFIG.AWS.BUCKET, cores: int = CORES, slices: int = 6,
                      region: List[str] = None, districts: List[str] = None):
    debug = True

    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    requested_locations = []
    if region is None:
        for region in region_info:
            for district in region_info[region]['districts']:
                requested_locations.append((region, district))

    elif districts is None:
        for district in region_info[region]['districts']:
            requested_locations.append((region, district))

    else:
        for district in districts:
            requested_locations.append((region, district))

    sentinel2_api = SinergiseSentinelAPI()
    elevation_api = ElevationAPI()

    for location in requested_locations:
        region = location[0]
        district = location[1]
        bbox = region_info[region]['districts'][district]['bbox']
        dates = region_info[region]['dates']

        # Load composites from S3
        download_composites(region, [district], s3_bucket_name, cores)

        # Download any s2 data that doesn't exist
        # TODO: Reimplement this
        # s2_dir = os.path.join(SENTINEL_2_DIR, region, district)
        # for date in dates:
        #     sentinel2_api.download(bbox, 100, s2_dir, date[0], date[1], bands=['B08'])

        print('Making ir composites')
        sentinel2_to_composite(slices, cores, bands=['B08'], region=region, districts=[district])

        composite_dir = os.path.join(COMPOSITE_DIR, region, district)

        rgb_composites = OpticalComposite.find_files(composite_dir, ['B02', 'B03', 'B04'], recursive=True)
        print(composite_dir)
        print(rgb_composites)
        for rgb_comp in rgb_composites:
            rgb_file = OpticalComposite.create(rgb_comp)

            multivariate_file = MultiVariateCompositeFile(region, district, rgb_file.mgrs)
            if os.path.exists(multivariate_file.archive_path):
                continue

            rgb_tiff_file = gdal.Open(rgb_file.archive_path)
            print(rgb_tiff_file.GetGeoTransform(), 'rgb_gt')

            ir_file = OpticalComposite(region, district, rgb_file.mgrs, ['B08'])
            assert os.path.isfile(ir_file.archive_path), f'IR composite should already exist for {ir_file.archive_path}'

            ir_tiff_file = gdal.Open(ir_file.archive_path)
            print(ir_tiff_file.GetGeoTransform(), 'ir_gt')

            # TODO: Check if all_bands_file exists
            all_bands_file = OpticalComposite(region, district, rgb_file.mgrs, ['B02', 'B03', 'B04', 'B08'])
            if not os.path.exists(all_bands_file.archive_path):
                combine_bands(rgb_file.archive_path, all_bands_file.archive_path, new_bands=3)
                combine_bands(ir_file.archive_path, all_bands_file.archive_path, new_bands=1)

            mgrs_lat_lon_bbox = mgrs_to_bbox(all_bands_file.mgrs)
            epsg_code = get_utm_epsg(mgrs_lat_lon_bbox[3], mgrs_lat_lon_bbox[0])
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(epsg_code)
            projection = proj.ExportToWkt()

            all_bands_tiff_file = gdal.Open(all_bands_file.archive_path)
            high_res_geo_reference = all_bands_tiff_file.GetGeoTransform()

            mgrs_elevation_outfile = ElevationFile(region, district, rgb_file.mgrs)
            os.makedirs(os.path.dirname(mgrs_elevation_outfile.archive_path), exist_ok=True)
            if not os.path.exists(mgrs_elevation_outfile.archive_path):
                # Clip to bbox so we can convert to meters
                mgrs_bbox = mgrs_to_bbox(rgb_file.mgrs)
                elevation_api.download_bbox(mgrs_elevation_outfile.archive_path, mgrs_bbox, buffer=5000)
                elevation_api.lat_lon_to_meters(mgrs_elevation_outfile.archive_path)
                high_res_elevation = subsample_geo_tiff(mgrs_elevation_outfile.archive_path,
                                                        all_bands_file.archive_path)
                numpy_array_to_raster(mgrs_elevation_outfile.archive_path, high_res_elevation, high_res_geo_reference,
                                      projection)

            mgrs_slope_outfile = SlopeFile(region, district, rgb_file.mgrs)
            os.makedirs(os.path.dirname(mgrs_slope_outfile.archive_path), exist_ok=True)
            if not os.path.exists(mgrs_slope_outfile.archive_path):
                elevation_to_slope(mgrs_elevation_outfile.archive_path, mgrs_slope_outfile.archive_path)

                high_res_slope = subsample_geo_tiff(mgrs_slope_outfile.archive_path,
                                                    all_bands_file.archive_path)
                numpy_array_to_raster(mgrs_slope_outfile.archive_path, high_res_slope, high_res_geo_reference,
                                      projection)

            osm_file = OSMFile(region, district, rgb_file.mgrs)
            os.makedirs(os.path.dirname(osm_file.archive_path), exist_ok=True)

            getOsm(all_bands_file.archive_path, osm_file.archive_path)

            multivariate_file = MultiVariateCompositeFile(region, district, rgb_file.mgrs)
            os.makedirs(os.path.dirname(multivariate_file.archive_path), exist_ok=True)
            combine_bands(all_bands_file.archive_path, multivariate_file.archive_path, new_bands=4)  # 1, 2, 3, 4
            combine_bands(osm_file.archive_path, multivariate_file.archive_path, new_bands=2)  # 5, 6
            combine_bands(mgrs_elevation_outfile.archive_path, multivariate_file.archive_path, new_bands=1)  # 7
            combine_bands(mgrs_slope_outfile.archive_path, multivariate_file.archive_path, new_bands=1)  # 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--region',
        '-r',
        type=str,
        required=False,
        help='Name of the composite region (Ex Uganda)'
    )
    parser.add_argument(
        '--districts',
        '-d',
        type=str,
        nargs='+',
        required=False,
        help='Name of the composite district (Ex. Fafan Ibanda)'
    )
    parser.add_argument('--s3_bucket_name', '-b', required=False, default=CONFIG.AWS.BUCKET, type=str,
                        help='Name of s3 bucket to search for composites in. Default is from project config, which is'
                             f' currently set to {CONFIG.AWS.BUCKET}')
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='Number of cores to use when making tiles in parallel. Default is cpu_count - 1')
    parser.add_argument('--slices', required=False, type=int, default=6,
                        help='Number of slices to use when making composites. Default is 6')
    args = parser.parse_args()

    create_date_cubes(s3_bucket_name=args.s3_bucket_name, cores=args.cores, slices=args.slices, region=args.region,
                      districts=args.districts)
