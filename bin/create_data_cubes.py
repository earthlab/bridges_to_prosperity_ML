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
from PIL import Image

from bin.composites_to_tiles import create_tiles
from definitions import REGION_FILE_PATH, COMPOSITE_DIR, TILE_DIR, TRUTH_DIR, SENTINEL_2_DIR, ELEVATION_DIR, SLOPE_DIR,\
    S3_COMPOSITE_DIR
from src.api.sentinel2 import initialize_s3_bucket
from src.utilities.imaging import numpy_array_to_raster
from src.utilities.config_reader import CONFIG
from src.utilities.imaging import elevation_to_slope, subsample_geo_tiff
from src.utilities.coords import get_bridge_locations
from bin.download_sentinel2 import download_sentinel2
from bin.sentinel2_to_composite import sentinel2_to_composite
from bin.download_composites import download_composites, get_requested_locations
from src.api.sentinel2 import SinergiseSentinelAPI
from src.api.lp_daac import Elevation as ElevationAPI
from src.api.osm import getOsm
import mgrs
from geopy import Point
from geopy.distance import distance

from file_types import OpticalComposite, Elevation as ElevationFile, Slope as SlopeFile, OSM as OSMFile,\
    MultiVariateComposite as MultiVariateCompositeFile

CORES = mp.cpu_count() - 1
Image.MAX_IMAGE_PIXELS = None


def combine_bands(source_file: str, target_file: str):
    with rasterio.open(source_file, 'r') as src_file:
        # copy and update the metadata from the input raster for the output
        meta = src_file.meta.copy()
        d = meta['count']
        meta.update(
            count=d + 1
        )
        print(meta)
        with rasterio.open(target_file, 'w+', **meta) as dst:
            for i in range(d):
                dst.write_band(i + 1, src_file.read(i + 1))


def mgrs_to_bbox(mgrs_string: str):
    m = mgrs.MGRS()
    lat, lon = m.toLatLon(mgrs_string)
    # Calculate the bounding box
    sw_point = Point(latitude=lat, longitude=lon)
    ne_point = distance(kilometers=100.0).destination(sw_point, 45)
    bounding_box = (sw_point.longitude, sw_point.latitude, ne_point.longitude, ne_point.latitude)
    return list(bounding_box)


def copy_geo_transform(source_file: str, target_file: str):
    print('Setting geo transform')
    source_tif = gdal.Open(source_file)
    target_tif = gdal.Open(target_file, gdal.GA_Update)
    print(source_tif.GetGeoTransform())
    target_tif.SetGeoTransform(source_tif.GetGeoTransform())

    source_tif = None
    target_tif = None


def create_date_cubes(s3_bucket_name: str = CONFIG.AWS.BUCKET, cores: int = CORES, region: List[str] = None,
                      districts: List[str] = None):
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

        # Download any s2 data that doesn't exist  # TODO: Change this to all bands
        # TODO: Reimplement this
        # s2_dir = os.path.join(SENTINEL_2_DIR, region, district)
        # for date in dates:
        #     sentinel2_api.download(bbox, 100, s2_dir, date[0], date[1], bands=['B08'])

        print('Making ir composites')
        # TODO: Make slices configurable
        sentinel2_to_composite(10, cores, bands=['B08'], region=region, districts=[district])

        # Get elevation and slope if they don't exist
        print('Elevation')
        elevation_file = ElevationFile(region, district)
        os.makedirs(os.path.dirname(elevation_file.archive_path), exist_ok=True)
        if not os.path.exists(elevation_file.archive_path):
            elevation_api.download_district(elevation_file.archive_path, region, district, buffer=100)

        print('Slope')
        slope_file = SlopeFile(region, district)
        os.makedirs(os.path.dirname(slope_file.archive_path), exist_ok=True)
        if not os.path.exists(slope_file.archive_path):
            elevation_to_slope(elevation_file.archive_path, slope_file.archive_path)

        composite_dir = os.path.join(COMPOSITE_DIR, region, district)

        rgb_composites = OpticalComposite.find_files(composite_dir, ['B02', 'B03', 'B04'], recursive=True)
        print(composite_dir)
        print(rgb_composites)
        for rgb_comp in rgb_composites:
            rgb_file = OpticalComposite.create(rgb_comp)

            rgb_tiff_file = gdal.Open(rgb_file.archive_path)
            print(rgb_tiff_file.GetGeoTransform(), 'rgb_gt')

            ir_file = OpticalComposite(region, district, rgb_file.mgrs, ['B08'])
            assert os.path.isfile(ir_file.archive_path), f'IR composite should already exist for {ir_file.archive_path}'

            ir_tiff_file = gdal.Open(ir_file.archive_path)
            print(ir_tiff_file.GetGeoTransform(), 'ir_gt')

            # TODO: Check if all_bands_file exists
            all_bands_file = OpticalComposite(region, district, rgb_file.mgrs, ['B02', 'B03', 'B04', 'B08'])
            combine_bands(rgb_file.archive_path, all_bands_file.archive_path)
            combine_bands(ir_file.archive_path, all_bands_file.archive_path)

            multivariate_file = MultiVariateCompositeFile(region, district, rgb_file.mgrs)
            os.makedirs(os.path.dirname(multivariate_file.archive_path), exist_ok=True)
            all_bands_tiff_file = gdal.Open(all_bands_file.archive_path)
            high_res_geo_reference = all_bands_tiff_file.GetGeoTransform()

            mgrs_elevation_outfile = ElevationFile(region, district, rgb_file.mgrs)
            if not os.path.exists(mgrs_elevation_outfile.archive_path):
                # Clip to bbox so we can convert to meters
                mgrs_bbox = mgrs_to_bbox(rgb_file.mgrs)
                elevation_api.clip(elevation_file.archive_path, mgrs_elevation_outfile.archive_path, mgrs_bbox)
                print('Copying geo transform')
                copy_geo_transform(rgb_file.archive_path, mgrs_elevation_outfile.archive_path)
                high_res_elevation = subsample_geo_tiff(mgrs_elevation_outfile.archive_path,
                                                        all_bands_file.archive_path)
                numpy_array_to_raster(mgrs_elevation_outfile.archive_path, high_res_elevation, high_res_geo_reference,
                                      'wgs84')

            mgrs_slope_outfile = SlopeFile(region, district, rgb_file.mgrs)
            if not os.path.exists(mgrs_slope_outfile.archive_path):
                mgrs_bbox = mgrs_to_bbox(rgb_file.mgrs)
                elevation_api.clip(slope_file.archive_path, mgrs_slope_outfile.archive_path, mgrs_bbox)
                copy_geo_transform(rgb_file.archive_path, mgrs_slope_outfile.archive_path)
                high_res_slope = subsample_geo_tiff(mgrs_slope_outfile.archive_path, all_bands_file.archive_path)
                numpy_array_to_raster(mgrs_slope_outfile.archive_path, high_res_slope, high_res_geo_reference, 'wgs84')

            osm_file = OSMFile(region, district, rgb_file.mgrs)
            os.makedirs(os.path.dirname(osm_file.archive_path), exist_ok=True)

            getOsm(all_bands_file.archive_path, osm_file.archive_path)

            combine_bands(osm_file.archive_path, multivariate_file.archive_path)
            combine_bands(elevation_file.archive_path, multivariate_file.archive_path)
            combine_bands(slope_file.archive_path, multivariate_file.archive_path)


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
    args = parser.parse_args()

    create_date_cubes(s3_bucket_name=args.s3_bucket_name, cores=args.cores, region=args.region,
                      districts=args.districts)
