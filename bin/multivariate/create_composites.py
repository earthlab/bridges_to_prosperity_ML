import argparse
import multiprocessing as mp
import os
from typing import List
from argparse import Namespace

import rasterio
import yaml
from tqdm.contrib.concurrent import process_map
from osgeo import gdal
from PIL import Image

from definitions import REGION_FILE_PATH
from src.utilities.imaging import numpy_array_to_raster
from src.utilities.config_reader import CONFIG
from src.utilities.imaging import elevation_to_slope, subsample_geo_tiff
from bin.optical.download_composites import download_composites
from src.api.sentinel2 import SinergiseSentinelAPI
from src.api.lp_daac import Elevation as ElevationAPI
from src.api.osm import getOsm
from src.utilities.coords import tiff_to_bbox
from bin.optical.create_composites import create_composites as create_optical_composites

from file_types import OpticalComposite, Elevation as ElevationFile, Slope as SlopeFile, OSM as OSMFile,\
    MultiVariateComposite as MultiVariateCompositeFile, File

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


def mgrs_task(args: Namespace):
    elevation_api = ElevationAPI()
    four_band_optical = OpticalComposite.create(args.four_band_optical)
    region = args.region
    district = args.district

    multivariate_file = MultiVariateCompositeFile(region, district, four_band_optical.mgrs)
    if os.path.exists(multivariate_file.archive_path()):
        return

    four_band_tiff = gdal.Open(four_band_optical.archive_path())
    projection = four_band_tiff.GetProjection()
    geo_transform = four_band_tiff.GetGeoTransform()

    # Create elevation file
    up_sample_elevation = False
    mgrs_elevation_outfile = ElevationFile(region, district, four_band_optical.mgrs)
    os.makedirs(os.path.dirname(mgrs_elevation_outfile.archive_path()), exist_ok=True)
    if not os.path.exists(mgrs_elevation_outfile.archive_path()):
        # Clip to bbox so can convert to meters
        bbox = tiff_to_bbox(four_band_optical.archive_path())
        mgrs_bbox = [bbox[3][1], bbox[3][0], bbox[1][1], bbox[1][0]]
        elevation_api.download_bbox(mgrs_elevation_outfile.archive_path(), mgrs_bbox, buffer=5000)
        up_sample_elevation = True

    # Calculate slope from elevation gradient before up-sampling elevation data
    mgrs_slope_outfile = SlopeFile(region, district, four_band_optical.mgrs)
    os.makedirs(os.path.dirname(mgrs_slope_outfile.archive_path()), exist_ok=True)
    if not os.path.exists(mgrs_slope_outfile.archive_path()):
        elevation_to_slope(mgrs_elevation_outfile.archive_path(), mgrs_slope_outfile.archive_path())
        elevation_api.lat_lon_to_meters(mgrs_slope_outfile.archive_path())
        high_res_slope = subsample_geo_tiff(mgrs_slope_outfile.archive_path(), four_band_optical.archive_path())
        numpy_array_to_raster(mgrs_slope_outfile.archive_path(), high_res_slope, geo_transform, projection)

    # Up-sample elevation if it was just made
    if up_sample_elevation:
        elevation_api.lat_lon_to_meters(mgrs_elevation_outfile.archive_path())
        high_res_elevation = subsample_geo_tiff(mgrs_elevation_outfile.archive_path(), four_band_optical.archive_path)
        numpy_array_to_raster(mgrs_elevation_outfile.archive_path(), high_res_elevation, geo_transform, projection)

    # Create the OSM file
    osm_file = OSMFile(region, district, four_band_optical.mgrs)
    os.makedirs(os.path.dirname(osm_file.archive_path()), exist_ok=True)
    getOsm(four_band_optical.archive_path, osm_file.archive_path())

    # Combine the files into the multivariate file
    multivariate_file = MultiVariateCompositeFile(region, district, four_band_optical.mgrs)
    os.makedirs(os.path.dirname(multivariate_file.archive_path()), exist_ok=True)
    combine_bands(four_band_optical.archive_path, multivariate_file.archive_path(), new_bands=4)  # Bands 1, 2, 3, 4
    combine_bands(osm_file.archive_path(), multivariate_file.archive_path(), new_bands=2)  # Bands 5, 6
    combine_bands(mgrs_elevation_outfile.archive_path(), multivariate_file.archive_path(), new_bands=1)  # Band 7
    combine_bands(mgrs_slope_outfile.archive_path(), multivariate_file.archive_path(), new_bands=1)  # Band 8


def create_date_cubes(s3_bucket_name: str = CONFIG.AWS.BUCKET, cores: int = CORES, slices: int = 6,
                      region: List[str] = None, districts: List[str] = None, mgrs: List[str] = None):
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

    for location in requested_locations:
        region = location[0]
        district = location[1]
        bbox = region_info[region]['districts'][district]['bbox']
        mgrs_grids = sentinel2_api.bounds_to_mgrs(bbox) if mgrs is None else mgrs

        # Load composites from S3
        download_composites(file_type=FileType.OPTICAL_COMPOSITE, region=region, districts=[district],
                            s3_bucket_name=s3_bucket_name, cores=cores, mgrs=mgrs_grids)

        task_args = []
        for mgrs_grid in mgrs_grids:
            four_band_optical = OpticalComposite(region, district, military_grid=mgrs_grid,
                                                 bands=['B02', 'B03', 'B04', 'B08'])
            if not four_band_optical.exists:
                rgb_optical = OpticalComposite(region, district, military_grid=mgrs_grid, bands=['B02', 'B03', 'B04'])
                ir_optical = OpticalComposite(region, district, military_grid=mgrs_grid, bands=['B08'])

                if not rgb_optical.exists and not ir_optical.exists:
                    create_optical_composites(region, bands=['B02', 'B03', 'B04', 'B08'], mgrs=mgrs_grid,
                                              districts=[district], buffer=0, slices=slices,n_cores=cores)
                else:
                    if not rgb_optical:
                        create_optical_composites(region, bands=['B02', 'B03', 'B04'], mgrs=mgrs_grid,
                                                  districts=[district], buffer=0, slices=slices, n_cores=cores)
                    elif not ir_optical:
                        create_optical_composites(region, bands=['B08'], mgrs=mgrs_grid, districts=[district], buffer=0,
                                                  slices=slices, n_cores=cores)
                    combine_bands(rgb_optical.archive_path(), four_band_optical.archive_path(), 3)
                    combine_bands(ir_optical.archive_path(), four_band_optical.archive_path(), 1)

            task_args.append(Namespace(four_band_optical=four_band_optical, region=region, district=district))

        process_map(
            mgrs_task,
            task_args,
            max_workers=cores
        )


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
    parser.add_argument(
        '--mgrs',
        '-m',
        type=str,
        nargs='+',
        required=False,
        help='Name of the mgrs tiles to make tiles for'
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
                      districts=args.districts, mgrs=args.mgrs)
