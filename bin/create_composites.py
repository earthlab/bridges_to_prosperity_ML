"""
Creates the multivariate composites for input regions, districts, and mgrs tiles. Multivariate composites are comprised of 
Sentinel2 red, green, blue, infrared, osm water, osm admin boundaries, elevation, and slope data respectively for a single region. 
"""
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
import numpy as np

from definitions import REGION_FILE_PATH
from src.utilities.imaging import numpy_array_to_raster
from src.utilities.config_reader import CONFIG
from src.utilities.imaging import elevation_to_slope, subsample_geo_tiff
from src.api.sentinel2 import SinergiseSentinelAPI
from src.api.lp_daac import Elevation as ElevationAPI
from src.api.osm import create_osm_composite
from src.utilities.coords import tiff_to_bbox
from src.utilities.aws import list_s3_files, s3_download_task
from src.utilities.optical_composites import create_composites as create_optical_composites

from file_types import OpticalComposite, Elevation as ElevationFile, Slope as SlopeFile,\
    MultiVariateComposite as MultiVariateCompositeFile

CORES = mp.cpu_count() - 1
Image.MAX_IMAGE_PIXELS = None


def download_optical_composites(region: str, district: str, s3_bucket_name: str, cores: int, mgrs: List[str],
                                bands: List[str]):
    s3_files = list_s3_files(OpticalComposite.ROOT_S3_DIR, s3_bucket_name)
    files_to_download = OpticalComposite.filter_files(s3_files, region, district, bands, mgrs)
    
    parallel_inputs = []
    for i, s3_file_path in enumerate(np.array_split(files_to_download, cores)):
        file_class = OpticalComposite.create(s3_file_path)
        parallel_inputs.append((s3_file_path, file_class.archive_path, str(i + 1), s3_bucket_name))
    process_map(
        s3_download_task,
        parallel_inputs,
        max_workers=cores
    )


def combine_bands(source_file_path: str, target_file_path: str, new_bands: int) -> None:
    """
    Adds the bands from the source file to the target file. Must specify the number of new bands that will be added
    to the target file. Source bands will be appended to the end of the existing bands in the target file.
    If target file does not yet exist it will be created.
    Args:
        source_file_path (str): Path to the tiff file with bands that will be added to the target file
        target_file_path (str): Path to the tiff file that will have the source file bands added to it
        new_bands (int): Number of bands from the source file that will be added to the target file
    """
    if os.path.exists(target_file_path):
        with rasterio.open(target_file_path, 'r') as target_f:
            existing_band_count = target_f.meta['count']
            with rasterio.open(source_file_path, 'r') as src_file:
                # copy and update the metadata from the input raster for the output
                meta = src_file.meta.copy()
                meta.update(
                    count=existing_band_count + new_bands
                )
                with rasterio.open(target_file_path, 'w+', **meta) as dst:
                    for i in range(existing_band_count + new_bands):
                        if i + 1 > existing_band_count:
                            dst.write_band(i + 1, src_file.read(i + 1 - existing_band_count))
                        else:
                            dst.write_band(i + 1, target_f.read(i + 1))
    else:
        existing_band_count = 0
        with rasterio.open(source_file_path, 'r') as src_file:
            # copy and update the metadata from the input raster for the output
            meta = src_file.meta.copy()
            meta.update(
                count=existing_band_count + new_bands
            )
            with rasterio.open(target_file_path, 'w+', **meta) as dst:
                for i in range(new_bands):
                    dst.write_band(i + 1, src_file.read(i+1))


def mgrs_task(args: Namespace) -> None:
    """
    Task to be run in parallel for creating a multivariate composite for a single region, district, and mgrs tile. 
    Multivariate composites are comprised of Sentinel2 red, green, blue, infrared, osm water, osm admin boundaries, elevation, and slope data bands respectively
    Args:
        args.region (str): Region to create the multivariate tile for 
        args.district (str): District to create the multivariate tile for
        args.four_band_optical (str): Path to the four band (r,g,b,ir) optical composite to be used as the base of the multivariate composite
    """
    elevation_api = ElevationAPI()
    four_band_optical = OpticalComposite.create(args.four_band_optical)
    region = args.region
    district = args.district

    four_band_tiff = gdal.Open(args.four_band_optical)
    projection = four_band_tiff.GetProjection()
    geo_transform = four_band_tiff.GetGeoTransform()

    # Create elevation file
    up_sample_elevation = False
    mgrs_elevation_outfile = ElevationFile(region, district, four_band_optical.mgrs)
    mgrs_elevation_outfile.create_archive_dir()
    if not mgrs_elevation_outfile.exists:
        # Clip to bbox so can convert to meters
        bbox = tiff_to_bbox(args.four_band_optical)
        mgrs_bbox = [bbox[3][1], bbox[3][0], bbox[1][1], bbox[1][0]]
        elevation_api.download_bbox(mgrs_elevation_outfile.archive_path, mgrs_bbox, buffer=5000)
        up_sample_elevation = True

    # Calculate slope from elevation gradient before up-sampling elevation data
    mgrs_slope_outfile = SlopeFile(region, district, four_band_optical.mgrs)
    mgrs_slope_outfile.create_archive_dir()
    if not mgrs_slope_outfile.exists:
        elevation_to_slope(mgrs_elevation_outfile.archive_path, mgrs_slope_outfile.archive_path)
        elevation_api.lat_lon_to_meters(mgrs_slope_outfile.archive_path)
        high_res_slope = subsample_geo_tiff(mgrs_slope_outfile.archive_path, args.four_band_optical)
        numpy_array_to_raster(mgrs_slope_outfile.archive_path, high_res_slope, geo_transform, projection)

    # Up-sample elevation if it was just made
    if up_sample_elevation:
        elevation_api.lat_lon_to_meters(mgrs_elevation_outfile.archive_path)
        high_res_elevation = subsample_geo_tiff(mgrs_elevation_outfile.archive_path, args.four_band_optical)
        numpy_array_to_raster(mgrs_elevation_outfile.archive_path, high_res_elevation, geo_transform, projection)

    # Create the OSM file
    osm_file = create_osm_composite(four_band_optical)

    # Combine the files into the multivariate file
    multivariate_file = MultiVariateCompositeFile(region, district, four_band_optical.mgrs)
    multivariate_file.create_archive_dir()
    combine_bands(args.four_band_optical, multivariate_file.archive_path, new_bands=4)  # Bands 1, 2, 3, 4
    combine_bands(osm_file.archive_path, multivariate_file.archive_path, new_bands=2)  # Bands 5, 6
    combine_bands(mgrs_elevation_outfile.archive_path, multivariate_file.archive_path, new_bands=1)  # Band 7
    combine_bands(mgrs_slope_outfile.archive_path, multivariate_file.archive_path, new_bands=1)  # Band 8


def create_multivariate_compsites(s3_bucket_name: str = CONFIG.AWS.BUCKET, cores: int = CORES, slices: int = 6,
                                  regions: List[str] = None, districts: List[str] = None, mgrs: List[str] = None) -> None:
    """
    Creates multivariate composites comprised of Sentinel2 red, green, blue, infrared, osm water, osm admin boundaries, elevation, and slope data respectively for a single region.
    Specific districts and mgrs tiles can be specified.
    Args:
        s3_bucket_name (str): Name of the s3 bucket with any potentially existing sentinel2 optical composites. Default is config bucket.
        cores (int): Number of cores to use for parallel processing. Default is cpu_count - 1
        slices (int): Number of slices to use when creating optical composites. The more slices the less memory will be used at a time
        regions (list): The regions to make the composites for. Must be in the regions_info.yaml file. Defualt is all the regions in the regions_info.yaml file
        districts (list): The districts to make the composites for in each region. Must be in the regions_info.yaml file for at least on region. Defualt is all districts for every specified region
        mgrs (list): The military grid tiles to make the composites for. Defualt is all the mgrs tiles for each specified region / district combination
    """
    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)

    requested_locations = []
    if regions is None:
        regions = []
        for region in region_info:
            regions.append(region)

    for region in regions:
        for district in region_info[region]['districts']:
            if districts is None or district in districts:
                requested_locations.append((region, district))
    
    sentinel2_api = SinergiseSentinelAPI()

    for location in requested_locations:
        region = location[0]
        district = location[1]

        bbox = region_info[region]['districts'][district]['bbox']
        mgrs_grids = sentinel2_api.bounds_to_mgrs(bbox) if mgrs is None else mgrs

        # Load any 4-band optical composites from S3
        download_optical_composites(region=region, district=district, s3_bucket_name=s3_bucket_name, cores=cores, mgrs=mgrs_grids,
                                    bands=['B02', 'B03', 'B04', 'B08'])
        
        # Load any RGB optical composites from S3
        download_optical_composites(region=region, district=district, s3_bucket_name=s3_bucket_name, cores=cores, mgrs=mgrs_grids,
                                    bands=['B02', 'B03', 'B04'])
        
        # Load any IR optical composites from S3
        download_optical_composites(region=region, district=district, s3_bucket_name=s3_bucket_name, cores=cores, mgrs=mgrs_grids,
                                    bands=['B08'])

        task_args = []
        for mgrs_grid in mgrs_grids:
            # First check if multivariate composite already exists locally
            multivariate_file = MultiVariateCompositeFile(region, district, mgrs_grid)
            if multivariate_file.exists:
                continue

            four_band_optical = OpticalComposite(region, district, military_grid=mgrs_grid,
                                                 bands=['B02', 'B03', 'B04', 'B08'])
            if not four_band_optical.exists:
                four_band_optical.create_archive_dir()
                rgb_optical = OpticalComposite(region, district, military_grid=mgrs_grid, bands=['B02', 'B03', 'B04'])
                ir_optical = OpticalComposite(region, district, military_grid=mgrs_grid, bands=['B08'])

                if not rgb_optical.exists and not ir_optical.exists:
                    create_optical_composites(region, bands=['B02', 'B03', 'B04', 'B08'], mgrs=mgrs_grid,
                                              districts=[district], buffer=0, slices=slices, n_cores=cores)
                else:
                    if not rgb_optical.exists:
                        create_optical_composites(region, bands=['B02', 'B03', 'B04'], mgrs=mgrs_grid,
                                                  districts=[district], buffer=0, slices=slices, n_cores=cores)
                    else:
                        create_optical_composites(region, bands=['B08'], mgrs=mgrs_grid, districts=[district], buffer=0,
                                                  slices=slices, n_cores=cores)
                    combine_bands(rgb_optical.archive_path, four_band_optical.archive_path, 3)
                    combine_bands(ir_optical.archive_path, four_band_optical.archive_path, 1)

            task_args.append(Namespace(four_band_optical=four_band_optical.archive_path, region=region, district=district))

        process_map(
            mgrs_task,
            task_args,
            max_workers=cores
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', '-r', type=str, required=False, nargs='+', help='Name of the composite region (Ex Uganda). Default is all regions in region_info.yaml file')
    parser.add_argument('--districts', '-d', type=str, nargs='+', required=False, help='Name of the composite district(s) (Ex. Fafan Ibanda). Default is all districts for region.')
    parser.add_argument('--mgrs', '-m', type=str, nargs='+', required=False, help='Name of the mgrs tiles to make tiles for. Default is all mgrs for each district')
    parser.add_argument('--s3_bucket_name', '-b', required=False, default=CONFIG.AWS.BUCKET, type=str, help='Name of s3 bucket to search for composites in. Default is from project config, which is currently set to {CONFIG.AWS.BUCKET}')
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='Number of cores to use when making tiles in parallel. Default is cpu_count - 1')
    parser.add_argument('--slices', required=False, type=int, default=12,
                        help='Number of slices to use when making composites. Default is 12')
    args = parser.parse_args()

    create_multivariate_compsites(s3_bucket_name=args.s3_bucket_name, cores=args.cores, slices=args.slices, regions=args.regions,
                                  districts=args.districts, mgrs=args.mgrs)
