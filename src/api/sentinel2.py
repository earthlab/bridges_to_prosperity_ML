import getpass
import json

import multiprocessing as mp
import os
import pathlib

import subprocess as sp
from argparse import Namespace
from datetime import datetime, timedelta
from typing import List, Tuple, Union

import boto3
import geojson
import tqdm
from shapely.geometry import Polygon

from definitions import MGRS_INDEX_FILE
from src.utilities.config_reader import CONFIG
from src.utilities.aws import initialize_s3_client, parse_aws_credentials

from file_types import Sentinel2Tile, Sentinel2Cloud

PATH = os.path.dirname(__file__)


def _download_task(namespace: Namespace) -> None:
    """
    Downloads a single file from the indicated s3 bucket. This function is intended to be spawned in parallel from the
    parent process.
    Args:
        namespace (Namespace): Contains the bucket name, s3 file name, and destination required for s3 download request.
        Each value in the namespace must be a pickle-izable type (i.e. str).
    """
    s3 = initialize_s3_client(CONFIG.AWS.BUCKET)
    s3.download_file(namespace.bucket_name, namespace.available_file,
                     namespace.dest,
                     ExtraArgs={'RequestPayer': 'requester'}
                     )


class SinergiseSentinelAPI:
    """
    Contains methods for downloading data from the Sinergise Sentinel2 LC1 data hosted on AWS. On EC2 instances with
    the proper IAM configuration, authentication is handled automatically. From any other platform, users must provide
    credentials.
    Ex.,
        api = SinergiseSentinelAPI()
        api.download([27.876, -45.678, 28.012, -45.165], 1000, '/example/outdir', '2021-01-01', '2021-02-01')
    """
    def __init__(self) -> None:
        """
        Creates a boto3 client object for making requests. If using a CU AWS account the user's identikey can be input
        to create temporary credentials.
        """
        self._bucket_name = 'sentinel-s2-l1c'

        # When the API is not used from an EC2 instance with the proper IAM profile configured credentials need to be
        # created

        self._s3 = initialize_s3_client()

    def download(self, bounds: List[float], buffer: float, region: str, district: str, start_date: str, end_date: str,
                 bands=None) -> None:
        """
        Downloads a list of .jp2 files from the Sinergise Sentinel2 LC1 bucket given a bounding box defined in lat/long,
         a buffer in meters, and a start and end date . Only Bands 2-4 are requested.
         Args:
             bounds (list): Bounding box defined in lat / lon [min_lon, min_lat, max_lon, max_lat]
             buffer (float): Amount by which to extend the bounding box by, in meters
             start_date (str): Beginning of requested data creation date YYYY-MM-DD
             end_date (str): End of requested data creation date YYYY-MM-DD
             bands (list): The bands to download for each file. Default is ['B02', 'B03', 'B04', 'B08'] for R, G, B, and
              near wave IR, respectively
        """
        # Convert the buffer from meters to degrees lat/long at the equator
        if bands is None:
            bands = ['B02', 'B03', 'B04', 'B08']
        buffer /= 111000

        # Adjust the bounding box to include the buffer (subtract from min lat/long values, add to max lat/long values)
        bounds[0] -= buffer
        bounds[1] -= buffer
        bounds[2] += buffer
        bounds[3] += buffer

        available_files = self._find_available_files(bounds, start_date, end_date, bands)

        args = []
        total_data = 0
        for file_info in available_files:
            file_path = file_info[0]
            if '/preview/' in file_path:
                continue

            created_file_path = f"{region}_{district}_{file_path.replace('_qi_', '').replace('/', '_').replace('tiles_', '')}"

            file = Sentinel2Tile.create(created_file_path)
            if file is None:
                file = Sentinel2Cloud.create(created_file_path)
                if file is None:
                    continue
            file.create_archive_dir()

            # Skip if file is already local
            if file.exists:
                continue

            total_data += file_info[1]

            args.append(Namespace(available_file=file_path, bucket_name=self._bucket_name, dest=file.archive_path))

        total_data /= 1E9
        print(f'Found {len(args)} files for download. Total size of files is'
              f' {round(total_data, 2)}GB and estimated cost will be ${round(0.09 * total_data, 2)}'
              )

        with mp.Pool(mp.cpu_count() - 1) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(_download_task, args), total=len(args)):
                pass

    def _lookup_mgrs(self, bbox: List[float]) -> Union[List[str], None]:
        self._create_mgrs_index()
        with open(MGRS_INDEX_FILE, 'r') as f:
            mgrs_index = json.load(f)

        key = [str(num) for num in bbox]
        key = ', '.join(key)

        if key in mgrs_index:
            return mgrs_index[key]
        return None

    def _write_mgrs_index(self, bbox: List[float], mgrs: List[str]) -> None:
        self._create_mgrs_index()
        with open(MGRS_INDEX_FILE, 'r') as f:
            mgrs_index = json.load(f)

        key = [str(num) for num in bbox]
        key = ', '.join(key)

        mgrs_index[key] = mgrs

        with open(MGRS_INDEX_FILE, 'w') as f:
            json.dump(mgrs_index, f)

    @staticmethod
    def _create_mgrs_index():
        if not os.path.exists(MGRS_INDEX_FILE):
            with open(MGRS_INDEX_FILE, 'w+') as f:
                json.dump({}, f)

    def bounds_to_mgrs(self, bounds: List[float]) -> List[str]:
        mgrs_grids = self._lookup_mgrs(bbox=bounds)
        if mgrs_grids is None:
            mgrs_grids = self._find_overlapping_mgrs(bounds)
            self._write_mgrs_index(bounds, mgrs_grids)

        return mgrs_grids

    def _find_available_files(self, bounds: List[float], start_date: str, end_date: str,
                              bands: List[str]) -> List[Tuple[str, str]]:
        """
        Given a bounding box and start / end date, finds which files are available on the bucket and meet the search
        criteria
        Args:
            bounds (list): Lower left and top right corner of bounding box defined in lat / lon [min_lon, min_lat,
            max_lon, max_lat]
            start_date (str): Beginning of requested data creation date YYYY-MM-DD
            end_date (str): End of requested data creation date YYYY-MM-DD
        """
        ref_date = self._str_to_datetime(start_date)
        date_paths = []
        while ref_date <= self._str_to_datetime(end_date):
            tt = ref_date.timetuple()
            date_paths.append(f'/{tt.tm_year}/{tt.tm_mon}/{tt.tm_mday}/')
            ref_date = ref_date + timedelta(days=1)

        info = []
        mgrs_grids = self.bounds_to_mgrs(bounds)
        for grid_string in mgrs_grids:
            utm_code = grid_string[:2]
            latitude_band = grid_string[2]
            square = grid_string[3:5]
            grid = f'tiles/{utm_code}/{latitude_band}/{square}'
            response = self._s3.list_objects_v2(
                Bucket=self._bucket_name,
                Prefix=grid + '/',
                MaxKeys=300,
                RequestPayer='requester'
            )
            if 'Contents' not in list(response.keys()):
                continue

            for date in date_paths:
                response = self._s3.list_objects_v2(
                    Bucket=self._bucket_name,
                    Prefix=grid + date + '0/',  # '0/' is for the sequence, which in most cases will be 0
                    MaxKeys=100,
                    RequestPayer='requester'
                )
                if 'Contents' in list(response.keys()):
                    info += [
                        (v['Key'], v['Size']) for v in response['Contents'] if
                        any([band + '.jp2' in v['Key'] for band in bands]) or 'MSK_CLOUDS_B00.gml' in v['Key']
                    ]

        return info

    @staticmethod
    def _find_overlapping_mgrs(bounds: List[float]) -> List[str]:
        """
        Files in the Sinergise Sentinel2 S3 bucket are organized by which military grid they overlap. Thus, the
        military grids that overlap the input bounding box defined in lat / lon must be found. A lookup table that
        includes each grid name and its geometry is used to find the overlapping grids.
        """
        print('Finding overlapping tiles... ')
        input_bounds = Polygon([(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]),
                                (bounds[0], bounds[3]), (bounds[0], bounds[1])])
        with open(os.path.join(PATH, 'mgrs_lookup.geojson'), 'r') as f:
            ft = geojson.load(f)
            return [i['properties']['mgs'] for i in ft[1:] if
                    input_bounds.intersects(Polygon(i['geometry']['coordinates'][0]))]

    @staticmethod
    def _str_to_datetime(date: str):
        return datetime.strptime(date, '%Y-%m-%d')
