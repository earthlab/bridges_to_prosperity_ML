"""
Classes wrapping http requests to LP DAAC servers for SRTMGL1 v003 elevation data and NASADEM_SC v001 slope and
curvature data.
"""

import getpass
import math
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import urllib
import zipfile
from abc import ABC
from argparse import Namespace
from http.cookiejar import CookieJar
from typing import Tuple, List, Any
from src.utilities.imaging import get_utm_epsg

import numpy as np
import rasterio
import yaml
from netCDF4 import Dataset
from osgeo import gdal, gdalconst
from osgeo import osr, ogr
from rasterio.merge import merge
from tqdm import tqdm

from definitions import B2P_DIR, REGION_FILE_PATH
from src.api.util import generate_secrets_file


def _base_download_task(task_args: Namespace) -> None:
    """
    Downloads data from the LP DAAC servers. Authentication is established using NASA EarthData username and password.
    Args:
        task_args (Namespace): Contains attributes required for requesting data from LP DAAC
            task_args.link (str): URL to datafile on servers
            task_args.username (str): NASA EarthData username
            task_args.password (str): NASA EarthData password
            task_args.dest (str): Path to where the data will be written
    """
    link = task_args.link

    pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    pm.add_password(None, "https://urs.earthdata.nasa.gov", task_args.username, task_args.password)
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(pm),
        urllib.request.HTTPCookieProcessor(cookie_jar)
    )
    urllib.request.install_opener(opener)
    myrequest = urllib.request.Request(link)
    response = urllib.request.urlopen(myrequest)
    response.begin()

    with open(task_args.dest, 'wb') as fd:
        while True:
            chunk = response.read()
            if chunk:
                fd.write(chunk)
            else:
                break


# Functions for elevation parallel data processing


def _elevation_download_task(task_args: Namespace) -> None:
    """
    Downloads SRTMGL1 v003 slope data from LP DAAC servers. Files are downloaded as netCDF files and are thus
    converted to tif files so file structures are uniform throughout the api
    Args:
        task_args (Namespace): Contains attributes required for requesting data from EarthData servers
            task_args.link (str): URL to datafile on servers
            task_args.out_dir (str): Path to directory where files will be written
            task_args.username (str): NASA EarthData username
            task_args.password (str): NASA EarthData password
            task_args.top_left_coord (list): List giving coordinate of top left of image [lon, lat] so tif file can be
            geo-referenced
    """
    dest = os.path.join(task_args.out_dir, os.path.basename(task_args.link))
    task_args.dest = dest
    _base_download_task(task_args)

    _nc_to_tif(task_args.dest, task_args.top_left_coord, task_args.out_dir)

    # Just want the .tif file at the end
    os.remove(task_args.dest)


def _nc_to_tif(nc_path: str, top_left_coord: Tuple[float, float], out_dir: str,
               cell_resolution: float = 0.000277777777777778) -> None:
    """
    Converts netCDF files to tif file. Band that will be transferred to tif file needs to be extracted and
    geo-referencing needs to be added as well.
    Args:
        nc_path (str): Path to input netCDF file
        top_left_coord (tuple): List giving coordinate of top left of image [lon, lat] so tif file can be geo-referenced
        out_dir (str): Path to directory where tif file will be written
        cell_resolution (float): Spatial resolution of tif file in degrees
    """
    # Open the netCDF file
    nc_file = Dataset(nc_path)

    # Read the data and metadata from the netCDF file
    var = nc_file.variables['SRTMGL1_DEM'][:].squeeze()
    x = nc_file.variables['lon'][:]
    y = nc_file.variables['lat'][:]
    crs = nc_file.variables['crs'].spatial_ref

    # Define the output GeoTIFF file
    tif_path = os.path.join(out_dir, os.path.basename(nc_path).replace('.nc', '.tif'))

    # Create a new GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    tif_dataset = driver.Create(tif_path, len(x), len(y), 1, gdal.GDT_Float32)

    # Set the projection and transform of the GeoTIFF file
    proj = osr.SpatialReference()
    proj.ImportFromWkt(str(crs))
    tif_dataset.SetProjection(proj.ExportToWkt())
    tif_dataset.SetGeoTransform(
        (top_left_coord[0],
         cell_resolution,
         0,
         top_left_coord[1],
         0,
         -1 * cell_resolution)
    )

    # Write the data to the GeoTIFF file
    tif_band = tif_dataset.GetRasterBand(1)
    tif_band.WriteArray(var)

    # Close the GeoTIFF file and netCDF file
    tif_dataset = None
    nc_file.close()


class BaseAPI(ABC):
    """
    Defines all the attributes and methods common to the child APIs. NASA EarthData credentials are queried for and
    written to secrets.yaml file if it is not found
    """

    BASE_URL = None

    def __init__(self):
        """
        Initializes the NASA EarthData credentials
        """
        self._username, self._password = self._get_auth_credentials()


# TODO: Write unit tests!!!
class Elevation(BaseAPI):
    """
    SRTMGL1 v003 elevation data specific methods for requesting data from LP DAAC servers. Files are downloaded in
    parallel as netCDF files and then converted to tif format. Tif files are then made into a single tif mosaic covering
    the entire requested region.
    Data homepage: https://lpdaac.usgs.gov/products/srtmgl1v003/
    Ex.
        elevation = Elevation()
        elevation.download_district('data/elevation/ethiopia/yem.tif', 'Ethiopia', 'Yem')
        OR
        elevation.download_bbox('data/elevation/ethiopia/yem.tif', [37.391, 7.559, 37.616, 8.012])
    """
    BASE_URL = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1_NC.003/2000.02.11/'

    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_auth_credentials() -> Tuple[str, str]:
        """
        Ask the user for their urs.earthdata.nasa.gov username and login if secrets.yaml file is not found in project
        root. After the first query, the secrets.yaml will be created for any subsequent initializations
        Returns:
            username (str): urs.earthdata.nasa.gov username
            password (str): urs.earthdata.nasa.gov password
        """
        secrets_file_path = os.path.join(B2P_DIR, 'secrets.yaml')
        if not os.path.exists(secrets_file_path):
            print(f'Please input your earthdata.nasa.gov username and password. If you do not have one, '
                  f'you can register here: https://urs.earthdata.nasa.gov/users/new . Credentials will be cached for '
                  f'any subsequent initializations at {secrets_file_path}')
            username = input('Username:')
            password = getpass.getpass('Password:', stream=None)
            generate_secrets_file(nasaearthdata_u=username, nasaearthdata_p=password)

        with open(secrets_file_path, 'r') as f:
            secrets = yaml.load(f, Loader=yaml.FullLoader)

        username = secrets['nasaearthdata']['username']
        password = secrets['nasaearthdata']['password']

        if username == '' or password == '':
            username = input('Username:')
            password = getpass.getpass('Password:', stream=None)
            secrets['nasaearthdata']['username'] = username
            secrets['nasaearthdata']['password'] = password
            with open(secrets_file_path, 'w') as f:
                yaml.dump(secrets, f)

        return username, password

    @staticmethod
    def _create_substrings(min_deg: int, max_deg: int, min_ord: str, max_ord: str, padding: int) -> List[str]:
        """
        The file conventions for the data files give information about the location of the data in each file. For
        example, an elevation file named like N00E003.SRTMGL1_NC.nc will be a 1x1 deg North pointing area with its
        bottom left pixel situated at N0 E003. Thus, given the min and max value and ordinal for either lat or lon
        this function will create all ranges of substrings in the range.
        Args:
            min_deg (int): The minimum value of the range in degrees
            max_deg (int): The maximum value of the range in degrees
            min_ord (str): The ordinal direction (n,e,s,w) of the minimum degree value
            max_ord (str): The ordinal direction (n,e,s,w) of the maximum degree value
            padding (int): The amount of zero padding to add to substring values. For lat padding is 2, 3 for lon
        Returns:
            substrings (List): List of substrings that are within range of [min_deg, max_deg]
        """
        substrings = []
        format_str = '{0:03d}' if padding == 3 else '{0:02d}'
        if min_ord == max_ord:
            abs_min = min(min_deg, max_deg)
            abs_max = max(min_deg, max_deg)
            deg_range = np.arange(abs_min, abs_max + (1 if min_ord in ['s', 'w'] else 0), 1)
            for deg in deg_range:
                substrings.append(min_ord + format_str.format(deg))
        else:
            # Only other combo would be min_lon_ord is w and max_lon_ord is e
            neg_range = np.arange(1, min_deg + 1, 1)
            pos_range = np.arange(0, max_deg, 1)
            for deg in neg_range:
                substrings.append(min_ord + format_str.format(deg))
            for deg in pos_range:
                substrings.append(max_ord + format_str.format(deg))

        return substrings

    @staticmethod
    def _mosaic_tif_files(input_dir: str, output_file: str) -> None:
        """
        Creates a mosaic from a group of tif files in the specified input_dir
        Args:
            input_dir (str): The path to the input dir containing the tif files to be mosaiced
            output_file (str): Path to where the output mosaic file will be written
        """
        # Specify the input directory containing the TIFF files
        tiff_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                      f.endswith('.tif')]  # Open all the TIFF files using rasterio
        src = None
        src_files_to_mosaic = []
        for file in tiff_files:
            src = rasterio.open(file)
            src_files_to_mosaic.append(src)  # Merge the TIFF files using rasterio.merge
        mosaic, out_trans = merge(src_files_to_mosaic)  # Specify the output file path and name

        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans
                         })  # Write the merged TIFF file to disk using rasterio
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)

    def download_bbox(self, out_file: str, bbox: List[float], buffer: float = 0) -> None:
        """
        Downloads data given a region and district. Bounding box will be read in from the region_info.yaml file in the
        root data directory. If the region or district is not found an error will be raised.
        Args:
            out_file (str): Path to where the output mosaic tif file of requested data will be written to
            bbox (list): Bounding box defining area of data request in [min_lon, min_lat, max_lon, max_lat]
            buffer (float): Buffer to add to the bounding box in meters
        """
        temp_dir = self._download_bbox(bbox, buffer)

        # 2) Create a composite of all tiffs in the temp_dir
        self._mosaic_tif_files(temp_dir, output_file=out_file)

        # 3) Cleanup
        shutil.rmtree(temp_dir)

    def download_district(self, out_file: str, region: str, district: str, buffer: float = 0) -> None:
        """
        Downloads data given a region and district. Bounding box will be read in from the region_info.yaml file in the
        root data directory. If the region or district is not found an error will be raised.
        Args:
            out_file (str): Path to where the output mosaic tif file of requested data will be written to
            region (str): Name of the region in the region_info.yaml file
            district (str): Name of the district in the specified region's entry in the region_info.yaml file
            buffer (float): Buffer to add to the bounding box in meters
        """
        with open(REGION_FILE_PATH, 'r') as f:
            region_file_info = yaml.load(f, Loader=yaml.FullLoader)

        if region not in region_file_info:
            raise ValueError(f'No region {region} in {REGION_FILE_PATH}')
        region_info = region_file_info[region]

        if district not in region_info['districts']:
            raise ValueError(f'No district {district} found for region {region}')

        bbox = region_info['districts'][district]['bbox']
        temp_dir = self._download_bbox(bbox, buffer)

        # 2) Create a composite of all tiffs in the temp_dir
        self._mosaic_tif_files(temp_dir, output_file=out_file)

        # 3) Cleanup
        shutil.rmtree(temp_dir)

    def lat_lon_to_meters(self, input_tiff: str):
        input_tiff_file = gdal.Open(input_tiff, gdal.GA_Update)

        src_crs = osr.SpatialReference()
        src_crs.ImportFromEPSG(4326)  # Lat / lon

        geo_transform = input_tiff_file.GetGeoTransform()
        print(geo_transform, 'gglatlon')
        dst_epsg = get_utm_epsg(geo_transform[3], geo_transform[0])
        dst_crs = osr.SpatialReference()
        dst_crs.ImportFromEPSG(dst_epsg)

        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(geo_transform[3], geo_transform[0])
        transform = osr.CoordinateTransformation(src_crs, dst_crs)
        point.Transform(transform)
        new_geo_transform = [point.GetX(), 30.87, 0, point.GetY(), 0, -30.87]
        input_tiff_file.SetGeoTransform(new_geo_transform)
        input_tiff_file = None

    def _download_bbox(self, bbox: List[float], buffer: float = 0) -> str:
        """
        Downloads data given a region and district. Bounding box will be read in from the region_info.yaml file in the
        root data directory. If the region or district is not found an error will be raised.
        Args:
            bbox (list): Bounding box defining area of data request in [min_lon, min_lat, max_lon, max_lat]
            buffer (float): Buffer to add to the bounding box in meters
        """

        # Convert the buffer from meters to degrees lat/long at the equator
        buffer /= 111000

        # Adjust the bounding box to include the buffer (subtract from min lat/long values, add to max lat/long values)
        bbox[0] -= buffer
        bbox[1] -= buffer
        bbox[2] += buffer
        bbox[3] += buffer

        print(bbox)

        temp_dir = tempfile.mkdtemp(prefix='b2p')

        try:
            # 1) Download all overlapping files and convert to tiff in parallel
            file_names = self._resolve_filenames(bbox)
            print(file_names)
            task_args = []
            for file_name in file_names:
                task_args.append(
                    Namespace(
                        link=(os.path.join(self.BASE_URL, file_name)),
                        out_dir=temp_dir,
                        top_left_coord=self._get_top_left_coordinate_from_filename(file_name),
                        username=self._username,
                        password=self._password
                    )
                )
            with mp.Pool(mp.cpu_count() - 1) as pool:
                for _ in tqdm(pool.imap(_elevation_download_task, task_args), total=len(task_args)):
                    pass

            return temp_dir

        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e

    @staticmethod
    def _get_top_left_coordinate_from_filename(infile: str) -> Tuple[float, float]:
        """
        Parses the specified infile name for coordinate info. File names indicate the bottom left pixel's coordinate,
        so need to add 1 deg (files are 1x1 deg and North-up) to lat to get top left corner.
        Args:
            infile (str): Name of path of file to get top left coordinate for
        Returns:
            top_left_coord (Tuple): Top left corner as (lon, lat)
        """
        infile = os.path.basename(infile)

        match = re.search(r"[nsNS](\d{2})[weWE](\d{3})", infile)
        if match:
            n_or_s = match.group(0)[0]
            e_or_w = match.group(0)[3]
            n_value = match.group(1)
            e_value = match.group(2)

        lat = float(n_value) * (1 if n_or_s.lower() == 'n' else -1) + 1
        lon = float(e_value) * (1 if e_or_w.lower() == 'e' else -1)

        return lon, lat

    def _resolve_filenames(self, bbox: List[float]) -> List[str]:
        """
        Files are stored with the following naming convention, for example: N00E003.SRTMGL1_NC.nc where N00 is 0 deg
        latitude and E003 is 3 deg longitude, referring to the lower left coordinate of the data file. So create all
        the possible file name combinations within the range of the bounding box
        Args:
            bbox (list): Bounding box coordinates in [min_lon, min_lat, max_lon, max_lat]
        Returns:
            file_names (list): List of SRTMGL1_NC formatted file names within the bbox range
        """
        min_lon = bbox[0]
        min_lat = bbox[1]
        max_lon = bbox[2]
        max_lat = bbox[3]

        # First find the longitude range
        if min_lon < 0:
            min_lon_ord = 'w'
            round_min_lon = math.ceil(abs(min_lon))
        else:
            min_lon_ord = 'e'
            round_min_lon = math.floor(min_lon)

        if max_lon < 0:
            round_max_lon = math.floor(abs(max_lon))
            max_lon_ord = 'w' if round_max_lon != 0 else 'e'
        else:
            max_lon_ord = 'e'
            round_max_lon = math.ceil(max_lon)

        # Next latitude range
        if min_lat < 0:
            min_lat_ord = 's'
            round_min_lat = math.ceil(abs(min_lat))
        else:
            min_lat_ord = 'n'
            round_min_lat = math.floor(min_lat)

        if max_lat < 0:
            round_max_lat = math.floor(abs(max_lat))
            max_lat_ord = 's' if round_max_lat != 0 else 'n'
        else:
            max_lat_ord = 'n'
            round_max_lat = math.ceil(max_lat)

        print(round_min_lat, round_max_lat, min_lat_ord, max_lat_ord)
        lon_substrings = self._create_substrings(round_min_lon, round_max_lon, min_lon_ord, max_lon_ord, 3)
        lat_substrings = self._create_substrings(round_min_lat, round_max_lat, min_lat_ord, max_lat_ord, 2)

        file_names = []
        for lon in lon_substrings:
            for lat in lat_substrings:
                file_names.append(f"{lat.upper()}{lon.upper()}.SRTMGL1_NC.nc")

        return file_names
