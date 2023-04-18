import getpass
import os
import shutil
from typing import Tuple, List
from argparse import Namespace
from tqdm import tqdm
import re
import urllib
import math
import numpy as np
import certifi
from http.cookiejar import CookieJar
from osgeo import gdal
from osgeo import osr
import zipfile
from array import array
import tempfile
import rasterio
from rasterio.merge import merge

import yaml

from definitions import B2P_DIR, REGION_FILE_PATH
from src.api.util import generate_secrets_file

import requests

BASE_URL = 'https://appeears.earthdatacloud.nasa.gov/api/'


def _download_task(args: Namespace):
    url = os.path.join(BASE_URL, 'bundle', args.task_id, args.file_id)

    header = {
        'Authorization': f'Bearer {args.auth_token}'
    }

    response = requests.get(url, headers=header, allow_redirects=True, stream=True)

    os.makedirs(args.out_dir, exist_ok=True)

    print(args.file_name)
    progress_bar = tqdm(total=args.file_size, unit='iB', unit_scale=True)
    with open(os.path.join(args.out_dir, args.file_name), 'wb') as f:
        for data in response.iter_content(chunk_size=8192):
            progress_bar.update(len(data))
            f.write(data)


class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs.
    """

    BASE_URL = BASE_URL

    def __init__(self):
        """
        Initializes the common attributes required for each data type's API
        """
        self._username, self._password = self._get_auth_credentials()

        self._core_count = os.cpu_count()

    @staticmethod
    def _get_auth_credentials() -> Tuple[str, str]:
        """
        Ask the user for their urs.earthdata.nasa.gov username and login
        Returns:
            username (str): urs.earthdata.nasa.gov username
            password (str): urs.earthdata.nasa.gov password
        """
        secrets_file_path = os.path.join(B2P_DIR, 'secrets.yaml')
        if not os.path.exists(secrets_file_path):
            print('Please input your earthdata.nasa.gov username and password. If you do not have one, you can register'
                  ' here: https://urs.earthdata.nasa.gov/users/new')
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


class Elevation(BaseAPI):
    def __init__(self):
        super().__init__()

        self._auth_token = self._get_auth_token()
        self._AUTH_HEADER = {
            'Authorization': f'Bearer {self._auth_token}'
        }

        self._session_tasks = []

    def download_session_task_files(self, out_dir: str):
        super().download_session_task_files(out_dir, 'SRTMGL1_NC', '_DEM_')

    def submit_task(self, region: str = None, district: str = None, bbox: List[float] = None,
                    start_date: str = '02-11-2000', end_date: str = '02-21-2000', file_type: str = 'geotiff',
                    projection: str = "native"):
        super().submit_task('SRTMGL1_NC.003', 'SRTMGL1_DEM', region, district, bbox, start_date, end_date, file_type,
                            projection)

    def _authorized_get_request(self, url):
        response = requests.get(url, headers=self._AUTH_HEADER)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(response.json())

    def list_task_descriptions(self):
        url = os.path.join(self.BASE_URL, 'task')
        return self._authorized_get_request(url)

    def get_task_description(self, task_id: str):
        url = os.path.join(self.BASE_URL, 'task', task_id)
        return self._authorized_get_request(url)

    def list_task_statuses(self):
        url = os.path.join(self.BASE_URL, 'status')
        return self._authorized_get_request(url)

    def get_task_status(self, task_id: str):
        url = os.path.join(self.BASE_URL, 'status', task_id)
        return self._authorized_get_request(url)

    def list_task_files(self, task_id: str):
        url = os.path.join(self.BASE_URL, 'bundle', task_id)
        return self._authorized_get_request(url)

    def download_file(self, task_id: str, file_id: str, out_dir: str):
        url = os.path.join(self.BASE_URL, 'bundle', task_id, file_id)

        response = requests.get(url, headers=self._AUTH_HEADER, allow_redirects=True, stream=True)

        file_response = self.list_task_files(task_id)
        file_name = None
        file_size = None
        for file in file_response['files']:
            if file['file_id'] == file_id:
                file_name = os.path.basename(file['file_name'])
                file_size = file['file_size']
                break
        if file_size is None or file_name is None:
            raise LookupError('Could not find file id in manifest')

        if response.status_code == 200:
            os.makedirs(out_dir, exist_ok=True)

            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
            with open(os.path.join(out_dir, file_name), 'wb') as f:
                for data in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(data))
                    f.write(data)
        else:
            print(response.json())

    def list_session_tasks_status(self):
        for task in self.session_tasks:
            print(task['task_id'])
            print(self.get_task_status(task))

    @staticmethod
    def _bbox_to_polygon(bbox: List[float]):
        return [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]

    def download_session_task_files(self, out_dir: str, *args):
        task_args = []
        for task in self.session_tasks:
            task_id = task['task_id']
            status_response = self.get_task_status(task_id)
            for session_task in self._session_tasks:
                if session_task['task_id'] == task_id:
                    session_task['status'] = status_response['status']

            if 'status' in status_response and status_response['status'] == 'done':
                files_response = self.list_task_files(task_id)
                for file_info in files_response['files']:
                    file_name = os.path.basename(file_info['file_name'])
                    file_id = file_info['file_id']
                    file_size = file_info['file_size']
                    print(args)
                    match = True
                    for match_string in args:
                        if match_string not in file_name:
                            match = False
                            break

                    if not match:
                        continue

                    if match and ('.tif' in file_name or '.ncdf4' in file_name):
                        task_args.append(Namespace(task_id=task_id, file_id=file_id, file_name=file_name,
                                                   file_size=file_size, out_dir=out_dir, auth_token=self._auth_token))

        for task_arg in task_args:
            print(vars(task_arg))
            _download_task(task_arg)
        print(f'Session tasks: {self.session_tasks}')
        # with mp.Pool(mp.cpu_count() - 1) as pool:
        #     for _ in tqdm(pool.imap_unordered(_download_task, args), total=len(args)):
        #         pass

    def submit_task(self, product: str, layer: str, region: str = None, district: str = None, bbox: List[float] = None,
                    start_date: str = '02-11-2000', end_date: str = '02-21-2000', file_type: str = 'geotiff',
                    projection: str = "native"):
        if region is None and bbox is None:
            raise ValueError('Must specify region or bbox')

        if district is not None and region is None:
            raise ValueError('Must specify a region with the district')

        url = os.path.join(self.BASE_URL, 'task')

        header = self._AUTH_HEADER
        header['Content-Type'] = 'application/json'

        if region is not None:
            with open(REGION_FILE_PATH, 'r') as f:
                region_file_info = yaml.load(f, Loader=yaml.FullLoader)
        region_info = region_file_info[region]

        polygons = []
        if district is not None:
            polygons.append(self._bbox_to_polygon(region_info['districts'][district]['bbox']))
        else:
            for district in region_info['districts']:
                polygons.append(self._bbox_to_polygon(region_info[district]['bbox']))

        body = {
            "task_type": "area",
            "task_name": f"elevation_{1}",
            "params":
                {
                    "dates": [{
                        "startDate": start_date,
                        "endDate": end_date
                    }],
                    "layers": [{
                        "product": product,
                        "layer": layer
                    }],
                    "geo": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": polygons
                                },
                                "type": "Feature", "properties": {}
                            }
                        ]
                    },
                    "output":
                        {
                            "format":
                                {
                                    "type": file_type
                                },
                            "projection": projection
                        }
                }
        }

        response = requests.post(url, headers=header, json=body)

        if response.status_code == 202:
            self._session_tasks.append(response.json())
            print(response.json())
        else:
            raise ValueError(response.json())

    def _get_auth_token(self) -> str:
        url = os.path.join(self.BASE_URL, 'login')

        response = requests.post(url, auth=(self._username, self._password))

        if response.status_code == 200:
            return response.json()['token']
        else:
            raise ValueError(response.json())

    @property
    def session_tasks(self):
        return self._session_tasks


class Slope(BaseAPI):
    BASE_URL = 'https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_SC.001/2000.02.11/'

    def __init__(self):
        super().__init__()

    def _configure(self) -> None:
        """
        Queries the user for credentials and configures SSL certificates
        """
        # This is a macOS thing... need to find path to SSL certificates and set the following environment variables
        ssl_cert_path = certifi.where()
        if 'SSL_CERT_FILE' not in os.environ or os.environ['SSL_CERT_FILE'] != ssl_cert_path:
            os.environ['SSL_CERT_FILE'] = ssl_cert_path

        if 'REQUESTS_CA_BUNDLE' not in os.environ or os.environ['REQUESTS_CA_BUNDLE'] != ssl_cert_path:
            os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path

    def _download(self, query: Tuple[str, str]) -> None:
        """
        Downloads data from the NASA earthdata servers. Authentication is established using the username and password
        found in the local ~/.netrc file.
        Args:
            query (tuple): Contains the remote location and the local path destination, respectively
        """
        link = query[0]
        out_dir = query[1]

        pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pm.add_password(None, "https://urs.earthdata.nasa.gov", self._username, self._password)
        cookie_jar = CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPBasicAuthHandler(pm),
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )
        urllib.request.install_opener(opener)
        myrequest = urllib.request.Request(link)
        response = urllib.request.urlopen(myrequest)
        response.begin()

        dest = os.path.join(out_dir, os.path.basename(link))

        with open(dest, 'wb') as fd:
            while True:
                chunk = response.read()
                if chunk:
                    fd.write(chunk)
                else:
                    break

        # First unzip the file and then find the .slope file
        unzipped_dir = dest.replace('.zip', '')
        os.makedirs(unzipped_dir, exist_ok=True)
        with zipfile.ZipFile(dest, 'r') as zip_ref:
            zip_ref.extractall(unzipped_dir)

        slope_file = None
        for file in os.listdir(unzipped_dir):
            if file.endswith('.slope'):
                slope_file = os.path.join(unzipped_dir, file)
                break

        if slope_file is not None:
            self._binary_to_tif(slope_file, out_dir)
        else:
            raise FileNotFoundError(f'Could not find .slope file in {unzipped_dir}')

        # Just want the .tif file at the end
        shutil.rmtree(unzipped_dir)
        os.remove(dest)

    @staticmethod
    def _create_substrings(min_deg: int, max_deg: int, min_ord: str, max_ord: str, padding: int) -> List[str]:
        substrings = []
        format_str = '{0:03d}' if padding == 3 else '{0:02d}'
        if min_ord == max_ord:
            abs_min = min(min_deg, max_deg)
            abs_max = max(min_deg, max_deg)
            deg_range = np.arange(abs_min, abs_max, 1)
            for deg in deg_range:
                substrings.append(min_ord + format_str.format(deg))
        else:
            # Only other combo would be min_lon_ord is w and max_lon_ord is e
            neg_range = np.arange(1, min_deg, 1)
            pos_range = np.arange(0, max_deg, 1)
            for deg in neg_range:
                substrings.append(min_ord + format_str.format(deg))
            for deg in pos_range:
                substrings.append(max_ord + format_str.format(deg))

        return substrings

    def _resolve_filenames(self, bbox: List[float]):
        """
        Files are stored with the following naming convention: NASADEM_SC_n00e011.zip where n00 is 0 deg latitude and
        e011 is 11 deg longitude, referring to the lower left coordinate of the data file. So create all the possible
        file name combination from the bounding box.
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
            max_lon_ord = 'w'
            round_max_lon = math.floor(abs(max_lon))
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
            max_lat_ord = 's'
            round_max_lat = math.floor(abs(max_lat))
        else:
            max_lat_ord = 'n'
            round_max_lat = math.ceil(max_lat)

        lon_substrings = self._create_substrings(round_min_lon, round_max_lon, min_lon_ord, max_lon_ord, 3)
        lat_substrings = self._create_substrings(round_min_lat, round_max_lat, min_lat_ord, max_lat_ord, 2)

        file_names = []
        for lon in lon_substrings:
            for lat in lat_substrings:
                file_names.append(f"NASADEM_SC_{lat}{lon}.zip")

        return file_names

    def download_bbox(self, bbox: List[int], output_file: str):
        temp_dir = tempfile.mkdtemp(prefix='b2p')

        try:
            # 1) Download all overlapping files
            # 2) Convert raw binary .slope files into geo-referenced tiff
            file_names = self._resolve_filenames(bbox)
            for file_name in file_names:
                self._download((os.path.join(self.BASE_URL, file_name), temp_dir))

            # 3) Create a composite of all tiffs in the temp_dir
            self._mosaic_tiff_files(temp_dir, output_file=output_file)

            # 4) Cleanup
            shutil.rmtree(temp_dir)

        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e

    @staticmethod
    def _coords_from_filename(infile: str):
        infile = os.path.basename(infile)

        match = re.search(r"[ns](\d{2})[we](\d{3})", infile)
        if match:
            n_or_s = match.group(0)[0]
            e_or_w = match.group(0)[3]
            n_value = match.group(1)
            e_value = match.group(2)

        lat = float(n_value) * (1 if n_or_s == 'n' else -1)
        lon = float(e_value) * (1 if e_or_w == 'e' else -1)

        return [lon, lat]

    def _binary_to_tif(self, infile: str, out_dir: str):
        """
        SLOPE layers are non-georeferenced binary files which need to be converted to a more useful file type
        (in this case .tif). A discussion about this can be found
        here: https://forum.earthdata.nasa.gov/viewtopic.php?t=553
        """
        columns = 3601
        rows = 3601

        with open(infile, 'rb') as f:
            data = np.fromfile(f, dtype='>u2')  # Big endian encoding for slope data

        data = data.reshape(columns, rows) / 100  # Slope data has a scale factor of 100

        tiff_path = os.path.join(out_dir, os.path.basename(infile).replace('.slope', '.tif'))

        # File name tells us bottom left coordinate, geotransform needs top left so add 1 deg to lat
        upper_left_coords = self._coords_from_filename(infile)
        upper_left_coords[1] = upper_left_coords[1] + 1

        numpy_array_to_raster(output_path=tiff_path, numpy_array=data, upper_left_tuple=upper_left_coords)

    @staticmethod
    def _mosaic_tiff_files(input_dir: str, output_file: str):
        # Specify the input directory containing the TIFF files
        tiff_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                      f.endswith('.tif')]  # Open all the TIFF files using rasterio
        src_files_to_mosaic = []
        for file in tiff_files:
            src = rasterio.open(file)
            src_files_to_mosaic.append(src)  # Merge the TIFF files using rasterio.merge
        mosaic, out_trans = merge(src_files_to_mosaic)  # Specify the output file path and name
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans})  # Write the merged TIFF file to disk using rasterio
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)


"""
Code for converting numpy array into a tiff file, credit:
https://gis.stackexchange.com/questions/290776/how-to-create-a-tiff-file-using-gdal-from-a-numpy-array-and-specifying-nodata-va
"""


def create_raster(output_path,
                  columns,
                  rows,
                  nband=1,
                  gdal_data_type=gdal.GDT_UInt16,
                  driver=r'GTiff'):
    # create driver
    driver = gdal.GetDriverByName(driver)

    output_raster = driver.Create(output_path,
                                  int(columns),
                                  int(rows),
                                  nband,
                                  eType=gdal_data_type)
    return output_raster


def numpy_array_to_raster(output_path,
                          numpy_array,
                          upper_left_tuple,
                          cell_resolution=0.000277777777777778,
                          nband=1,
                          no_data=15,
                          gdal_data_type=gdal.GDT_UInt16,
                          spatial_reference_system_wkid=4326,
                          driver=r'GTiff'):
    ''' returns a gdal raster data source

    keyword arguments:

    output_path -- full path to the raster to be written to disk
    numpy_array -- numpy array containing data to write to raster
    upper_left_tuple -- the upper left point of the numpy array (should be a tuple structured as (x, y))
    cell_resolution -- the cell resolution of the output raster
    nband -- the band to write to in the output raster
    no_data -- value in numpy array that should be treated as no data
    gdal_data_type -- gdal data type of raster (see gdal documentation for list of values)
    spatial_reference_system_wkid -- well known id (wkid) of the spatial reference of the data
    driver -- string value of the gdal driver to use

    '''
    rows, columns = numpy_array.shape

    # create output raster
    output_raster = create_raster(output_path,
                                  int(columns),
                                  int(rows),
                                  nband,
                                  gdal_data_type)

    geotransform = (upper_left_tuple[0],
                    cell_resolution,
                    0,
                    upper_left_tuple[1],
                    0,
                    - 1 * cell_resolution)

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    output_raster.SetProjection(spatial_reference.ExportToWkt())
    output_raster.SetGeoTransform(geotransform)
    output_band = output_raster.GetRasterBand(1)
    output_band.SetNoDataValue(no_data)
    output_band.WriteArray(numpy_array)
    output_band.FlushCache()
    output_band.ComputeStatistics(False)

    if os.path.exists(output_path) == False:
        raise Exception('Failed to create raster: %s' % output_path)

    return output_raster
