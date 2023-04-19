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
import tempfile
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from netCDF4 import Dataset

import yaml

from definitions import B2P_DIR, REGION_FILE_PATH
from src.api.util import generate_secrets_file


class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs.
    """

    BASE_URL = None

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

    def _download(self, query: Tuple[str, str]) -> Tuple[str, str]:
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

        return dest, out_dir

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

        return lon_substrings, lat_substrings

    @staticmethod
    def _configure() -> None:
        """
        Queries the user for credentials and configures SSL certificates
        """
        # This is a macOS thing... need to find path to SSL certificates and set the following environment variables
        ssl_cert_path = certifi.where()
        if 'SSL_CERT_FILE' not in os.environ or os.environ['SSL_CERT_FILE'] != ssl_cert_path:
            os.environ['SSL_CERT_FILE'] = ssl_cert_path

        if 'REQUESTS_CA_BUNDLE' not in os.environ or os.environ['REQUESTS_CA_BUNDLE'] != ssl_cert_path:
            os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path

    @staticmethod
    def _mosaic_tif_files(input_dir: str, output_file: str):
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
                         "transform": out_trans})  # Write the merged TIFF file to disk using rasterio
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)


class Elevation(BaseAPI):
    BASE_URL = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1_NC.003/2000.02.11/'

    def __init__(self):
        super().__init__()

    def _resolve_filenames(self, bbox: List[float]):
        lon_substrings, lat_substrings = super()._resolve_filenames(bbox)

        file_names = []
        for lon in lon_substrings:
            for lat in lat_substrings:
                file_names.append(f"{lat.upper()}{lon.upper()}.SRTMGL1_NC.nc")

        return file_names

    @staticmethod
    def _coords_from_filename(infile: str):
        infile = os.path.basename(infile)
        match = re.search(r"[NS](\d{2})[WE](\d{3})", infile)
        if match:
            n_or_s = match.group(0)[0]
            e_or_w = match.group(0)[3]
            n_value = match.group(1)
            e_value = match.group(2)

        lat = float(n_value) * (1 if n_or_s == 'N' else -1)
        lon = float(e_value) * (1 if e_or_w == 'E' else -1)

        return [lon, lat]

    def download_bbox(self, bbox: List[int], output_file: str):
        temp_dir = tempfile.mkdtemp(prefix='b2p')

        try:
            # 1) Download all overlapping files
            # 2) Convert raw binary .slope files into geo-referenced tiff
            file_names = self._resolve_filenames(bbox)
            for file_name in file_names:
                print(os.path.join(self.BASE_URL, file_name))
                self._download((os.path.join(self.BASE_URL, file_name), temp_dir))

            # 3) Create a composite of all tiffs in the temp_dir
            self._mosaic_tif_files(temp_dir, output_file=output_file)

            # 4) Cleanup
            #shutil.rmtree(temp_dir)

        except Exception as e:
            #shutil.rmtree(temp_dir)
            raise e

    def _download(self, query: Tuple[str, str]) -> None:
        """
        Downloads data from the NASA earthdata servers. Authentication is established using the username and password
        found in the local ~/.netrc file.
        Args:
            query (tuple): Contains the remote location and the local path destination, respectively
        """
        dest, out_dir = super()._download(query)

        bottom_left_coords = self._coords_from_filename(os.path.basename(dest))
        self._nc_to_tif(dest, bottom_left_coords, out_dir)
        print(dest)
        # Just want the .tif file at the end
        #os.remove(dest)

    @staticmethod
    def _nc_to_tif(nc_path: str, upper_left_tuple: List[float], out_dir: str,
                   cell_resolution: float = 0.000277777777777778):
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
            (upper_left_tuple[0],
             cell_resolution,
             0,
             upper_left_tuple[1],
             0,
             cell_resolution)
        )

        # Write the data to the GeoTIFF file
        tif_band = tif_dataset.GetRasterBand(1)
        tif_band.WriteArray(np.flipud(var))

        # Close the GeoTIFF file and netCDF file
        tif_dataset = None
        nc_file.close()


class Slope(BaseAPI):
    BASE_URL = 'https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_SC.001/2000.02.11/'

    def __init__(self):
        super().__init__()

    def _resolve_filenames(self, bbox: List[float]):
        lon_substrings, lat_substrings = super()._resolve_filenames(bbox)

        file_names = []
        for lon in lon_substrings:
            for lat in lat_substrings:
                file_names.append(f"NASADEM_SC_{lat}{lon}.zip")

        return file_names

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

    def download_bbox(self, bbox: List[int], output_file: str):
        temp_dir = tempfile.mkdtemp(prefix='b2p')

        try:
            # 1) Download all overlapping files
            # 2) Convert raw binary .slope files into geo-referenced tiff
            file_names = self._resolve_filenames(bbox)
            for file_name in file_names:
                print(os.path.join(self.BASE_URL, file_name))
                self._download((os.path.join(self.BASE_URL, file_name), temp_dir))

            # 3) Create a composite of all tiffs in the temp_dir
            self._mosaic_tif_files(temp_dir, output_file=output_file)

            # 4) Cleanup
            shutil.rmtree(temp_dir)

        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e

    def _download(self, query: Tuple[str, str]) -> None:
        """
        Downloads data from the NASA earthdata servers. Authentication is established using the username and password
        found in the local ~/.netrc file.
        Args:
            query (tuple): Contains the remote location and the local path destination, respectively
        """
        dest, out_dir = super()._download(query)

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

        self._numpy_array_to_raster(output_path=tiff_path, numpy_array=data, upper_left_tuple=upper_left_coords)

    # TODO: Create abstract method for coords_from_filename

    # Methods for converting numpy array into a tiff file, credit:
    # https://gis.stackexchange.com/questions/290776/how-to-create-a-tiff-file-using-gdal-from-a-numpy-array-and-specifying-nodata-va

    @staticmethod
    def _create_raster(output_path,
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

    def _numpy_array_to_raster(self,
                               output_path,
                               numpy_array,
                               upper_left_tuple,
                               cell_resolution=0.000277777777777778,
                               nband=1,
                               no_data=15,
                               gdal_data_type=gdal.GDT_UInt16,
                               spatial_reference_system_wkid=4326):
        """
        Returns a gdal raster data source
        Args:
            output_path -- full path to the raster to be written to disk
            numpy_array -- numpy array containing data to write to raster
            upper_left_tuple -- the upper left point of the numpy array (should be a tuple structured as (x, y))
            cell_resolution -- the cell resolution of the output raster
            nband -- the band to write to in the output raster
            no_data -- value in numpy array that should be treated as no data
            gdal_data_type -- gdal data type of raster (see gdal documentation for list of values)
            spatial_reference_system_wkid -- well known id (wkid) of the spatial reference of the data
            driver -- string value of the gdal driver to use

        """
        rows, columns = numpy_array.shape

        # create output raster
        output_raster = self._create_raster(output_path,
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

        if not os.path.exists(output_path):
            raise Exception('Failed to create raster: %s' % output_path)

        return output_raster
