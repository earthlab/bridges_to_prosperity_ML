import os
import re
from typing import List, Union
from definitions import COMPOSITE_DIR, B2P_DIR, TILE_DIR, ELEVATION_DIR, SLOPE_DIR, SENTINEL_2_DIR, MULTIVARIATE_DIR,\
    OSM_DIR
import glob


class FileType:
    def __init__(self):
        pass

    def identify(self):
        pass


class OpticalComposite(FileType):
    base_regex = r'optical_composite_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_'

    def __init__(self, region: str, district: str, military_grid: str, bands: List[str]):
        self.region = region
        self.district = district
        self.mgrs = military_grid
        self.bands = sorted(bands)
        super().__init__()

    @property
    def name(self):
        base_string = f'optical_composite_{self.region}_{self.district}_{self.mgrs}_'
        for band in self.bands:
            base_string += band + ('_' if band != self.bands[-1] else '')
        return base_string + '.tif'

    @property
    def archive_path(self):
        return os.path.join(B2P_DIR, 'data', COMPOSITE_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, bands: List[str] = None, recursive: bool = False):
        if bands is None:
            bands = []
        bands = sorted(bands)
        regex = OpticalComposite.base_regex
        for band in bands:
            regex += r'{}'.format(band) + (r'_' if band != bands[-1] else r'')
        if not bands:
            regex += '.*'

        regex += r'\.tif$'
        files = glob.glob(in_dir + '/**/*', recursive=recursive)
        print(files)

        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.base_regex + r'(?P<bands>[a-zA-Z0-9]{3}(?:_[a-zA-Z0-9]{3})*)\.tif$'
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'],
                       bands=group_dict['bands'].split('_'))
        else:
            return None


class Sentinel2Tile(FileType):
    base_regex = r'tiles_(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_'

    def __init__(self, utm_code: str, latitude_band: str, square: str, year: int, month: int, day: int, band: str,
                 sequence: int = 0):
        self.utm_code = utm_code
        self.latitude_band = latitude_band
        self.square = square
        self.year = year
        self.month = month
        self.day = day
        self.band = band
        self.sequence = sequence
        super().__init__()

    @property
    def name(self) -> str:
        return f'tiles_{self.utm_code}_{self.latitude_band}_{self.square}_{self.year}_{self.month}_{self.day}_{self.sequence}_' \
               f'{self.band}.jp2'

    @property
    def mgrs_grid(self):
        return f'{self.utm_code}{self.latitude_band}{self.square}'

    @property
    def date_str(self):
        return f'{self.year}{self.month}{self.day}'

    def archive_path(self, region: str, district: str) -> str:
        return os.path.join(SENTINEL_2_DIR, region, district, self.mgrs_grid, self.date_str, self.name)

    @staticmethod
    def find_files(in_dir: str, bands: List[str], recursive: bool = False):
        bands = sorted(bands)
        regex = Sentinel2Tile.base_regex
        for band in bands:
            regex += r'{}'.format(band) + (r'_' if band != bands[-1] else r'')
        regex += r'\.jp2$'
        files = glob.glob(in_dir + '/**/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        regex = cls.base_regex + r'(?P<band>B\d{2})\.jp2$'
        name = os.path.basename(file_path)
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(utm_code=group_dict['utm_code'], latitude_band=group_dict['latitude_band'],
                       square=group_dict['square'], year=group_dict['year'], month=group_dict['month'],
                       day=group_dict['day'], band=group_dict['band'], sequence=group_dict['sequence'])
        else:
            return None


class Sentinel2Cloud(FileType):
    base_regex = r'tiles_(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_qi_MSK_CLOUDS_B00.gml'

    def __init__(self, utm_code: str, latitude_band: str, square: str, year: int, month: int, day: int,
                 sequence: int = 0):
        self.utm_code = utm_code
        self.latitude_band = latitude_band
        self.square = square
        self.year = year
        self.month = month
        self.day = day
        self.sequence = sequence
        super().__init__()

    @property
    def name(self) -> str:
        return f'tiles_{self.utm_code}_{self.latitude_band}_{self.square}_{self.year}_{self.month}_{self.day}_' \
               f'{self.sequence}_qi_MSK_CLOUDS_B00.gml'

    @property
    def mgrs_grid(self):
        return f'{self.utm_code}{self.latitude_band}{self.square}'

    @property
    def date_str(self):
        return f'{self.year}{self.month}{self.day}'

    def archive_path(self, region: str, district: str) -> str:
        return os.path.join(SENTINEL_2_DIR, region, district, self.mgrs_grid, self.date_str, self.name)

    @staticmethod
    def find_files(in_dir: str, recursive: bool = False):
        files = glob.glob(in_dir + '/**/*', recursive=recursive)
        matching_files = [f for f in files if re.match(Sentinel2Cloud.base_regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.base_regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(utm_code=group_dict['utm_code'], latitude_band=group_dict['latitude_band'],
                       square=group_dict['square'], year=group_dict['year'], month=group_dict['month'],
                       day=group_dict['day'], sequence=group_dict['sequence'])
        else:
            return None


class Elevation(FileType):
    base_regex = r'elevation_(?P<region>[^_]+)_(?P<district>[^_]+)(?:_(?P<mgrs>[^_]{5}))?\.tif$'

    def __init__(self, region: str, district: str, mgrs: str = None):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def name(self):
        base_string = f'elevation_{self.region}_{self.district}'
        if self.mgrs is not None:
            base_string += f'_{self.mgrs}'
        return base_string + '.tif'

    @property
    def archive_path(self):
        return os.path.join(ELEVATION_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, recursive: bool = False):
        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(Elevation.base_regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.base_regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'],
                       mgrs=group_dict['mgrs'] if 'mgrs' in group_dict else None)
        else:
            return None


class Slope(FileType):
    base_regex = r'slope_(?P<region>[^_]+)_(?P<district>[^_]+)(?:_(?P<mgrs>[^_]{5}))?\.tif$'

    def __init__(self, region: str, district: str, mgrs: str = None):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def name(self):
        base_string = f'slope_{self.region}_{self.district}'
        if self.mgrs is not None:
            base_string += f'_{self.mgrs}'
        return base_string + '.tif'

    @property
    def archive_path(self):
        return os.path.join(SLOPE_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, recursive: bool = False):
        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(Slope.base_regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.base_regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'],
                       mgrs=group_dict['mgrs'] if 'mgrs' in group_dict else None)
        else:
            return None


class OSM(FileType):
    base_regex = r'osm_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)\.tif$'

    def __init__(self, region: str, district: str, mgrs: str = None):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def name(self):
        base_string = f'osm_{self.region}_{self.district}_{self.mgrs}'
        return base_string + '.tif'

    @property
    def archive_path(self):
        return os.path.join(OSM_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, recursive: bool = False):
        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(Slope.base_regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.base_regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], mgrs=group_dict['mgrs'])
        else:
            return None


class TileGeoLoc(FileType):
    base_regex = r'^(?P<bands>[a-zA-Z0-9]{3}|multivariate(?:_[a-zA-Z0-9]{3}|multivariate)*)_geoloc\.csv$'

    def __init__(self, bands=None):
        if bands is None:
            bands = ['multivariate']
        self.bands = sorted(bands)
        super().__init__()

    @property
    def name(self):
        base_string = ''
        for band in self.bands:
            base_string += band + '_'
        base_string += 'geoloc.csv'
        return base_string

    def archive_path(self, region: str, district: str, military_grid: str) -> str:
        return os.path.join(TILE_DIR, region, district, military_grid, self.name)

    @staticmethod
    def find_files(in_dir: str, bands: List[str], recursive: bool = False):
        bands = sorted(bands)
        regex = r''
        for band in bands:
            regex += r'{}'.format(band) + r'_'
        regex += r'geoloc\.csv$'

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.base_regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(bands=group_dict['bands'].split('_'))
        else:
            return None


class TileMatch(FileType):
    base_regex = r'^(?P<bands>[a-zA-Z0-9]{3}|multivariate(?:_[a-zA-Z0-9]{3}|multivariate)*)_tile_match\.csv$'

    def __init__(self, bands=None):
        if bands is None:
            bands = ['multivariate']
        self.bands = sorted(bands)
        super().__init__()

    @property
    def name(self):
        base_string = ''
        for band in self.bands:
            base_string += band + '_'
        base_string += 'tile_match.csv'
        return base_string

    def archive_path(self, region: str, district: str) -> str:
        return os.path.join(TILE_DIR, region, district, self.name)

    @staticmethod
    def find_files(in_dir: str, bands: List[str], recursive: bool = False):
        bands = sorted(bands)
        regex = r''
        for band in bands:
            regex += r'{}'.format(band) + r'_'
        regex += r'tile_match\.csv$'

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.base_regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(bands=group_dict['bands'].split('_'))
        else:
            return None


class Tile(FileType):
    base_regex = r'^(?P<bands>[a-zA-Z0-9]{3}|multivariate(?:_[a-zA-Z0-9]{3}|multivariate)*)_(?P<x_min>\d+)_(?P<y_min>\d+)\.tif$'

    def __init__(self, x_min: int, y_min: int, bands=None):
        if bands is None:
            bands = ['multivariate']
        self.bands = sorted(bands)
        self.x_min = str(x_min)
        self.y_min = str(y_min)
        super().__init__()

    @property
    def name(self) -> str:
        base_string = ''
        for band in self.bands:
            base_string += band + '_'
        base_string += self.x_min + '_' + self.y_min + '.tif'
        return base_string

    def archive_path(self, region: str, district: str, military_grid: str) -> str:
        return os.path.join(TILE_DIR, region, district, military_grid, self.name)

    @staticmethod
    def find_files(in_dir: str, bands: List[str], x_min: int, y_min: int, recursive: bool = False):
        bands = sorted(bands)
        regex = r''
        for band in bands:
            regex += r'{}'.format(band) + r'_'
        regex += r'{}'.format(x_min) + r'_' + r'{}'.format(y_min) + r'\.tif$'

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.base_regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(bands=group_dict['bands'].split('_'), x_min=int(group_dict['x_min']),
                       y_min=int(group_dict['y_min']))
        else:
            return None


class PyTorch(FileType):
    base_regex = r'^(?P<bands>[a-zA-Z0-9]{3}|multivariate(?:_[a-zA-Z0-9]{3}|multivariate)*)_(?P<x_min>\d+)_(?P<y_min>\d+)\.pt$'

    def __init__(self, x_min: int, y_min: int, bands=None):
        if bands is None:
            bands = ['multivariate']
        self.bands = sorted(bands)
        self.x_min = str(x_min)
        self.y_min = str(y_min)
        super().__init__()

    @property
    def name(self) -> str:
        base_string = ''
        for band in self.bands:
            base_string += band + '_'
        base_string += self.x_min + '_' + self.y_min + '.pt'
        return base_string

    def archive_path(self, region: str, district: str, military_grid: str) -> str:
        return os.path.join(TILE_DIR, region, district, military_grid, self.name)

    @staticmethod
    def find_files(in_dir: str, bands: List[str], x_min: int, y_min: int, recursive: bool = False):
        bands = sorted(bands)
        regex = r''
        for band in bands:
            regex += r'{}'.format(band) + r'_'
        regex += r'{}'.format(x_min) + r'_' + r'{}'.format(y_min) + r'\.pt$'

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.base_regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(bands=group_dict['bands'].split('_'), x_min=int(group_dict['x_min']),
                       y_min=int(group_dict['y_min']))
        else:
            return None


class MultiVariateComposite(FileType):
    base_regex = r'multivariate_composite_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)\.tif$'

    def __init__(self, region: str, district: str, mgrs: str = None):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def name(self):
        base_string = f'multivariate_composite_{self.region}_{self.district}_{self.mgrs}'
        return base_string + '.tif'

    @property
    def archive_path(self):
        return os.path.join(MULTIVARIATE_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, recursive: bool = False):
        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(Slope.base_regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.base_regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], mgrs=group_dict['mgrs'])
        else:
            return None
