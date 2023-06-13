import os
import re
from enum import Enum, auto
from typing import List, Union
from definitions import COMPOSITE_DIR, B2P_DIR, ELEVATION_DIR, SLOPE_DIR, SENTINEL_2_DIR, MULTIVARIATE_DIR, OSM_DIR, \
    MODEL_DIR, INFERENCE_RESULTS_DIR, TILE_DIR
import glob


class FileType(Enum):
    MULTIVARIATE_COMPOSITE = auto()
    OPTICAL_COMPOSITE = auto()


class File:
    def __init__(self):
        self._s3 = False

    @classmethod
    def create(cls, file_name):
        for subclass in cls.__subclasses__():
            if hasattr(subclass, 'regex') and re.match(subclass.regex, file_name):
                return subclass.create(file_name)
        return None

    @property
    def exists(self):
        return os.path.exists(self.archive_path())

    @property
    def archive_dir(self):
        return os.path.dirname(self.archive_path())

    def s3_archive_path(self):
        if self._s3:
            return self.archive_path().replace(os.path.join(B2P_DIR, 'data'), '')
        return None


class OpticalComposite(File):
    base_regex = r'optical_composite_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_'
    regex = base_regex + r'(?P<bands>[a-zA-Z0-9]{3}(?:_[a-zA-Z0-9]{3})*)\.tif$'

    def __init__(self, region: str, district: str, military_grid: str, bands: List[str]):
        self.region = region
        self.district = district
        self.mgrs = military_grid
        self.bands = sorted(bands)
        super().__init__()
        self._s3 = True

    def _build_regex(self):
        return

    @property
    def name(self):
        base_string = f'optical_composite_{self.region}_{self.district}_{self.mgrs}_'
        for band in self.bands:
            base_string += band + ('_' if band != self.bands[-1] else '')
        return base_string + '.tif'

    def archive_path(self):
        return os.path.join(B2P_DIR, 'data', COMPOSITE_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, bands: List[str] = None, mgrs: List[str] = None, recursive: bool = False):
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

        if mgrs is not None:
            matching_files = []
            for f in files:
                match = re.match(OpticalComposite.base_regex, os.path.basename(f))
                if match and match.groupdict()['mgrs'] in mgrs:
                    matching_files.append(f)
        else:
            matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'],
                       bands=group_dict['bands'].split('_'))
        else:
            return None


class Sentinel2Tile(File):
    base_regex = r'tiles_(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_'
    regex = base_regex + r'(?P<band>B\d{2})\.jp2$'

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

        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(utm_code=group_dict['utm_code'], latitude_band=group_dict['latitude_band'],
                       square=group_dict['square'], year=group_dict['year'], month=group_dict['month'],
                       day=group_dict['day'], band=group_dict['band'], sequence=group_dict['sequence'])
        else:
            return None


class Sentinel2Cloud(File):
    regex = r'tiles_(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_qi_MSK_CLOUDS_B00.gml'

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
        matching_files = [f for f in files if re.match(Sentinel2Cloud.regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(utm_code=group_dict['utm_code'], latitude_band=group_dict['latitude_band'],
                       square=group_dict['square'], year=group_dict['year'], month=group_dict['month'],
                       day=group_dict['day'], sequence=group_dict['sequence'])
        else:
            return None


class Elevation(File):
    regex = r'elevation_(?P<region>[^_]+)_(?P<district>[^_]+)(?:_(?P<mgrs>[^_]{5}))?\.tif$'

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

    def archive_path(self):
        return os.path.join(ELEVATION_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, recursive: bool = False):
        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(Elevation.regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'],
                       mgrs=group_dict['mgrs'] if 'mgrs' in group_dict else None)
        else:
            return None


class Slope(File):
    regex = r'slope_(?P<region>[^_]+)_(?P<district>[^_]+)(?:_(?P<mgrs>[^_]{5}))?\.tif$'

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

    def archive_path(self):
        return os.path.join(SLOPE_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, recursive: bool = False):
        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(Slope.regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'],
                       mgrs=group_dict['mgrs'] if 'mgrs' in group_dict else None)
        else:
            return None


class OSM(File):
    regex = r'osm_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)\.tif$'

    def __init__(self, region: str, district: str, mgrs: str = None):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def name(self):
        base_string = f'osm_{self.region}_{self.district}_{self.mgrs}'
        return base_string + '.tif'

    def archive_path(self):
        return os.path.join(OSM_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, recursive: bool = False):
        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(OSM.regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], mgrs=group_dict['mgrs'])
        else:
            return None


class TileGeoLoc(File):
    regex = r'^(?P<bands>[a-zA-Z0-9]{3}|multivariate(?:_[a-zA-Z0-9]{3}|multivariate)*)_geoloc\.csv$'

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

    def archive_path(self, tile_dir: str, region: str, district: str, military_grid: str,
                     bands: List[str] = None) -> str:
        base_path = os.path.join(tile_dir, region, district)
        if bands is not None:
            base_path = os.path.join(base_path, '_'.join(bands))
        base_path = os.path.join(base_path, military_grid, self.name)
        return base_path

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
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(bands=group_dict['bands'].split('_'))
        else:
            return None


class TileMatch(File):
    regex = r'^(?P<bands>[a-zA-Z0-9]{3}|multivariate(?:_[a-zA-Z0-9]{3}|multivariate)*)_tile_match\.csv$'

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

    def archive_path(self, region: str, district: str = None) -> str:
        base_path = os.path.join(TILE_DIR, region)
        if district is not None:
            base_path = os.path.join(base_path, district)
        base_path = os.path.join(base_path, self.name)
        return base_path

    @staticmethod
    def find_files(in_dir: str, bands: List[str] = None, recursive: bool = False):
        regex = r''
        if bands is not None:
            bands = sorted(bands)
            for band in bands:
                regex += r'{}'.format(band) + r'_'
        else:
            regex += 'multivariate_'
        regex += r'tile_match\.csv$'

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(bands=group_dict['bands'].split('_'))
        else:
            return None


class Tile(File):
    regex = r'^(?P<bands>[a-zA-Z0-9]{3}|multivariate(?:_[a-zA-Z0-9]{3}|multivariate)*)_(?P<x_min>\d+)_(?P<y_min>\d+)\.tif$'

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

    def archive_path(self, tile_dir: str, region: str, district: str, military_grid: str) -> str:
        return os.path.join(tile_dir, region, district, military_grid, self.name)

    @staticmethod
    def find_files(in_dir: str, x_min: int, y_min: int, bands: List[str] = None, recursive: bool = False):
        bands = sorted(bands)
        regex = r''
        if bands is not None:
            for band in bands:
                regex += r'{}'.format(band) + r'_'
        else:
            regex += 'multivariate_'
        regex += r'{}'.format(x_min) + r'_' + r'{}'.format(y_min) + r'\.tif$'

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(bands=group_dict['bands'].split('_'), x_min=int(group_dict['x_min']),
                       y_min=int(group_dict['y_min']))
        else:
            return None


class TrainedModel(File):
    lr = 'red|blue|green|nir|osm-water|osm-boundary|elevation|slope'
    regex = rf'^(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_(?P<layer_1>{lr})_((?P<layer_2>{lr}_)?((?P<layer_3>{lr})_)epoch(?P<epoch>\d+)?(?P<best>_best)\.tar$'

    def __init__(self, architecture: str, layers: List[str], epoch: int, ratio: float, best: bool = False):
        self.architecture = architecture
        self.layers = sorted(layers)
        self.epoch = epoch
        self.ratio = ratio
        self.best = best
        super().__init__()
        self._s3 = True

    @property
    def name(self) -> str:
        base_str = self.architecture + f'_r{self.ratio}_' + '_'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '.tar'
        return base_str

    def archive_path(self):
        return os.path.join(MODEL_DIR, self.architecture, f'r{self.ratio}', self.name)

    @staticmethod
    def find_files(in_dir: str, architecture: str, layers: List[str], epoch: str, ratio: float, best: bool = False,
                   recursive: bool = False):
        layers = sorted(layers)
        regex = rf'{architecture}_r{ratio}_'
        for layer in layers:
            regex += r'{}'.format(layer) + r'_'
        regex += rf'{epoch}'
        if best:
            regex += '_best'
        regex += "\.tar$"

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            layers = []
            for i in range(1, 4):
                layer = group_dict[f'layer_{i}']
                if layer is not None:
                    layers.append(layer)
            return cls(architecture=group_dict['architecture'], layers=layers, epoch=group_dict['epoch'],
                       ratio=group_dict['ratio'], best=group_dict['best'] is not None)
        else:
            return None


class PyTorch(File):
    regex = r'^(?P<bands>[a-zA-Z0-9]{3}|multivariate(?:_[a-zA-Z0-9]{3}|multivariate)*)_(?P<x_min>\d+)_(?P<y_min>\d+)\.pt$'

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

    def archive_path(self, tile_dir: str, region: str, district: str, military_grid: str,
                     bands: List[str] = None) -> str:
        base_path = os.path.join(tile_dir, region, district)
        if bands is not None:
            base_path = os.path.join(base_path, '_'.join(bands))
        base_path = os.path.join(base_path, military_grid, self.name)
        return base_path

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
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(bands=group_dict['bands'].split('_'), x_min=int(group_dict['x_min']),
                       y_min=int(group_dict['y_min']))
        else:
            return None


class TrainSplit(File):
    regex = r'train_(?P<regions>\w+(?:_\w+)*)_(?P<ratio>\d+)\.csv$'

    def __init__(self, regions: List[str], ratio: int):
        self.regions = regions
        self.ratio = ratio
        super().__init__()

    @property
    def name(self):
        return f"train_{'_'.join(self.regions)}_{self.ratio}.csv"

    def archive_path(self) -> str:
        return os.path.join(TILE_DIR, 'train_validate_splits', self.name)

    @staticmethod
    def find_files(in_dir: str, regions: List[str], ratio: int, recursive: bool = False):
        files = glob.glob(in_dir + '/**/*', recursive=recursive)

        regex = r'train_'
        for region in regions:
            regex += region + '_'
        regex += f'{ratio}.csv'

        matching_files = [f for f in files if re.match(TrainSplit.regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(regions=group_dict['regions'].split('_'), ratio=group_dict['ratio'])
        else:
            return None


class ValidateSplit(File):
    regex = r'validate_(?P<regions>\w+(?:_\w+)*)_(?P<ratio>\d+)\.csv$'

    def __init__(self, regions: List[str], ratio: int):
        self.regions = regions
        self.ratio = ratio
        super().__init__()

    @property
    def name(self):
        return f"validate_{'_'.join(self.regions)}_{self.ratio}.csv"

    def archive_path(self) -> str:
        return os.path.join(TILE_DIR, 'train_validate_splits', self.name)

    @staticmethod
    def find_files(in_dir: str, regions: List[str], ratio: int, recursive: bool = False):
        files = glob.glob(in_dir + '/**/*', recursive=recursive)

        regex = r'validate_'
        for region in regions:
            regex += region + '_'
        regex += f'{ratio}.csv'

        matching_files = [f for f in files if re.match(TrainSplit.regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(regions=group_dict['regions'].split('_'), ratio=group_dict['ratio'])
        else:
            return None


class InferenceResultsCSV(File):
    lr = 'red|blue|green|nir|osm-water|osm-boundary|elevation|slope'
    regex = rf'^(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_(?P<layer_1>{lr})_((?P<layer_2>{lr}_)?((?P<layer_3>{lr})_)epoch(?P<epoch>\d+)?(?P<best>_best)_results\.csv$'

    def __init__(self, architecture: str, layers: List[str], epoch: int, ratio: float, best: bool = False):
        self.architecture = architecture
        self.layers = sorted(layers)
        self.epoch = epoch
        self.ratio = ratio
        self.best = best
        super().__init__()
        self._s3 = True

    @property
    def name(self) -> str:
        base_str = self.architecture + f'_r{self.ratio}_' + '_'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '_results' '.csv'
        return base_str

    def archive_path(self, region: str, ):
        return os.path.join(INFERENCE_RESULTS_DIR, self.architecture, f'r{self.ratio}', self.name)

    @staticmethod
    def find_files(in_dir: str, architecture: str, layers: List[str], epoch: str, ratio: float, best: bool = False,
                   recursive: bool = False):
        layers = sorted(layers)
        regex = rf'{architecture}_r{ratio}_'
        for layer in layers:
            regex += r'{}'.format(layer) + r'_'
        regex += rf'{epoch}'
        if best:
            regex += '_best'
        regex += "_results\.csv$"

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            layers = []
            for i in range(1, 4):
                layer = group_dict[f'layer_{i}']
                if layer is not None:
                    layers.append(layer)
            return cls(architecture=group_dict['architecture'], layers=layers, epoch=group_dict['epoch'],
                       ratio=group_dict['ratio'], best=group_dict['best'] is not None)
        else:
            return None


class InferenceResultsShapefile(File):
    lr = 'red|blue|green|nir|osm-water|osm-boundary|elevation|slope'
    regex = rf'^(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_(?P<layer_1>{lr})_((?P<layer_2>{lr}_)?((?P<layer_3>{lr})_)epoch(?P<epoch>\d+)?(?P<best>_best)_results\.shp$'

    def __init__(self, architecture: str, layers: List[str], epoch: int, ratio: float, best: bool = False):
        self.architecture = architecture
        self.layers = sorted(layers)
        self.epoch = epoch
        self.ratio = ratio
        self.best = best
        super().__init__()
        self._s3 = True

    @property
    def name(self) -> str:
        base_str = self.architecture + f'_r{self.ratio}_' + '_'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '_results' '.shp'
        return base_str

    def archive_path(self):
        sub_dir = '_'.join(self.layers) + f'_{self.epoch}' + '_best' if self.best else ''
        return os.path.join(INFERENCE_RESULTS_DIR, self.architecture, f'r{self.ratio}', sub_dir, self.name)

    @staticmethod
    def find_files(in_dir: str, architecture: str, layers: List[str], epoch: str, ratio: float, best: bool = False,
                   recursive: bool = False):
        layers = sorted(layers)
        regex = rf'{architecture}_r{ratio}_'
        for layer in layers:
            regex += r'{}'.format(layer) + r'_'
        regex += rf'{epoch}'
        if best:
            regex += '_best'
        regex += "_results\.csv$"

        files = glob.glob(in_dir + '/*', recursive=recursive)
        matching_files = [f for f in files if re.match(regex, os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            layers = []
            for i in range(1, 4):
                layer = group_dict[f'layer_{i}']
                if layer is not None:
                    layers.append(layer)
            return cls(architecture=group_dict['architecture'], layers=layers, epoch=group_dict['epoch'],
                       ratio=group_dict['ratio'], best=group_dict['best'] is not None)
        else:
            return None


class MultiVariateComposite(File):
    regex = r'multivariate_composite_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)\.tif$'

    def __init__(self, region: str, district: str, mgrs: str = None):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()
        self._s3 = True

    @property
    def name(self):
        base_string = f'multivariate_composite_{self.region}_{self.district}_{self.mgrs}'
        return base_string + '.tif'

    def archive_path(self):
        return os.path.join(MULTIVARIATE_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, mgrs: List[str] = None, recursive: bool = False):
        files = glob.glob(in_dir + '/**/*', recursive=recursive)
        if mgrs is not None:
            matching_files = []
            for f in files:
                match = re.match(MultiVariateComposite.regex, os.path.basename(f))
                if match and match.groupdict()['mgrs'] in mgrs:
                    matching_files.append(f)
        else:
            matching_files = [f for f in files if re.match(MultiVariateComposite.regex,
                                                           os.path.basename(f)) is not None]

        return matching_files

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], mgrs=group_dict['mgrs'])
        else:
            return None
