import os
import re
from typing import List, Set
from definitions import COMPOSITE_DIR, ELEVATION_DIR, SLOPE_DIR, SENTINEL_2_DIR, OSM_DIR, \
    MODEL_DIR, INFERENCE_RESULTS_DIR, TILE_DIR, TRAIN_VALIDATE_SPLIT_DIR, MULTI_REGION_TILE_MATCH, DATA_DIR
import glob
from pathlib import Path
from abc import ABC, abstractmethod


class File(ABC):
    def __init__(self):
        pass
    
    @classmethod
    @abstractmethod
    def create(cls, file_name):
        for subclass in cls.__subclasses__():
            if hasattr(subclass, 'regex') and re.match(subclass.regex, file_name):
                return subclass.create(file_name)
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def archive_path(self) -> str:
        pass

    @property
    def exists(self) -> bool:
        return os.path.exists(self.archive_path)
    
    def create_archive_dir(self) -> None:
        Path(os.path.dirname(self.archive_path)).mkdir(parents=True, exist_ok=True)


class _BaseCompositeFile(File):
    _ROOT_DATA_DIR = COMPOSITE_DIR

    def __init__(self, region: str, district: str, military_grid: str):
        super().__init__()
        self.region = region
        self.district = district
        self.mgrs = military_grid
    
    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.name)
    
    @property
    def s3_archive_path(self) -> str:
        return self.archive_path.replace(DATA_DIR + os.sep, '')

    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, mgrs: List[str] = None) -> List[str]:
        files = glob.glob(_BaseCompositeFile._ROOT_DATA_DIR + '/**/*', recursive=True)

        matching_files = []
        for f in files:
            match = re.match(regex, os.path.basename(f))
            if not match:
                continue
            group_dict = match.groupdict()
            if region is not None and group_dict['region'].lower() != region.lower():
                continue
            if district is not None and group_dict['district'].lower() != district.lower():
                continue
            if mgrs is not None and group_dict['mgrs'].lower() not in [m.lower() for m in mgrs]:
                continue
            matching_files.append(f)

        return sorted(matching_files)
    

class OpticalComposite(_BaseCompositeFile):
    regex = r'optical_composite_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_(?P<bands>[a-zA-Z0-9]{3}(?:_[a-zA-Z0-9]{3})*)\.tif$'
    
    def __init__(self, region: str, district: str, military_grid: str, bands: List[str]):
        super().__init__(region, district, military_grid)
        self.bands = sorted(bands)

    @property
    def name(self) -> str:
        base_string = f'optical_composite_{self.region}_{self.district}_{self.mgrs}_'
        for band in self.bands:
            base_string += band + ('_' if band != self.bands[-1] else '')
        return base_string + '.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, bands: List[str] = None,
                   mgrs: List[str] = None) -> List[str]:
        files = _BaseCompositeFile.find_files(OpticalComposite.regex, region, district, mgrs)

        matching_files = []
        for file in files:
            match = re.match(OpticalComposite.regex, os.path.basename(file))
            group_dict = match.groupdict()
            if bands is not None and group_dict['bands'].split('_') != sorted(bands):
                continue
            matching_files.append(file)

        return sorted(matching_files)

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
    

class OpticalCompositeSlice(_BaseCompositeFile):
    regex = r'optical_composite_slice_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_(?P<band>[^_]+)_(?P<left_bound>\d+)_(?P<right_bound>\d+)\.tif$'

    def __init__(self, region: str, district: str, military_grid: str, band: str, left_bound: int, right_bound: int):
        super().__init__(region, district, military_grid)
        if left_bound > right_bound:
            raise ValueError('Left bound must be less than right bound')
        self.band = band
        self.left_bound = left_bound
        self.right_bound = right_bound

    @property
    def name(self) -> str:
        return f'optical_composite_slice_{self.region}_{self.district}_{self.mgrs}_{self.band}_{self.left_bound}_' \
               f'{self.right_bound}.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, band: str = None,
                   mgrs: List[str] = None, left_bound: int = None, right_bound: int = None) -> List[str]:
        files = _BaseCompositeFile.find_files(OpticalCompositeSlice.regex, region, district, mgrs)

        matching_files = []
        for file in files:
            match = re.match(OpticalCompositeSlice.regex, os.path.basename(file))
            group_dict = match.groupdict()
            if band is not None and group_dict['band'].lower() != band.lower():
                continue
            if left_bound is not None and int(group_dict['left_bound']) != left_bound:
                continue
            if right_bound is not None and int(group_dict['right_bound']) != right_bound:
                continue
            matching_files.append(file)

        return sorted(matching_files)

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.mgrs, self.name)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'],
                       band=group_dict['band'], left_bound=int(group_dict['left_bound']),
                       right_bound=int(group_dict['right_bound']))
        else:
            return None


class MultiVariateComposite(_BaseCompositeFile):
    regex = r'multivariate_composite_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)\.tif$'

    def __init__(self, region: str, district: str, military_grid: str):
        super().__init__(region, district, military_grid)

    @property
    def name(self):
        return f'multivariate_composite_{self.region}_{self.district}_{self.mgrs}.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: List[str] = None):
        return _BaseCompositeFile.find_files(MultiVariateComposite.regex, region, district, mgrs)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'])
        else:
            return None
        

class _BaseSentinel2File(File):
    _ROOT_DATA_DIR = SENTINEL_2_DIR

    def __init__(self, region: str, district: str, utm_code: str, latitude_band: str, square: str, year: int,
                 month: int, day: int, band: str, sequence: int = 0):
        super().__init__()
        self.region = region
        self.district = district
        self.utm_code = utm_code
        self.latitude_band = latitude_band
        self.square = square
        self.year = year
        self.month = month
        self.day = day
        self.band = band
        self.sequence = sequence

    @property
    def mgrs_grid(self) -> str:
        return f'{self.utm_code}{self.latitude_band}{self.square}'

    @property
    def date_str(self) -> str:
        return f'{self.year}_{self.month}_{self.day}'
    
    @staticmethod
    def get_mgrs_dirs(region: str, district: str) -> Set[str]:
        mgrs_regex = r"\d{2}[a-zA-Z]{3}"
        mgrs_dirs = set()
        for dir in os.listdir(os.path.join(_BaseSentinel2File._ROOT_DATA_DIR, region, district)):
            match = re.match(mgrs_regex, dir)
            if match:
                mgrs_dirs.add(dir)
        return mgrs_dirs
    
    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.mgrs_grid, self.date_str, self.name)
    
    @staticmethod
    def _mgrs_from_utm_lat_square(utm_code: str, latitude_band: str, square: str) -> str:
        return f'{utm_code}{latitude_band}{square}'

    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, mgrs: str = None,
                   year: int = None, month: int = None, day: int = None, sequence: int = None, band: str = None) -> List[str]:
        files = glob.glob(_BaseSentinel2File._ROOT_DATA_DIR + '/**/*', recursive=True)

        matching_files = []
        for file in files:
            match = re.match(regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if mgrs is not None and _BaseSentinel2File._mgrs_from_utm_lat_square(
                    group_dict['utm_code'], group_dict['latitude_band'], group_dict['square']).lower() != mgrs.lower():
                continue
            if region is not None and group_dict['region'].lower() != region.lower():
                continue
            if district is not None and group_dict['district'].lower() != district.lower():
                continue
            if year is not None and int(group_dict['year']) != year:
                continue
            if month is not None and int(group_dict['month']) != month:
                continue
            if day is not None and int(group_dict['day']) != day:
                continue
            if sequence is not None and int(group_dict['sequence']) != sequence:
                continue
            if band is not None and group_dict['band'].lower() != band.lower():
                continue
            matching_files.append(file)

        return sorted(matching_files)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], utm_code=group_dict['utm_code'],
                       latitude_band=group_dict['latitude_band'],
                       square=group_dict['square'], year=int(group_dict['year']), month=int(group_dict['month']),
                       day=int(group_dict['day']), sequence=int(group_dict['sequence']), band=group_dict['band'])
        else:
            return None


class Sentinel2Tile(_BaseSentinel2File):
    regex = r'(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_(?P<band>B\d{2})\.jp2$'

    def __init__(self, region: str, district: str, utm_code: str, latitude_band: str, square: str, year: int, month: int, day: int, band: str,
                 sequence: int = 0):
        super().__init__(region, district, utm_code, latitude_band, square, year, month, day, band, sequence)

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.utm_code}_{self.latitude_band}_{self.square}_{self.year}_{self.month}_{self.day}_' \
               f'{self.sequence}_{self.band}.jp2'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None, year: int = None, month: int = None,
                   day: int = None, sequence: int = None, band: str = None) -> List[str]:
        return _BaseSentinel2File.find_files(Sentinel2Tile.regex, region, district, mgrs, year,
                                             month, day, sequence, band)

    @classmethod
    def create(cls, file_path: str):
        return super(Sentinel2Tile, cls).create(file_path)


class Sentinel2Cloud(_BaseSentinel2File):
    regex = r'(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_qi_MSK_CLOUDS_(?P<band>B\d{2}).gml'

    def __init__(self, region: str, district: str, utm_code: str, latitude_band: str, square: str, year: int, month: int, day: int,
                 sequence: int = 0, band: str = 'B00'):
        super().__init__(region, district, utm_code, latitude_band, square, year, month, day, band, sequence)

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.utm_code}_{self.latitude_band}_{self.square}_{self.year}_{self.month}_{self.day}_' \
               f'{self.sequence}_qi_MSK_CLOUDS_{self.band}.gml'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None, year: int = None, month: int = None,
                   day: int = None, sequence: int = None) -> List[str]:
        return _BaseSentinel2File.find_files(Sentinel2Cloud.regex, region, district, mgrs, year,
                                             month, day, sequence)

    @classmethod
    def create(cls, file_path: str):
        return super(Sentinel2Cloud, cls).create(file_path)
    

class _BaseNonOpticalBand(File):
    regex = r'(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]{5})\.tif$'
    _ROOT_DATA_DIR = None

    def __init__(self, region: str, district: str, mgrs: str):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.mgrs}.tif'

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(in_dir: str, region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        files = glob.glob(in_dir + '/**/*', recursive=True)
        
        matching_files = []
        for file in files:
            match = re.match(_BaseNonOpticalBand.regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if region is not None and group_dict['region'].lower() != region.lower():
                continue
            if district is not None and group_dict['district'].lower() != district.lower():
                continue
            if mgrs is not None and group_dict['mgrs'].lower() != mgrs.lower():
                continue
            matching_files.append(file)

        return sorted(matching_files)
    
    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'],
                       mgrs=group_dict['mgrs'])
        else:
            return None


class Elevation(_BaseNonOpticalBand):
    _ROOT_DATA_DIR = ELEVATION_DIR

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return _BaseNonOpticalBand.find_files(Elevation._ROOT_DATA_DIR, region, district, mgrs)

    @classmethod
    def create(cls, file_path: str):
        return super(Elevation, cls).create(file_path)


class Slope(_BaseNonOpticalBand):
    _ROOT_DATA_DIR = SLOPE_DIR

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return _BaseNonOpticalBand.find_files(Slope._ROOT_DATA_DIR, region, district, mgrs)

    @classmethod
    def create(cls, file_path: str):
        return super(Slope, cls).create(file_path)


class OSM(_BaseNonOpticalBand):
    _ROOT_DATA_DIR = OSM_DIR

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return _BaseNonOpticalBand.find_files(OSM._ROOT_DATA_DIR, region, district, mgrs)

    @classmethod
    def create(cls, file_path: str):
        return super(OSM, cls).create(file_path)


class SingleRegionTileMatch(File):
    _ROOT_DATA_DIR = TILE_DIR
    regex = r'tile_match_(?P<region>[^_]+)_?((?P<district>[^_]+)_)?((?P<mgrs>[^_]+)_)(?P<tile_size>\d+)\.csv$'

    def __init__(self, region: str, tile_size: int, district: str = None, military_grid: str = None):
        self.region = region
        self.district = district 
        self.military_grid = military_grid
        self.tile_size = tile_size
        super().__init__()

    @property
    def name(self):
        return f'tile_match_{self.region}_{self.district}_{self.military_grid}_{self.tile_size}.csv'

    @property
    def archive_path(self) -> str:
        base_path = os.path.join(self._ROOT_DATA_DIR, self.region)
        if self.district is not None:
            base_path = os.path.join(base_path, self.district)
        if self.military_grid is not None:
            base_path = os.path.join(base_path, self.military_grid)
        path = os.path.join(base_path, self.name)

        return path

    @staticmethod
    def find_files(tile_size: int, region: str = None, district: str = None, military_grid: str = None) -> List[str]:
        files = glob.glob(SingleRegionTileMatch._ROOT_DATA_DIR + '/**/*', recursive=True)
        matching_files = []
        for file in files:
            match = re.match(SingleRegionTileMatch.regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if region is not None and group_dict['region'].lower() != region.lower():
                continue
            if district is not None and group_dict['district'].lower() != district.lower():
                continue
            if military_grid is not None and group_dict['mgrs'].lower() != military_grid.lower():
                continue
            if int(group_dict['tile_size']) != tile_size:
                continue
            matching_files.append(file)

        return sorted(matching_files)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'], tile_size=int(group_dict['tile_size']))
        else:
            return None


class MultiRegionTileMatch(File):
    _ROOT_DATA_DIR = MULTI_REGION_TILE_MATCH
    regex = r'tile_match_(?P<regions>\w+(?:_\w+)*)_(?P<tile_size>\d+)\.csv$'

    def __init__(self, regions: List[str], tile_size: int):
        self.regions = sorted(regions)
        self.tile_size = tile_size
        super().__init__()

    @property
    def name(self):
        return f"tile_match_{'_'.join(self.regions)}_{self.tile_size}.csv"

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, '_'.join(self.regions), self.name)
        
    @staticmethod
    def find_files(regions: List[str] = None, tile_size: int = None):
        files = glob.glob(MultiRegionTileMatch._ROOT_DATA_DIR + '/**/*', recursive=True)
        matching_files = []
        for file in files:
            match = re.match(MultiRegionTileMatch.regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if regions is not None and group_dict['regions'].split('_') != sorted(regions):
                continue
            if tile_size is not None and int(group_dict['tile_size']) != tile_size:
                continue
            matching_files.append(file)

        return sorted(matching_files)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(regions=group_dict['regions'].split('_'), tile_size=int(group_dict['tile_size']))
        else:
            return None
        

class _BaseTileFile(File):
    _ROOT_DATA_DIR = TILE_DIR

    def __init__(self, region: str, district: str, military_grid: str, tile_size: int, x_min: int, y_min: int):
        self.region = region
        self.district = district
        self.military_grid = military_grid
        self.tile_size = tile_size
        self.x_min = x_min
        self.y_min = y_min
        super().__init__()

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.military_grid, str(self.tile_size), self.name)

    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, military_grid: str = None, tile_size: int = None) -> List[str]:
        files = glob.glob(_BaseTileFile._ROOT_DATA_DIR + '/**/*', recursive=True)
        
        matching_files = []
        for file in files:
            match = re.match(regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if region is not None and group_dict['region'].lower() != region.lower():
                continue
            if district is not None and group_dict['district'].lower() != district.lower():
                continue
            if military_grid is not None and group_dict['mgrs'].lower() != military_grid.lower():
                continue
            if tile_size is not None and int(group_dict['tile_size']) != tile_size:
                continue
            matching_files.append(file)

        return sorted(matching_files)
    
    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'], tile_size=int(group_dict['tile_size']),
                        x_min=int(group_dict['x_min']), y_min=int(group_dict['y_min']))
        else:
            return None


class Tile(_BaseTileFile):
    regex = r'^(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_(?P<tile_size>\d+)_(?P<x_min>\d+)_(?P<y_min>\d+)\.tif$'

    def __init__(self, region: str, district: str, military_grid: str, tile_size: int, x_min: int, y_min: int):
        super().__init__(region, district, military_grid, tile_size, x_min, y_min)

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.mgrs}_{self.tile_size}_{self.x_min}_{self.y_min}.tif'
    
    @staticmethod
    def find_files(region: str = None, district: str = None, military_grid: str = None, tile_size: int = None) -> List[str]:
        return super().__init__(Tile.regex, region, district, military_grid, tile_size)

    @classmethod
    def create(cls, file_path: str):
        return super(Tile, cls).create(file_path)


class PyTorch(_BaseTileFile):
    regex = r'^(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_(?P<tile_size>\d+)_(?P<x_min>\d+)_(?P<y_min>\d+)\.pt$'

    def __init__(self, region: str, district: str, military_grid: str, tile_size: int, x_min: int, y_min: int):
        super().__init__(region, district, military_grid, tile_size, x_min, y_min)

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.mgrs}_{self.tile_size}_{self.x_min}_{self.y_min}.tif'
    
    @staticmethod
    def find_files(region: str = None, district: str = None, military_grid: str = None, tile_size: int = None) -> List[str]:
        return super().__init__(PyTorch.regex, region, district, military_grid, tile_size)

    @classmethod
    def create(cls, file_path: str):
        return super(PyTorch, cls).create(file_path)
        

class _BaseInferenceFiles(File):
    _ROOT_DATA_DIR = None
    base_layers = 'red|blue|green|nir|osm-water|osm-boundary|elevation|slope'

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float, tile_size: int, best: bool = False):
        if not 0 < len(layers) <= 3:
            raise ValueError('Must only between 1 and 3 layer(s)')
        self.regions = sorted(regions)
        self.architecture = architecture
        self.layers = sorted(layers)
        self.epoch = epoch
        self.ratio = ratio
        self.tile_size = tile_size
        self.best = best
        super().__init__()

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, '_'.join(self.regions), self.architecture, f'r{self.ratio}', self.name)
    
    def s3_archive_path(self) -> str:
        return self.archive_path.replace(DATA_DIR + os.sep, '')
    
    @staticmethod
    def find_files(in_dir: str, regex: str, regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None, ratio: float = None, tile_size: int = None,
                   best: bool = False) -> List[str]:
        if not 0 < len(layers) <= 3:
            raise ValueError('Must only between 1 and 3 layer(s)')
        
        files = glob.glob(in_dir + '/**/*', recursive=True)

        matching_files = []
        for file in files:
            match = re.match(regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.group_dict()
            if layers is not None:
                layers = sorted(layers)
                layer_match = True
                for i, layer in enumerate(layers):
                    if layer != group_dict[f'layer_{i+1}']:
                        layer_match = False
                        break
                if not layer_match:
                    continue
            if regions is not None and group_dict['regions'].lower() != '_'.join(sorted(regions)).lower():
                continue
            if architecture is not None and group_dict['architecture'] != architecture:
                continue
            if ratio is not None and float(group_dict['ratio']) != ratio:
                continue 
            if best and group_dict['best'] is None:
                continue
            if epoch is not None and epoch != group_dict['epoch']:
                continue
            if tile_size is not None and tile_size != int(group_dict['tile_size']):
                continue
            matching_files.append(file)

        return sorted(matching_files)
    
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
            return cls(regions=group_dict['regions'].split('_'), architecture=group_dict['architecture'], tile_size=int(group_dict['tile_size']), layers=layers, epoch=group_dict['epoch'],
                       ratio=group_dict['ratio'], best=group_dict['best'] is not None)
        else:
            return None


class TrainedModel(_BaseInferenceFiles):
    _ROOT_DATA_DIR = MODEL_DIR
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf'^(?P<regions>\w+(?:_\w+)*)_(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_ts(?P<tile_size>\d+)_(?P<layer_1>{base_layers})_?((?P<layer_2>{base_layers}_)?((?P<layer_3>{base_layers})_)epoch(?P<epoch>\d+)?(?P<best>_best)\.tar$'

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float, tile_size: int, best: bool = False):
        super().__init__(regions, architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = '_'.join(self.regions) + '_' + self.architecture + f'_r{self.ratio}_' + f'ts{self.tile_size}'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '.tar'
        return base_str
    
    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None, ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return super().find_files(TrainedModel._ROOT_DATA_DIR, TrainedModel.regex, regions, architecture, layers, epoch, ratio, tile_size, best)

    @classmethod
    def create(cls, file_path: str):
        return super(TrainedModel, cls).create(file_path)


class InferenceResultsCSV(_BaseInferenceFiles):
    _ROOT_DATA_DIR = INFERENCE_RESULTS_DIR
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf'^(?P<regions>\w+(?:_\w+)*)_(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_(?P<layer_1>{base_layers})_((?P<layer_2>{base_layers}_)?((?P<layer_3>{base_layers})_)epoch(?P<epoch>\d+)?(?P<best>_best)_results\.csv$'

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float, tile_size: int, best: bool = False):
        super().__init__(regions, architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = '_'.join(self.regions) + '_' + self.architecture + f'_r{self.ratio}_' + f'ts{self.tile_size}'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '.csv'
        return base_str

    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None, ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return super().find_files(InferenceResultsCSV._ROOT_DATA_DIR, InferenceResultsCSV.regex, regions, architecture, layers, epoch, ratio, tile_size, best)

    @classmethod
    def create(cls, file_path: str):
        return super(InferenceResultsCSV, cls).create(file_path)


class InferenceResultsShapefile(_BaseInferenceFiles):
    _ROOT_DATA_DIR = INFERENCE_RESULTS_DIR
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf'^(?P<regions>\w+(?:_\w+)*)_(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_(?P<layer_1>{base_layers})_((?P<layer_2>{base_layers}_)?((?P<layer_3>{base_layers})_)epoch(?P<epoch>\d+)?(?P<best>_best)_results\.shp$'

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float, tile_size: int, best: bool = False):
        super().__init__(regions, architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = '_'.join(self.regions) + '_' + self.architecture + f'_r{self.ratio}_' + '_'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '_results' '.shp'
        return base_str

    @property
    def archive_path(self) -> str:
        sub_dir = '_'.join(self.layers) + f'_{self.epoch}' + '_best' if self.best else '' + 'shapefile'
        path = os.path.join(self._ROOT_DATA_DIR, '_'.join(self.regions), self.architecture, f'r{self.ratio}',
                            sub_dir, self.name)
        return path

    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None, ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return super().find_files(InferenceResultsShapefile._ROOT_DATA_DIR, InferenceResultsShapefile.regex, regions, architecture, layers, epoch, ratio, tile_size, best)

    @classmethod
    def create(cls, file_path: str):
        return super(InferenceResultsShapefile, cls).create(file_path)


class _BaseDatasetSplit(File):
    _ROOT_DATA_DIR = TRAIN_VALIDATE_SPLIT_DIR

    def __init__(self, regions: List[str], ratio: int, tile_size: int):
        self.regions = sorted(regions)
        self.ratio = ratio
        self.tile_size = tile_size
        super().__init__()

    @property
    def archive_path(self) -> str:
        return os.path.join(TRAIN_VALIDATE_SPLIT_DIR, self.name)

    @staticmethod
    def find_files(regex: str, regions: List[str] = None, ratio: int = None) -> List[str]:
        files = glob.glob(_BaseDatasetSplit._ROOT_DATA_DIR + '/**/*', recursive=True)

        matching_files = []
        for file in files:
            match = re.match(regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if regions is not None:
                regions = '_'.join(sorted(regions))
                if regions.lower() != group_dict['regions'].lower():
                    continue
            if ratio is not None and ratio != int(group_dict['ratio']):
                continue
            matching_files.append(file)

        return sorted(matching_files)
    
    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(regions=group_dict['regions'].split('_'), ratio=int(group_dict['ratio']), tile_size=int(group_dict['tile_size']))
        else:
            return None


class TrainSplit(_BaseDatasetSplit):
    regex = r'train_(?P<regions>\w+(?:_\w+)*)_(?P<ratio>\d+)_(?P<tile_size>\d+)meter\.csv$'

    def __init__(self, regions: List[str], ratio: int, tile_size: int):
        super().__init__(regions, ratio, tile_size)

    @property
    def name(self) -> str:
        return f"train_{'_'.join(self.regions)}_{self.ratio}_{self.tile_size}meter.csv"

    @staticmethod
    def find_files(regions: List[str], ratio: int) -> List[str]:
        return super().find_files(TrainSplit.regex, regions, ratio)

    @classmethod
    def create(cls, file_path: str):
        return super(TrainSplit, cls).create(file_path)


class ValidateSplit(_BaseDatasetSplit):
    regex = r'validate_(?P<regions>\w+(?:_\w+)*)_(?P<ratio>\d+)_(?P<tile_size>\d+)meter\.csv$'

    def __init__(self, regions: List[str], ratio: int, tile_size: int):
        super().__init__(regions, ratio, tile_size)

    @property
    def name(self) -> str:
        return f"validate_{'_'.join(self.regions)}_{self.ratio}_{self.tile_size}meter.csv"

    @staticmethod
    def find_files(regions: List[str], ratio: int) -> List[str]:
        return super().find_files(ValidateSplit.regex, regions, ratio)

    @classmethod
    def create(cls, file_path: str):
        return super(ValidateSplit, cls).create(file_path)
