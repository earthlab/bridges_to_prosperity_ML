import os
import re
from typing import List
from definitions import COMPOSITE_DIR, B2P_DIR, ELEVATION_DIR, SLOPE_DIR, SENTINEL_2_DIR, OSM_DIR, \
    MODEL_DIR, INFERENCE_RESULTS_DIR, TILE_DIR, TRAIN_VALIDATE_SPLIT_DIR, MULTI_REGION_TILE_MATCH, DATA_DIR
import glob
from pathlib import Path
from abc import ABC, abstractmethod, abstractproperty


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
    
    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def archive_path(self):
        pass


class _BaseCompositeFile(File):
    _ROOT_DATA_DIR = COMPOSITE_DIR

    def __init__(self, region: str, district: str, military_grid: str):
        super().__init__()
        self.region = region
        self.district = district
        self.mgrs = military_grid
    
    def archive_path(self, create_dir: bool = False) -> str:
        path = os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.name)
        if create_dir:
            Path(path).mkdir(parents=True, exist_ok=True)
        return path
    
    def s3_archive_path(self) -> str:
        return self.archive_path().replace(DATA_DIR + os.sep, '')
    
    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, mgrs: List[str] = None) -> List[str]:
        files = glob.glob(_BaseCompositeFile._ROOT_DATA_DIR + '/**/*', recursive=True)

        matching_files = []
        for f in files:
            match = re.match(regex, os.path.basename(f))
            if not match:
                continue
            group_dict = match.groupdict()
            if region is not None and group_dict['region'] != region:
                continue
            if district is not None and group_dict['district'] != district:
                continue
            if mgrs is not None and group_dict['mgrs'] not in mgrs:
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
    def find_files(region: str = None, district: str = None, bands: List[str] = None, mgrs: List[str] = None) -> List[str]:
        files = _BaseCompositeFile.find_files(OpticalComposite.regex, region, district, mgrs)

        if bands is not None:
            matching_files = []
            for file in files:
                match = re.match(OpticalComposite.regex, os.path.basename(file))
                group_dict = match.groupdict()
                if bands is not None and group_dict['bands'].split('_') != sorted(bands):
                    continue
                matching_files.append(file)
            files = matching_files
        return sorted(files)

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
        

class MultiVariateComposite(_BaseCompositeFile):
    _ROOT_DATA_DIR = COMPOSITE_DIR
    regex = r'multivariate_composite_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)\.tif$'

    def __init__(self, region: str, district: str, military_grid: str):
        super().__init__(region, district, military_grid)

    @property
    def name(self):
        base_string = f'multivariate_composite_{self.region}_{self.district}_{self.mgrs}'
        return base_string + '.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: List[str] = None):
        return _BaseCompositeFile.find_files(MultiVariateComposite.regex, region, district, mgrs)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], mgrs=group_dict['mgrs'])
        else:
            return None
        

class _BaseSentinel2File(File):
    _ROOT_DATA_DIR = SENTINEL_2_DIR

    def __init__(self, utm_code: str, latitude_band: str, square: str, year: int, month: int, day: int, band: str,
                 sequence: int = 0):
        super().__init__()
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
    
    def archive_path(self, region: str, district: str, create_dir: bool = False) -> str:
        path = os.path.join(self._ROOT_DATA_DIR, region, district, self.mgrs_grid, self.date_str, self.name)
        if create_dir:
            Path(path).mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, utm_code: str = None, latitude_band: str = None, square: str = None,
                   year: int = None, month: int = None, day: int = None, sequence: int = None, band: str = None) -> List[str]:
        if region is None and district is not None:
            raise ValueError('Must specify region if district is specified')

        in_dir = _BaseSentinel2File._ROOT_DATA_DIR

        if region is not None:
            in_dir = os.path.join(in_dir, region)
        if district is not None:
            in_dir = os.path.join(in_dir, district)

        files = glob.glob(in_dir + '/**/*', recursive=True)

        matching_files = []
        for file in files:
            match = re.match(regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if utm_code is not None and group_dict['utm_code'] != utm_code:
                continue
            if latitude_band is not None and group_dict['latitude_band'] != latitude_band:
                continue
            if square is not None and group_dict['square'] != square:
                continue
            if year is not None and int(group_dict['year']) != year:
                continue
            if month is not None and int(group_dict['month']) != month:
                continue
            if day is not None and int(group_dict['day']) != day:
                continue
            if sequence is not None and int(group_dict['sequence']) != sequence:
                continue
            if band is not None and group_dict['band'] != band:
                continue
            matching_files.append(file)

        return sorted(matching_files)


class Sentinel2Tile(_BaseSentinel2File):
    regex = r'(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_(?P<band>B\d{2})\.jp2$'

    def __init__(self, utm_code: str, latitude_band: str, square: str, year: int, month: int, day: int, band: str,
                 sequence: int = 0):
        super().__init__(utm_code, latitude_band, square, year, month, day, band, sequence)

    @property
    def name(self) -> str:
        return f'{self.utm_code}_{self.latitude_band}_{self.square}_{self.year}_{self.month}_{self.day}_' \
               f'{self.sequence}_{self.band}.jp2'

    def archive_path(self, region: str, district: str, create_dir: bool = False) -> str:
        return super().archive_path(region, district, create_dir)

    @staticmethod
    def find_files(region: str = None, district: str = None, utm_code: str = None, latitude_band: str = None, square: str = None,
                   year: int = None, month: int = None, day: int = None, sequence: int = None, band: str = None) -> List[str]:
        return _BaseSentinel2File.find_files(Sentinel2Tile.regex, region, district, utm_code, latitude_band, square, year,
                                             month, day, sequence, band)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(utm_code=group_dict['utm_code'], latitude_band=group_dict['latitude_band'],
                       square=group_dict['square'], year=int(group_dict['year']), month=int(group_dict['month']),
                       day=int(group_dict['day']), band=group_dict['band'], sequence=int(group_dict['sequence']))
        else:
            return None


class Sentinel2Cloud(_BaseSentinel2File):
    regex = r'(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_qi_MSK_CLOUDS_B00.gml'

    def __init__(self, utm_code: str, latitude_band: str, square: str, year: int, month: int, day: int,
                 sequence: int = 0):
        super().__init__(utm_code, latitude_band, square, year, month, day, 'B00', sequence)

    @property
    def name(self) -> str:
        return f'{self.utm_code}_{self.latitude_band}_{self.square}_{self.year}_{self.month}_{self.day}_' \
               f'{self.sequence}_qi_MSK_CLOUDS_B00.gml'

    def archive_path(self, region: str, district: str, create_dir: bool = False) -> str:
        return super().archive_path(region, district, create_dir)

    @staticmethod
    def find_files(region: str = None, district: str = None, utm_code: str = None, latitude_band: str = None, square: str = None,
                   year: int = None, month: int = None, day: int = None, sequence: int = None) -> List[str]:
        return _BaseSentinel2File.find_files(Sentinel2Cloud.regex, region, district, utm_code, latitude_band, square, year,
                                             month, day, sequence)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(utm_code=group_dict['utm_code'], latitude_band=group_dict['latitude_band'],
                       square=group_dict['square'], year=int(group_dict['year']), month=int(group_dict['month']),
                       day=int(group_dict['day']), sequence=int(group_dict['sequence']))
        else:
            return None
        

class _BaseNonOpticalBand(File):
    regex = r'(?P<region>[^_]+)_(?P<district>[^_]+)(?:_(?P<mgrs>[^_]{5}))?\.tif$'
    _ROOT_DATA_DIR = None

    def __init__(self, region: str, district: str, mgrs: str = None):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def name(self) -> str:
        base_string = f'{self.region}_{self.district}'
        if self.mgrs is not None:
            base_string += f'_{self.mgrs}'
        return base_string + '.tif'

    def archive_path(self, create_dir: bool = False) -> str:
        path = os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.name)
        if create_dir:
            Path(path).mkdir(parents=True, exist_ok=True)      

        return path

    @staticmethod
    def find_files(in_dir: str, region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        files = glob.glob(in_dir + '/**/*', recursive=True)
        
        matching_files = []
        for file in files:
            match = re.match(_BaseNonOpticalBand.regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if region is not None and group_dict['region'] != region:
                continue
            if district is not None and group_dict['district'] != district:
                continue
            if mgrs is not None and group_dict['mgrs'].lower() != mgrs.lower():
                continue
            matching_files.append(file)

        return sorted(matching_files)


class Elevation(_BaseNonOpticalBand):
    _ROOT_DATA_DIR = ELEVATION_DIR

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return super().find_files(Elevation._ROOT_DATA_DIR, region, district, mgrs)

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


class Slope(_BaseNonOpticalBand):
    _ROOT_DATA_DIR = SLOPE_DIR

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return super().find_files(Slope._ROOT_DATA_DIR, region, district, mgrs)

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


class OSM(_BaseNonOpticalBand):
    _ROOT_DATA_DIR = OSM_DIR

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return super().find_files(OSM._ROOT_DATA_DIR, region, district, mgrs)

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


class SingleRegionTileMatch(File):
    _ROOT_DATA_DIR = TILE_DIR
    regex = r'tile_match_(?P<tile_size>\d+)\.csv$'

    def __init__(self, tile_size: int):
        self.tile_size = tile_size
        super().__init__()

    @property
    def name(self):
        return f'tile_match_{self.tile_size}.csv'

    def archive_path(self, region: str, district: str = None, military_grid: str = None, create_dir: bool = False) -> str:
        base_path = os.path.join(self._ROOT_DATA_DIR, region)
        if district is not None:
            base_path = os.path.join(base_path, district)
        if military_grid is not None:
            base_path = os.path.join(base_path, military_grid)
        base_path = os.path.join(base_path, self.name)

        if create_dir:
            Path(base_path).mkdir(parents=True, exist_ok=True)

        return base_path

    @staticmethod
    def find_files(tile_size: int, region: str = None, district: str = None, military_grid: str = None) -> List[str]:
        if region is None and district is not None:
            raise ValueError('Must specify region if district is specified')
        
        if district is None and military_grid is not None:
            raise ValueError('Must specify region and district if military grid is specified')

        in_dir = SingleRegionTileMatch._ROOT_DATA_DIR
        if region is not None:
            in_dir = os.path.join(in_dir, region)
        if district is not None:
            in_dir = os.path.join(in_dir, district)
        if military_grid is not None:
            in_dir = os.path.join(in_dir, military_grid)

        files = glob.glob( + '/**/*', recursive=True)
        matching_files = []
        for file in files:
            match = re.match(SingleRegionTileMatch.regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if int(group_dict['tile_size']) != tile_size:
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
            return cls(tile_size=int(group_dict['tile_size']))
        else:
            return None


class MultiRegionTileMatch(File):
    _ROOT_DATA_DIR = MULTI_REGION_TILE_MATCH
    regex = r'tile_match_(?P<tile_size>\d+)\.csv$'

    def __init__(self, tile_size: int):
        self.tile_size = tile_size
        super().__init__()

    @property
    def name(self):
        return f'tile_match_{self.tile_size}.csv'

    def archive_path(self, regions: List[str], create_dir: bool = False) -> str:
        regions = sorted(regions)
        path = os.path.join(self._ROOT_DATA_DIR, '_'.join(regions), self.name)
        if create_dir:
            Path(path).mkdir(parents=True, exist_ok=True)
        return path
        
    @staticmethod
    def find_files(regions: List[str] = None):
        in_dir = MultiRegionTileMatch._ROOT_DATA_DIR
        if regions is not None:
            regions = sorted(regions)
            in_dir = os.path.join(in_dir, '_'.join(regions))
        files = glob.glob(in_dir + '/**/*', recursive=True)
        matching_files = []
        for file in files:
            match = re.match(MultiRegionTileMatch.regex, os.path.basename(file))
            if not match:
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
            return cls(tile_size=int(group_dict['tile_size']))
        else:
            return None
        

class _BaseTileFile(File):
    _ROOT_DATA_DIR = TILE_DIR

    def __init__(self, x_min: int, y_min: int):
        self.x_min = x_min
        self.y_min = y_min
        super().__init__()

    def archive_path(self, region: str, district: str, military_grid: str, tile_size: int, create_dir: bool = False) -> str:
        path = os.path.join(self._ROOT_DATA_DIR, region, district, military_grid, str(tile_size), self.name)
        if create_dir:
            Path(path).mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, military_grid: str = None, tile_size: int = None) -> List[str]:
        if district is not None and region is None:
            raise ValueError('Must specify region if district is specified')
        if military_grid is not None and district is None:
            raise ValueError('Must specify region and district if military grid is specified')
        if tile_size is not None and military_grid is None:
            raise ValueError('Must specify region, district, and military grid if tile size is specified')

        in_dir = _BaseTileFile._ROOT_DATA_DIR
        if region is not None:
            in_dir = os.path.join(in_dir, region)
        if district is not None:
            in_dir = os.path.join(in_dir, district)
        if military_grid is not None:
            in_dir = os.path.join(in_dir, military_grid)
        if tile_size is not None:
            in_dir = os.path.join(in_dir, tile_size)

        files = glob.glob(in_dir + '/**/*', recursive=True)
        
        matching_files = []
        for file in files:
            match = re.match(regex, os.path.basename(file))
            if not match:
                continue
            matching_files.append(file)

        return sorted(matching_files)


class Tile(_BaseTileFile):
    regex = r'^(?P<x_min>\d+)_(?P<y_min>\d+)\.tif$'

    def __init__(self, x_min: int, y_min: int):
        super().__init__(x_min, y_min)

    @property
    def name(self) -> str:
        return f'{self.x_min}_{self.y_min}.tif'
    
    @staticmethod
    def find_files(region: str = None, district: str = None, military_grid: str = None, tile_size: int = None) -> List[str]:
        return super().__init__(Tile.regex, region, district, military_grid, tile_size)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(x_min=int(group_dict['x_min']), y_min=int(group_dict['y_min']))
        else:
            return None


class PyTorch(_BaseTileFile):
    regex = r'^(?P<x_min>\d+)_(?P<y_min>\d+)\.pt$'

    def __init__(self, x_min: int, y_min: int):
        super().__init__(x_min, y_min)

    @property
    def name(self) -> str:
        return f'{self.x_min}_{self.y_min}.tif'
    
    @staticmethod
    def find_files(region: str = None, district: str = None, military_grid: str = None, tile_size: int = None) -> List[str]:
        return super().__init__(PyTorch.regex, region, district, military_grid, tile_size)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        regex = cls.regex
        match = re.match(regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(x_min=int(group_dict['x_min']), y_min=int(group_dict['y_min']))
        else:
            return None
        

class _BaseInferenceFiles(File):
    _ROOT_DATA_DIR = None
    base_layers = 'red|blue|green|nir|osm-water|osm-boundary|elevation|slope'

    def __init__(self, architecture: str, layers: List[str], epoch: int, ratio: float, tile_size: int, best: bool = False):
        if not 0 < len(layers) <= 3:
            raise ValueError('Must only between 1 and 3 layer(s)')
        self.architecture = architecture
        self.layers = sorted(layers)
        self.epoch = epoch
        self.ratio = ratio
        self.tile_size = tile_size
        self.best = best
        super().__init__()

    def archive_path(self, regions: List[str], create_dir: bool = False) -> str:
        path = os.path.join(self._BASE_DATA_DIR, '_'.join(sorted(regions)), self.architecture, f'r{self.ratio}', self.name)
        if create_dir:
            Path(path).mkdir(parents=True, exist_ok=True)
        return path
    
    def s3_archive_path(self, regions: List[str]) -> str:
        return self.archive_path(regions).replace(DATA_DIR + os.sep, '')
    
    @staticmethod
    def find_files(in_dir: str, regex: str, regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None, ratio: float = None, tile_size: int = None,
                   best: bool = False) -> List[str]:
        if not 0 < len(layers) <= 3:
            raise ValueError('Must only between 1 and 3 layer(s)')

        if architecture is None and regions is not None:
            raise ValueError('Must specify region if architecture is specified')
        
        if ratio is None and architecture is None:
            raise ValueError('Must specify architecture if ratio is specified')

        if regions is not None:
            regions = sorted(regions)
            in_dir = os.path.join(in_dir, '_'.join(regions))
        if architecture is not None:
            in_dir = os.path.join(in_dir,  architecture)
        if ratio is not None:
            in_dir = os.path.join(in_dir, f'r{ratio}')
        
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
            if best and group_dict['best'] is None:
                continue
            if epoch is not None and epoch != group_dict['epoch']:
                continue
            if tile_size is not None and tile_size != int(group_dict['tile_size']):
                continue
            matching_files.append(file)

        return sorted(matching_files)


class TrainedModel(_BaseInferenceFiles):
    _ROOT_DATA_DIR = MODEL_DIR
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf'^(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_ts(?P<tile_size>\d+)_(?P<layer_1>{base_layers})_((?P<layer_2>{base_layers}_)?((?P<layer_3>{base_layers})_)epoch(?P<epoch>\d+)?(?P<best>_best)\.tar$'

    def __init__(self, architecture: str, layers: List[str], epoch: int, ratio: float, tile_size: int, best: bool = False):
        super().__init__(architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = self.architecture + f'_r{self.ratio}_' + f'ts{self.tile_size}'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '.tar'
        return base_str
    
    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None, ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return super().find_files(TrainedModel._ROOT_DATA_DIR, TrainedModel.regex, regions, architecture, layers, epoch, ratio, tile_size, best)

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
            return cls(architecture=group_dict['architecture'], tile_size=int(group_dict['tile_size']), layers=layers, epoch=group_dict['epoch'],
                       ratio=group_dict['ratio'], best=group_dict['best'] is not None)
        else:
            return None


class InferenceResultsCSV(_BaseInferenceFiles):
    _ROOT_DATA_DIR = INFERENCE_RESULTS_DIR
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf'^(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_(?P<layer_1>{base_layers})_((?P<layer_2>{base_layers}_)?((?P<layer_3>{base_layers})_)epoch(?P<epoch>\d+)?(?P<best>_best)_results\.csv$'

    def __init__(self, architecture: str, layers: List[str], epoch: int, ratio: float, tile_size: int, best: bool = False):
        super().__init__(architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = self.architecture + f'_r{self.ratio}_' + f'ts{self.tile_size}'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '.csv'
        return base_str

    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None, ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return super().find_files(InferenceResultsCSV._ROOT_DATA_DIR, InferenceResultsCSV.regex, regions, architecture, layers, epoch, ratio, tile_size, best)

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
            return cls(architecture=group_dict['architecture'], tile_size=int(group_dict['tile_size']), layers=layers, epoch=group_dict['epoch'],
                       ratio=group_dict['ratio'], best=group_dict['best'] is not None)
        else:
            return None


class InferenceResultsShapefile(_BaseInferenceFiles):
    _ROOT_DATA_DIR = INFERENCE_RESULTS_DIR
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf'^(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_(?P<layer_1>{base_layers})_((?P<layer_2>{base_layers}_)?((?P<layer_3>{base_layers})_)epoch(?P<epoch>\d+)?(?P<best>_best)_results\.shp$'

    def __init__(self, architecture: str, layers: List[str], epoch: int, ratio: float, tile_size: int, best: bool = False):
        super().__init__(architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = self.architecture + f'_r{self.ratio}_' + '_'

        for layer in self.layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + '_best' if self.best else '' + '_results' '.shp'
        return base_str

    def archive_path(self, regions: List[str]):
        sub_dir = '_'.join(self.layers) + f'_{self.epoch}' + '_best' if self.best else '' + 'shapefile'
        return os.path.join(self._ROOT_DATA_DIR, '_'.join(sorted(regions)), self.architecture, f'r{self.ratio}',
                            sub_dir, self.name)

    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None, ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return super().find_files(InferenceResultsShapefile._ROOT_DATA_DIR, InferenceResultsShapefile.regex, regions, architecture, layers, epoch, ratio, tile_size, best)

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
            return cls(architecture=group_dict['architecture'], tile_size=int(group_dict['tile_size']), layers=layers, epoch=group_dict['epoch'],
                       ratio=group_dict['ratio'], best=group_dict['best'] is not None)
        else:
            return None


class _BaseDatasetSplit(File):
    _ROOT_DATA_DIR = TRAIN_VALIDATE_SPLIT_DIR

    def __init__(self, regions: List[str], ratio: int):
        self.regions = sorted(regions)
        self.ratio = ratio
        super().__init__()

    def archive_path(self, create_dir: bool = False) -> str:
        path = os.path.join(TRAIN_VALIDATE_SPLIT_DIR, self.name)
        if create_dir:
            Path(path).mkdir(parents=True, exist_ok=True)
        return path

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
                if regions != group_dict['regions']:
                    continue
            if ratio is not None and ratio != int(group_dict['ratio']):
                continue
            matching_files.append(file)

        return sorted(matching_files)


class TrainSplit(_BaseDatasetSplit):
    regex = r'train_(?P<regions>\w+(?:_\w+)*)_(?P<ratio>\d+)\.csv$'

    def __init__(self, regions: List[str], ratio: int):
        super().__init__(regions, ratio)

    @property
    def name(self) -> str:
        return f"train_{'_'.join(self.regions)}_{self.ratio}.csv"

    @staticmethod
    def find_files(regions: List[str], ratio: int) -> List[str]:
        return super().find_files(TrainSplit.regex, regions, ratio)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(regions=group_dict['regions'].split('_'), ratio=int(group_dict['ratio']))
        else:
            return None


class ValidateSplit(File):
    regex = r'validate_(?P<regions>\w+(?:_\w+)*)_(?P<ratio>\d+)\.csv$'

    def __init__(self, regions: List[str], ratio: int):
        super().__init__(regions, ratio)

    @property
    def name(self) -> str:
        return f"validate_{'_'.join(self.regions)}_{self.ratio}.csv"

    @staticmethod
    def find_files(regions: List[str], ratio: int) -> List[str]:
        return super().find_files(TrainSplit.regex, regions, ratio)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(regions=group_dict['regions'].split('_'), ratio=int(group_dict['ratio']))
        else:
            return None
