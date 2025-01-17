"""
Contains factory classes for each file type used in the project. These factory classes are used to create file names,
archive paths, and search for files throughout the rest of the project.
"""

import os
import re
from typing import List, Set, Type
from definitions import COMPOSITE_DIR, ELEVATION_DIR, SLOPE_DIR, SENTINEL_2_DIR, OSM_DIR, \
    MODEL_DIR, INFERENCE_RESULTS_DIR, TILE_DIR, TRAIN_VALIDATE_SPLIT_DIR, MULTI_REGION_TILE_MATCH, DATA_DIR
import glob
from pathlib import Path
from abc import ABC, abstractmethod
import tarfile


def ci_str_sort(str_list: List[str]) -> List[str]:
    return sorted(str_list, key=str.casefold)


def find_child_classes_with_attribute(root_class: Type, attribute: str) -> List[Type]:
    with_regex = []
    for child_class in root_class.__subclasses__():
        if hasattr(child_class, attribute):
            with_regex.append(child_class)
        else:
            with_regex += find_child_classes_with_attribute(child_class, attribute)

    return with_regex


class File(ABC):
    """
    Base class for the rest of the file types. Defines the abstract properties and methods of each child file type
     class
    """

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def create(cls, file_path: str) -> Type:
        """
        Tries to match the input file name with each child class's regex. Once the regex matches, an instance of that
        class created with the input file name is returned
        Args:
            file_path (str): The name or path of the input file for which to create a file type object
        """
        child_classes_with_regex = find_child_classes_with_attribute(cls, 'regex')
        for child_class in child_classes_with_regex:
            if re.match(child_class.regex, os.path.basename(file_path)) is not None:
                return child_class.create(os.path.basename(file_path))
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Abstract method creating contract with child classes for name property. The name property will define
        how the file is named for a unique set of input parameters
        """
        pass

    @property
    @abstractmethod
    def archive_path(self) -> str:
        """
        Abstract method creating contract with child classes for archive_path property. The archive_path property will
        define the path to the file for a unique set of input parameters
        """
        pass

    @property
    def exists(self) -> bool:
        """
        Returns true if the file path exists and False if not
        """
        return os.path.exists(self.archive_path)

    def create_archive_dir(self) -> None:
        """
        Creates all the directories to the archive path
        """
        Path(os.path.dirname(self.archive_path)).mkdir(parents=True, exist_ok=True)


class _BaseCompositeFile(File):
    """
    Base attributes and methods for optical, multivariate, and slices composite files
    """
    _ROOT_DATA_DIR = COMPOSITE_DIR
    ROOT_S3_DIR = os.path.basename(_ROOT_DATA_DIR)

    def __init__(self, region: str, district: str, military_grid: str):
        """
        Initializes base composite file
        Args:
            region (str): Name of the region that the composite is in
            district (str): Name of the district that the composite is in
            military_grid (str): Military grid (35MRV, 36MGR, etc.) that the composite is in
        """
        super().__init__()
        self.region = region
        self.district = district
        self.mgrs = military_grid

    @property
    def archive_path(self) -> str:
        """
        Path to where the file will be saved locally
        """
        return os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.name)

    @property
    def s3_archive_path(self) -> str:
        """
        Path to where the file will be saved in AWS S3
        """
        return self.archive_path.replace(DATA_DIR + os.sep, '')

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Abstract method creating contract with child classes for name property. The name property will define
        how the file is named for a unique set of input parameters
        """
        pass

    @staticmethod
    def filter_files(input_files: List[str], regex: str, region: str = None, district: str = None,
                     mgrs: List[str] = None) -> List[str]:
        matching_files = []
        for f in input_files:
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

    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, mgrs: List[str] = None) -> List[str]:
        """
        Searches for files in the root data directory for the file type. Additional arguments can be added to filter
        the search by matching them to each file's regex parameters. Filters are logically AND-ed together.
        Args:
            regex (str): The regex to match in order to return a file
            region (str): If specified only files of this region will be returned
            district (str): If specified only files of this district will be returned
            mgrs (str): If specified only files of this mgrs tile will be returned
        Returns:
            matching_files (str): Sorted list of absolute paths to each matching file
        """
        files = glob.glob(_BaseCompositeFile._ROOT_DATA_DIR + '/**/*', recursive=True)

        return _BaseCompositeFile.filter_files(files, regex, region, district, mgrs)


class OpticalComposite(_BaseCompositeFile):
    """
    Methods and attributes that define the OpticalComposite file type structure and interaction
    """
    regex = r'optical_composite_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_(?P<bands>[a-zA-Z0-9]{3}(?:_[a-zA-Z0-9]{3})*)\.tif$'

    def __init__(self, region: str, district: str, military_grid: str, bands: List[str]):
        """
        Initializes optical composite file
        Args:
            region (str): Name of the region that the composite is in
            district (str): Name of the district that the composite is in
            military_grid (str): Military grid (35MRV, 36MGR, etc.) that the composite is in
            bands (list): The optical bands included in the file
        """
        super().__init__(region, district, military_grid)
        self.bands = ci_str_sort(bands)

    @property
    def name(self) -> str:
        """
        Defines the unique name of the file for the given instance parameters
        """
        base_string = f'optical_composite_{self.region}_{self.district}_{self.mgrs}_'
        for band in self.bands:
            base_string += band + ('_' if band != self.bands[-1] else '')
        return base_string + '.tif'

    @staticmethod
    def filter_files(input_files: List[str], region: str = None, district: str = None, bands: List[str] = None,
                     mgrs: List[str] = None) -> List[str]:
        input_files = _BaseCompositeFile.filter_files(input_files, OpticalComposite.regex, region, district, mgrs)
        matching_files = []
        for file in input_files:
            match = re.match(OpticalComposite.regex, os.path.basename(file))
            group_dict = match.groupdict()
            if bands is not None and group_dict['bands'].split('_') != ci_str_sort(bands):
                continue
            matching_files.append(file)

        return sorted(matching_files)

    @staticmethod
    def find_files(region: str = None, district: str = None, bands: List[str] = None,
                   mgrs: List[str] = None) -> List[str]:
        """
        Searches for files in the root data directory for the file type. Additional arguments can be added to filter
        the search by matching them to each file's regex parameters. Filters are logically AND-ed together.
        Args:
            region (str): If specified only files of this region will be returned
            district (str): If specified only files of this district will be returned
            bands (list): If specified only files with these bands will be returned
            mgrs (str): If specified only files of this mgrs tile will be returned
        Returns:
            matching_files (str): Sorted list of absolute paths to each matching file
        """
        files = _BaseCompositeFile.find_files(OpticalComposite.regex, region, district, mgrs)

        return OpticalComposite.filter_files(files, region, district, bands, mgrs)

    @classmethod
    def create(cls, file_path: str):
        """
        Creates an instance of the file type class if the input file path matches the regex instance attribute,
        otherwise returns None
        Args:
            file_path (str): The file path which will be attempted to match the class regex
        Returns:
            class (OpticalComposite, None): Either the OpticalComposite instance matching the input file_path or None
        """
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'],
                       bands=group_dict['bands'].split('_'))
        else:
            return None


class OpticalCompositeSlice(_BaseCompositeFile):
    regex = r'optical_composite_slice_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_(?P<band>[^_]+)_' \
            r'(?P<top_bound>\d+)_(?P<bottom_bound>\d+)\.tif$'

    def __init__(self, region: str, district: str, military_grid: str, band: str, top_bound: int, bottom_bound: int):
        super().__init__(region, district, military_grid)
        if top_bound > bottom_bound:
            raise ValueError('top bound must be less than bottom bound')
        self.band = band
        self.top_bound = top_bound
        self.bottom_bound = bottom_bound

    @property
    def name(self) -> str:
        return f'optical_composite_slice_{self.region}_{self.district}_{self.mgrs}_{self.band}_{self.top_bound}_' \
               f'{self.bottom_bound}.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, band: str = None,
                   mgrs: List[str] = None, top_bound: int = None, bottom_bound: int = None) -> List[str]:
        files = _BaseCompositeFile.find_files(OpticalCompositeSlice.regex, region, district, mgrs)

        matching_files = []
        for file in files:
            match = re.match(OpticalCompositeSlice.regex, os.path.basename(file))
            group_dict = match.groupdict()
            if band is not None and group_dict['band'].lower() != band.lower():
                continue
            if top_bound is not None and int(group_dict['top_bound']) != top_bound:
                continue
            if bottom_bound is not None and int(group_dict['bottom_bound']) != bottom_bound:
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
                       band=group_dict['band'], top_bound=int(group_dict['top_bound']),
                       bottom_bound=int(group_dict['bottom_bound']))
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
    def filter_files(input_files: List[str], region: str = None, district: str = None, mgrs: List[str] = None) -> List[
        str]:
        return _BaseCompositeFile.filter_files(input_files, MultiVariateComposite.regex, region, district, mgrs)

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

    @property
    @abstractmethod
    def name(self) -> None:
        return

    @staticmethod
    def _mgrs_from_utm_lat_square(utm_code: str, latitude_band: str, square: str) -> str:
        return f'{utm_code}{latitude_band}{square}'

    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, mgrs: str = None,
                   year: int = None, month: int = None, day: int = None, sequence: int = None,
                   band: str = None) -> List[str]:
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
    regex = r'(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_' \
            r'(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_(?P<band>B\d{2})\.jp2$'

    def __init__(self, region: str, district: str, utm_code: str, latitude_band: str, square: str, year: int,
                 month: int, day: int, band: str, sequence: int = 0):
        super().__init__(region, district, utm_code, latitude_band, square, year, month, day, band, sequence)

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.utm_code}_{self.latitude_band}_{self.square}_{self.year}_' \
               f'{self.month}_{self.day}_{self.sequence}_{self.band}.jp2'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None, year: int = None, month: int = None,
                   day: int = None, sequence: int = None, band: str = None) -> List[str]:
        return _BaseSentinel2File.find_files(Sentinel2Tile.regex, region, district, mgrs, year,
                                             month, day, sequence, band)

    @classmethod
    def create(cls, file_path: str):
        return super(Sentinel2Tile, cls).create(file_path)


class Sentinel2Cloud(_BaseSentinel2File):
    regex = r'(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<utm_code>\d+)_(?P<latitude_band>\S+)_(?P<square>\S+)_' \
            r'(?P<year>\d{4})_(?P<month>\d+)_(?P<day>\d+)_(?P<sequence>\d{1})_qi_MSK_CLOUDS_(?P<band>B\d{2}).gml'

    def __init__(self, region: str, district: str, utm_code: str, latitude_band: str, square: str, year: int,
                 month: int, day: int, sequence: int = 0, band: str = 'B00'):
        super().__init__(region, district, utm_code, latitude_band, square, year, month, day, band, sequence)

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.utm_code}_{self.latitude_band}_{self.square}_{self.year}' \
               f'_{self.month}_{self.day}_{self.sequence}_qi_MSK_CLOUDS_{self.band}.gml'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None, year: int = None, month: int = None,
                   day: int = None, sequence: int = None) -> List[str]:
        return _BaseSentinel2File.find_files(Sentinel2Cloud.regex, region, district, mgrs, year,
                                             month, day, sequence)

    @classmethod
    def create(cls, file_path: str):
        return super(Sentinel2Cloud, cls).create(file_path)


class _BaseNonOpticalBand(File):
    _ROOT_DATA_DIR = None

    def __init__(self, region: str, district: str, mgrs: str):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.name)

    @staticmethod
    def find_files(regex, in_dir: str, region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        files = glob.glob(in_dir + '/**/*', recursive=True)

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
    regex = r'elevation_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]{5})\.tif$'

    @property
    def name(self) -> str:
        return f'elevation_{self.region}_{self.district}_{self.mgrs}.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return _BaseNonOpticalBand.find_files(Elevation.regex, Elevation._ROOT_DATA_DIR, region, district, mgrs)

    @classmethod
    def create(cls, file_path: str):
        return super(Elevation, cls).create(file_path)


class Slope(_BaseNonOpticalBand):
    _ROOT_DATA_DIR = SLOPE_DIR
    regex = r'slope_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]{5})\.tif$'

    @property
    def name(self) -> str:
        return f'slope_{self.region}_{self.district}_{self.mgrs}.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return _BaseNonOpticalBand.find_files(Slope.regex, Slope._ROOT_DATA_DIR, region, district, mgrs)

    @classmethod
    def create(cls, file_path: str):
        return super(Slope, cls).create(file_path)


class OSM(_BaseNonOpticalBand):
    _ROOT_DATA_DIR = OSM_DIR
    regex = r'osm_(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]{5})\.tif$'

    @property
    def name(self) -> str:
        return f'osm_{self.region}_{self.district}_{self.mgrs}.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, mgrs: str = None) -> List[str]:
        return _BaseNonOpticalBand.find_files(OSM.regex, OSM._ROOT_DATA_DIR, region, district, mgrs)

    @classmethod
    def create(cls, file_path: str):
        return super(OSM, cls).create(file_path)


class SingleRegionTileMatch(File):
    _ROOT_DATA_DIR = TILE_DIR
    regex = r'sr_tile_match_(?P<region>[^_]+)_?(?P<district>[^_]+)?_?(?P<mgrs>[^_]+)?_(?P<tile_size>\d+)\.csv$'

    def __init__(self, region: str, tile_size: int, district: str = None, military_grid: str = None):
        self.region = region
        self.district = district
        self.mgrs = military_grid
        self.tile_size = tile_size
        super().__init__()

    @property
    def name(self):
        base_str = f'sr_tile_match_{self.region}_'
        if self.district is not None:
            base_str += f'{self.district}_'
        if self.mgrs is not None:
            base_str += f'{self.mgrs}_'
        base_str += f'{self.tile_size}.csv'
        return base_str

    @property
    def archive_path(self) -> str:
        base_path = os.path.join(self._ROOT_DATA_DIR, self.region)
        if self.district is not None:
            base_path = os.path.join(base_path, self.district)
        if self.mgrs is not None:
            base_path = os.path.join(base_path, self.mgrs)
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
            if district is not None and (group_dict['district'] is None or group_dict['district'].lower()
                                         != district.lower()):
                continue
            if military_grid is not None and (group_dict['mgrs'] is None or group_dict['mgrs'].lower()
                                              != military_grid.lower()):
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
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'],
                       tile_size=int(group_dict['tile_size']))
        else:
            return None


class MultiRegionTileMatch(File):
    _ROOT_DATA_DIR = MULTI_REGION_TILE_MATCH
    regex = r"mr_tile_match_(?P<regions>[\w\s']+(?:_[\w\s']+)*)_(?P<tile_size>\d+)\.csv$"

    def __init__(self, regions: List[str], tile_size: int):
        self.regions = ci_str_sort(regions)
        self.tile_size = tile_size
        super().__init__()

    @property
    def name(self):
        return f"mr_tile_match_{'_'.join(self.regions)}_{self.tile_size}.csv"

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.name)

    @staticmethod
    def find_files(regions: List[str] = None, tile_size: int = None):
        files = glob.glob(MultiRegionTileMatch._ROOT_DATA_DIR + '/**/*', recursive=True)
        matching_files = []
        for file in files:
            match = re.match(MultiRegionTileMatch.regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if regions is not None and group_dict['regions'].split('_') != ci_str_sort(regions):
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
        self.mgrs = military_grid
        self.tile_size = tile_size
        self.x_min = x_min
        self.y_min = y_min
        super().__init__()

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.region, self.district, self.mgrs, str(self.tile_size),
                            self.name)

    @property
    @abstractmethod
    def name(self):
        return

    @staticmethod
    def find_files(regex: str, region: str = None, district: str = None, military_grid: str = None,
                   tile_size: int = None) -> List[str]:
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
            return cls(region=group_dict['region'], district=group_dict['district'], military_grid=group_dict['mgrs'],
                       tile_size=int(group_dict['tile_size']), x_min=int(group_dict['x_min']),
                       y_min=int(group_dict['y_min']))
        else:
            return None


class Tile(_BaseTileFile):
    regex = r'^(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_(?P<tile_size>\d+)_(?P<x_min>\d+)_' \
            r'(?P<y_min>\d+)\.tif$'

    def __init__(self, region: str, district: str, military_grid: str, tile_size: int, x_min: int, y_min: int):
        super().__init__(region, district, military_grid, tile_size, x_min, y_min)

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.mgrs}_{self.tile_size}_{self.x_min}_{self.y_min}.tif'

    @staticmethod
    def find_files(region: str = None, district: str = None, military_grid: str = None,
                   tile_size: int = None) -> List[str]:
        return _BaseTileFile.find_files(Tile.regex, region, district, military_grid, tile_size)

    @classmethod
    def create(cls, file_path: str):
        return super(Tile, cls).create(file_path)


class PyTorch(_BaseTileFile):
    regex = r'^(?P<region>[^_]+)_(?P<district>[^_]+)_(?P<mgrs>[^_]+)_(?P<tile_size>\d+)_(?P<x_min>\d+)_' \
            r'(?P<y_min>\d+)\.pt$'

    def __init__(self, region: str, district: str, military_grid: str, tile_size: int, x_min: int, y_min: int):
        super().__init__(region, district, military_grid, tile_size, x_min, y_min)

    @property
    def name(self) -> str:
        return f'{self.region}_{self.district}_{self.mgrs}_{self.tile_size}_{self.x_min}_{self.y_min}.pt'

    @staticmethod
    def find_files(region: str = None, district: str = None, military_grid: str = None,
                   tile_size: int = None) -> List[str]:
        return _BaseTileFile.find_files(PyTorch.regex, region, district, military_grid, tile_size)

    @classmethod
    def create(cls, file_path: str):
        return super(PyTorch, cls).create(file_path)


class _BaseInferenceFiles(File):
    _ROOT_DATA_DIR = None
    base_layers = 'red|blue|green|nir|osm-water|osm-boundary|elevation|slope'

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float,
                 tile_size: int, best: bool = False):
        if not 0 < len(layers) <= 3:
            raise ValueError('Must only between 1 and 3 layer(s)')
        self.regions = ci_str_sort(regions)
        self.architecture = architecture
        self.sorted_layers = ci_str_sort(layers)
        self.epoch = epoch
        self.ratio = ratio
        self.tile_size = tile_size
        self.best = best
        super().__init__()

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, '_'.join(self.regions), self.architecture, f'r{self.ratio}', self.name)

    def layers(self, regex) -> List[str]:
        match = re.match(regex, self.name)
        group_dict = match.groupdict()
        layers = []
        for i in range(1, 4):
            layer = group_dict[f'layer_{i}']
            if layer is not None:
                layers.append(layer)
        return layers

    @property
    @abstractmethod
    def name(self):
        return

    @property
    def s3_archive_path(self) -> str:
        return self.archive_path.replace(DATA_DIR + os.sep, '')

    @staticmethod
    def filter_files(input_files: List[str], regex: str, regions: List[str] = None, architecture: str = None,
                     layers: List[str] = None, epoch: int = None, ratio: float = None, tile_size: int = None,
                     best: bool = False) -> List[str]:
        matching_files = []
        for file in input_files:
            match = re.match(regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if layers is not None:
                file_layers = []
                for i in range(1, 4):
                    layer = group_dict[f'layer_{i}']
                    if layer is not None:
                        file_layers.append(layer)
                if not all([layer in file_layers for layer in layers]) or \
                   not all([layer in layers for layer in file_layers]):
                    continue
            if regions is not None and group_dict['regions'].lower() != '_'.join(ci_str_sort(regions)).lower():
                continue
            if architecture is not None and group_dict['architecture'] != architecture:
                continue
            if ratio is not None and float(group_dict['ratio']) != ratio:
                continue
            if best and group_dict['best'] is None:
                continue
            if epoch is not None and epoch != int(group_dict['epoch']):
                continue
            if tile_size is not None and tile_size != int(group_dict['tile_size']):
                continue
            matching_files.append(file)

        return sorted(matching_files)

    @staticmethod
    def find_files(in_dir: str, regex: str, regions: List[str] = None, architecture: str = None,
                   layers: List[str] = None, epoch: int = None, ratio: float = None, tile_size: int = None,
                   best: bool = False) -> List[str]:
        if layers is not None and not 0 < len(layers) <= 3:
            raise ValueError('Must only between 1 and 3 layer(s)')
        files = glob.glob(in_dir + '/**/*', recursive=True)

        return _BaseInferenceFiles.filter_files(files, regex, regions, architecture, layers, epoch, ratio, tile_size,
                                                best)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            layers = []
            for i in range(1, 4):
                layer = group_dict[f'layer_{i}']
                if layer is not None:
                    layers.append(layer)
            return cls(regions=group_dict['regions'].split('_'), architecture=group_dict['architecture'],
                       tile_size=int(group_dict['tile_size']), layers=layers,
                       epoch=int(group_dict['epoch']),
                       ratio=float(group_dict['ratio']), best=group_dict['best'] is not None)
        else:
            return None


class TrainedModel(_BaseInferenceFiles):
    _ROOT_DATA_DIR = MODEL_DIR
    ROOT_S3_DIR = os.path.basename(_ROOT_DATA_DIR)
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf"^(?P<regions>[\w\s']+(?:_[\w\s']+)*)_(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_ts(?P<tile_size>\d+)_(?P<layer_1>{base_layers})_(?P<layer_2>{base_layers})?[_]?(?P<layer_3>{base_layers})?[_]?(epoch(?P<epoch>\d+))[_]?(?P<best>best)?\.tar$"

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float,
                 tile_size: int, best: bool = False):
        super().__init__(regions, architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = '_'.join(self.regions) + '_' + self.architecture + f'_r{self.ratio}_' + f'ts{self.tile_size}_'

        for layer in self.sorted_layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + ('_best' if self.best else '') + '.tar'
        return base_str

    @property
    def layers(self):
        return super().layers(self.regex)

    @staticmethod
    def filter_files(input_files: List[str], regions: List[str] = None, architecture: str = None,
                     layers: List[str] = None, epoch: int = None, ratio: float = None, tile_size: int = None,
                     best: bool = False) -> List[str]:
        return _BaseInferenceFiles.filter_files(input_files, TrainedModel.regex, regions, architecture, layers, epoch,
                                                ratio, tile_size, best)

    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: int = None,
                   ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return _BaseInferenceFiles.find_files(TrainedModel._ROOT_DATA_DIR, TrainedModel.regex, regions, architecture,
                                              layers,
                                              epoch, ratio, tile_size, best)

    @classmethod
    def create(cls, file_path: str):
        return super(TrainedModel, cls).create(file_path)


class InferenceResultsCSV(_BaseInferenceFiles):
    _ROOT_DATA_DIR = INFERENCE_RESULTS_DIR
    ROOT_S3_DIR = os.path.basename(_ROOT_DATA_DIR)
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf"^(?P<regions>[\w\s']+(?:_[\w\s']+)*)_(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_ts(?P<tile_size>\d+)_(?P<layer_1>{base_layers})_(?P<layer_2>{base_layers})?[_]?(?P<layer_3>{base_layers})?[_]?(epoch(?P<epoch>\d+))[_]?(?P<best>best)?\.csv$"

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float,
                 tile_size: int, best: bool = False):
        super().__init__(regions, architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = '_'.join(self.regions) + '_' + self.architecture + f'_r{self.ratio}_' + f'ts{self.tile_size}_'

        for layer in self.sorted_layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + ('_best' if self.best else '') + '.csv'
        return base_str

    @property
    def layers(self):
        return super().layers(self.regex)

    @staticmethod
    def filter_files(input_files: List[str], regions: List[str] = None, architecture: str = None,
                     layers: List[str] = None, epoch: int = None, ratio: float = None, tile_size: int = None,
                     best: bool = False) -> List[str]:
        return _BaseInferenceFiles.filter_files(input_files, InferenceResultsCSV.regex, regions, architecture, layers,
                                                epoch, ratio, tile_size, best)

    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None,
                   ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return _BaseInferenceFiles.find_files(InferenceResultsCSV._ROOT_DATA_DIR, InferenceResultsCSV.regex, regions,
                                              architecture,
                                              layers, epoch, ratio, tile_size, best)

    @classmethod
    def create(cls, file_path: str):
        return super(InferenceResultsCSV, cls).create(file_path)


class InferenceResultsShapefile(_BaseInferenceFiles):
    _ROOT_DATA_DIR = INFERENCE_RESULTS_DIR
    ROOT_S3_DIR = os.path.basename(_ROOT_DATA_DIR)
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf"^(?P<regions>[\w\s']+(?:_[\w\s']+)*)_(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_ts(?P<tile_size>\d+)_(?P<layer_1>{base_layers})_(?P<layer_2>{base_layers})?[_]?(?P<layer_3>{base_layers})?[_]?(epoch(?P<epoch>\d+))[_]?(?P<best>best)?\.shp$"

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float,
                 tile_size: int, best: bool = False):
        super().__init__(regions, architecture, layers, epoch, ratio, tile_size, best)
        self.tar_file = InferenceResultsTarfile(regions=self.regions, architecture=self.architecture,
                                                layers=self.layers, epoch=self.epoch, ratio=self.ratio,
                                                tile_size=self.tile_size, best=self.best)

    @property
    def layers(self):
        return super().layers(self.regex)

    @property
    def name(self) -> str:
        base_str = '_'.join(self.regions) + '_' + self.architecture + f'_r{self.ratio}_' + f'ts{self.tile_size}_'

        for layer in self.sorted_layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + ('_best' if self.best else '') + '.shp'
        return base_str

    def create_tar_file(self):
        with tarfile.open(self.tar_file.archive_path, "w") as tar:
            tar.add(os.path.dirname(self.archive_path), arcname="")

    @property
    def archive_path(self) -> str:
        sub_dir = '_'.join(self.layers) + f'_epoch{self.epoch}_' + ('best_' if self.best else '') + 'shapefile'
        path = os.path.join(self._ROOT_DATA_DIR, '_'.join(self.regions), self.architecture, f'r{self.ratio}',
                            sub_dir, self.name)
        return path

    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None,
                   ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return _BaseInferenceFiles.find_files(InferenceResultsShapefile._ROOT_DATA_DIR,
                                              InferenceResultsShapefile.regex, regions, architecture, layers, epoch,
                                              ratio, tile_size, best)

    @classmethod
    def create(cls, file_path: str):
        return super(InferenceResultsShapefile, cls).create(file_path)


class InferenceResultsTarfile(_BaseInferenceFiles):
    _ROOT_DATA_DIR = INFERENCE_RESULTS_DIR
    ROOT_S3_DIR = os.path.basename(_ROOT_DATA_DIR)
    base_layers = _BaseInferenceFiles.base_layers
    regex = rf"^(?P<regions>[\w\s']+(?:_[\w\s']+)*)_(?P<architecture>resnet\d+)_r(?P<ratio>\d+\.\d+)_ts(?P<tile_size>\d+)_(?P<layer_1>{base_layers})_(?P<layer_2>{base_layers})?[_]?(?P<layer_3>{base_layers})?[_]?(epoch(?P<epoch>\d+))[_]?(?P<best>best)?_shapefile\.tar$"

    def __init__(self, regions: List[str], architecture: str, layers: List[str], epoch: int, ratio: float,
                 tile_size: int, best: bool = False):
        super().__init__(regions, architecture, layers, epoch, ratio, tile_size, best)

    @property
    def name(self) -> str:
        base_str = '_'.join(self.regions) + '_' + self.architecture + f'_r{self.ratio}_' + f'ts{self.tile_size}_'

        for layer in self.sorted_layers:
            base_str += layer + '_'

        base_str += f'epoch{self.epoch}' + ('_best' if self.best else '') + '_shapefile.tar'
        return base_str

    @property
    def layers(self):
        return super().layers(self.regex)

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, '_'.join(self.regions), self.architecture, f'r{self.ratio}', self.name)

    @staticmethod
    def filter_files(input_files: List[str], regions: List[str] = None, architecture: str = None,
                     layers: List[str] = None, epoch: int = None, ratio: float = None, tile_size: int = None,
                     best: bool = False) -> List[str]:
        return _BaseInferenceFiles.filter_files(input_files, InferenceResultsCSV.regex, regions, architecture, layers,
                                                epoch, ratio, tile_size, best)

    @staticmethod
    def find_files(regions: List[str] = None, architecture: str = None, layers: List[str] = None, epoch: str = None,
                   ratio: float = None, tile_size: int = None, best: bool = False) -> List[str]:
        return _BaseInferenceFiles.find_files(InferenceResultsTarfile._ROOT_DATA_DIR, InferenceResultsTarfile.regex,
                                              regions,
                                              architecture, layers, epoch, ratio, tile_size, best)

    @classmethod
    def create(cls, file_path: str):
        return super(InferenceResultsTarfile, cls).create(file_path)


class _BaseDatasetSplit(File):
    _ROOT_DATA_DIR = TRAIN_VALIDATE_SPLIT_DIR

    def __init__(self, regions: List[str], ratio: int, tile_size: int):
        self.regions = ci_str_sort(regions)
        self.ratio = ratio
        self.tile_size = tile_size
        super().__init__()

    @property
    def archive_path(self) -> str:
        return os.path.join(self._ROOT_DATA_DIR, self.name)

    @property
    @abstractmethod
    def name(self):
        return

    @staticmethod
    def find_files(regex: str, regions: List[str] = None, ratio: int = None, tile_size: int = None) -> List[str]:
        files = glob.glob(_BaseDatasetSplit._ROOT_DATA_DIR + '/**/*', recursive=True)

        matching_files = []
        for file in files:
            match = re.match(regex, os.path.basename(file))
            if not match:
                continue
            group_dict = match.groupdict()
            if regions is not None:
                regions_str = '_'.join(ci_str_sort(regions))
                if regions_str.lower() != group_dict['regions'].lower():
                    continue
            if ratio is not None and ratio != int(group_dict['ratio']):
                continue
            if tile_size is not None and tile_size != int(group_dict['tile_size']):
                continue
            matching_files.append(file)

        return sorted(matching_files)

    @classmethod
    def create(cls, file_path: str):
        name = os.path.basename(file_path)
        match = re.match(cls.regex, name)
        if match:
            group_dict = match.groupdict()
            return cls(regions=group_dict['regions'].split('_'), ratio=int(group_dict['ratio']),
                       tile_size=int(group_dict['tile_size']))
        else:
            return None


class TrainSplit(_BaseDatasetSplit):
    regex = r"train_(?P<regions>[\w\s']+(?:_[\w\s']+)*)_(?P<ratio>\d+)_ts(?P<tile_size>\d+)\.csv$"

    def __init__(self, regions: List[str], ratio: int, tile_size: int):
        super().__init__(regions, ratio, tile_size)

    @property
    def name(self) -> str:
        return f"train_{'_'.join(self.regions)}_{self.ratio}_ts{self.tile_size}.csv"

    @staticmethod
    def find_files(regions: List[str] = None, ratio: int = None, tile_size: int = None) -> List[str]:
        return _BaseDatasetSplit.find_files(TrainSplit.regex, regions, ratio, tile_size)

    @classmethod
    def create(cls, file_path: str):
        return super(TrainSplit, cls).create(file_path)


class ValidateSplit(_BaseDatasetSplit):
    regex = r"validate_(?P<regions>[\w\s']+(?:_[\w\s']+)*)_(?P<ratio>\d+)_ts(?P<tile_size>\d+)\.csv$"

    def __init__(self, regions: List[str], ratio: int, tile_size: int):
        super().__init__(regions, ratio, tile_size)

    @property
    def name(self) -> str:
        return f"validate_{'_'.join(self.regions)}_{self.ratio}_ts{self.tile_size}.csv"

    @staticmethod
    def find_files(regions: List[str] = None, ratio: int = None, tile_size: int = None) -> List[str]:
        return _BaseDatasetSplit.find_files(ValidateSplit.regex, regions, ratio, tile_size)

    @classmethod
    def create(cls, file_path: str):
        return super(ValidateSplit, cls).create(file_path)
