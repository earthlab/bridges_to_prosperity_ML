import os
import re
from typing import List


class FileType:
    def __init__(self):
        pass

    def identify(self):
        pass


class OpticalComposite(FileType):
    #regex = r'optical_composite_{?<region>\s+}'

    def __init__(self, region: str, district: str, military_grid: str, bands: List[str]):
        self.region = region
        self.district = district
        self.mgrs = military_grid
        self.bands = bands
        super().__init__()

    @property
    def name(self):
        base_string = f'optical_composite_{self.region}_{self.district}_{self.mgrs}_'
        for band in self.bands:
            base_string += band + '_' if band != self.bands[-1] else ''
        return base_string + '.tif'

    @property
    def archive_path(self):
        return os.path.join('data', 'composites', self.region, self.district, self.mgrs, self.name)


class DataCube(FileType):
    #regex = r'optical_composite_{?<region>\s+}'

    def __init__(self, region: str, district: str, military_grid: str):
        self.region = region
        self.district = district
        self.mgrs = military_grid
        super().__init__()

    @property
    def name(self):
        return f'multivariate_{self.region}_{self.district}_{self.mgrs}.tif'

    @property
    def archive_path(self):
        return os.path.join('data', 'composites', self.region, self.district, self.mgrs, self.name)

