import os
import re


class FileType:
    def __init__(self):
        pass

    def identify(self):
        pass


class OpticalComposite(FileType):
    regex = r'optical_composite_{?<region>\s+}'

    def __init__(self, region: str, district: str, mgrs: str):
        self.region = region
        self.district = district
        self.mgrs = mgrs
        super().__init__()

    @property
    def name(self):
        return f'optical_composite_{self.region}_{self.district}_{self.mgrs}.tif'

    @property
    def archive_path(self):
        return os.path.join('data', 'optical', 'composites', self.region, self.district, self.mgrs, self.name)

