import os
import shutil
from unittest import TestCase
from pathlib import Path
from file_types import _BaseCompositeFile, _BaseSentinel2File
from file_types import *

class TestFileTypes(TestCase):
    TEST_DATA_DIR = os.path.join(B2P_DIR, 'tests', 'data', 'file_types')

    @classmethod
    def setUpClass(cls):
        Path(cls.TEST_DATA_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.TEST_DATA_DIR)

    @staticmethod
    def create_blank_file(file_path: str):
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as file:
            pass


class TestOpticalComposite(TestFileTypes):

    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseCompositeFile._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'composites')

    def test_init(self):
        optical_composite = OpticalComposite(region='Uganda', district='Kibaale', military_grid='35MRV',
                                             bands=['B03', 'B02', 'B04'])
        self.assertIsNotNone(optical_composite)
        self.assertEqual(optical_composite.region, 'Uganda')
        self.assertEqual(optical_composite.district, 'Kibaale')
        self.assertEqual(optical_composite.mgrs, '35MRV')
        self.assertEqual(optical_composite.bands, ['B02', 'B03', 'B04'])

    def test_name(self):
        optical_composite = OpticalComposite(region='Uganda', district='Kibaale', military_grid='35MRV',
                                             bands=['B03', 'B02', 'B04'])
        self.assertEqual(optical_composite.name, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif')

        optical_composite = OpticalComposite(region='Rwanda', district='all', military_grid='36MRH',
                                             bands=['B01', 'B02'])

        self.assertEqual(optical_composite.name, 'optical_composite_Rwanda_all_36MRH_B01_B02.tif')

    def test_archive_path(self):
        optical_composite = OpticalComposite(region='Uganda', district='Kibaale', military_grid='35MRV',
                                             bands=['B03', 'B02', 'B04'])

        self.assertEqual(optical_composite.archive_path(),
                         os.path.join(self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kibaale',
                                      'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'))

    def test_create(self):
        # Round trip
        optical_composite = OpticalComposite(region='Uganda', district='Kibaale', military_grid='35MRV',
                                             bands=['B03', 'B02', 'B04'])
        optical_composite_create_name = OpticalComposite.create(optical_composite.name)
        self.assertIsInstance(optical_composite_create_name, OpticalComposite)
        self.assertEqual(optical_composite_create_name.region, 'Uganda')
        self.assertEqual(optical_composite_create_name.district, 'Kibaale')
        self.assertEqual(optical_composite_create_name.mgrs, '35MRV')
        self.assertEqual(optical_composite_create_name.bands, ['B02', 'B03', 'B04'])
        self.assertEqual(optical_composite_create_name.name, optical_composite.name)

        optical_composite_create_path = OpticalComposite.create(optical_composite.archive_path())
        self.assertIsInstance(optical_composite_create_path, OpticalComposite)
        self.assertEqual(optical_composite_create_path.region, 'Uganda')
        self.assertEqual(optical_composite_create_path.district, 'Kibaale')
        self.assertEqual(optical_composite_create_path.mgrs, '35MRV')
        self.assertEqual(optical_composite_create_path.bands, ['B02', 'B03', 'B04'])
        self.assertEqual(optical_composite_create_path.name, optical_composite.name)

        self.assertIsNone(OpticalComposite.create('off_nominal.tif'))

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kibaale')
        os.makedirs(district_1)
        for file in [
            'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif',
            'optical_composite_Uganda_Kibaale_35MRV_B04.tif',
            'optical_composite_Uganda_Kibaale_36MGR_B02_B03_B04.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kasese')
        for file in [
            'optical_composite_Uganda_Kasese_35MRV_B02_B03_B04.tif',
            'optical_composite_Uganda_Kasese_35MRV_B04.tif',
            'optical_composite_Uganda_Kasese_36MGR_B02_B03_B04.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Rwanda', 'all')
        for file in [
            'optical_composite_Rwanda_all_35MRV_B02_B03_B04.tif',
            'optical_composite_Rwanda_all_35MRV_B04.tif',
            'optical_composite_Rwanda_all_36MGR_B02_B03_B04.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        composite_dir = os.path.join(self.TEST_DATA_DIR, 'composites')

        self.assertEqual(OpticalComposite.find_files(), sorted([
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_36MGR_B02_B03_B04.tif'),
            os.path.join(district_2, 'optical_composite_Uganda_Kasese_35MRV_B02_B03_B04.tif'),
            os.path.join(district_2, 'optical_composite_Uganda_Kasese_35MRV_B04.tif'),
            os.path.join(district_2, 'optical_composite_Uganda_Kasese_36MGR_B02_B03_B04.tif'),
            os.path.join(district_3, 'optical_composite_Rwanda_all_35MRV_B02_B03_B04.tif'),
            os.path.join(district_3, 'optical_composite_Rwanda_all_35MRV_B04.tif'),
            os.path.join(district_3, 'optical_composite_Rwanda_all_36MGR_B02_B03_B04.tif')
        ]))
        self.assertEqual(OpticalComposite.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif'),
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_36MGR_B02_B03_B04.tif'),
                             os.path.join(district_2, 'optical_composite_Uganda_Kasese_35MRV_B02_B03_B04.tif'),
                             os.path.join(district_2, 'optical_composite_Uganda_Kasese_35MRV_B04.tif'),
                             os.path.join(district_2, 'optical_composite_Uganda_Kasese_36MGR_B02_B03_B04.tif'),
                         ]))
        self.assertEqual(OpticalComposite.find_files(region='Uganda',
                                                     district='Kibaale'), sorted([
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_36MGR_B02_B03_B04.tif')
        ]))
        self.assertEqual(OpticalComposite.find_files(region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV', '36MGR']), sorted([
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_36MGR_B02_B03_B04.tif')
        ]))
        self.assertEqual(OpticalComposite.find_files(region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV']), sorted([
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif')
        ]))
        self.assertEqual(OpticalComposite.find_files(region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV'], bands=['B03', 'B02', 'B04']),
                         sorted([
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif')
                         ]))
        self.assertEqual(OpticalComposite.find_files(region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV'], bands=['B04']),
                         sorted([
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif')
                         ]))
        self.assertEqual(OpticalComposite.find_files(region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV'], bands=['B03', 'B02']), [])


class TestSentinel2Tile(TestFileTypes):

    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseSentinel2File._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'sentinel2')

    def test_init(self):
        sentinel_2_file = Sentinel2Tile(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1, band='B02')
        self.assertIsNotNone(sentinel_2_file)
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.band, 'B02')
        self.assertEqual(sentinel_2_file.sequence, 0)
    
    def test_name(self):
        sentinel_2_file = Sentinel2Tile(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1, band='B02')
        self.assertEqual(sentinel_2_file.name, '35_MR_V_2020_1_1_0_B02.jp2')

        sentinel_2_file = Sentinel2Tile(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1, band='B02', sequence=1)
        self.assertEqual(sentinel_2_file.name, '35_MR_V_2020_1_1_1_B02.jp2')

    def test_mgrs_grid(self):
        sentinel_2_file = Sentinel2Tile(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1, band='B02')
        self.assertEqual(sentinel_2_file.mgrs_grid, '35MRV')
    
    def test_date_str(self):
        sentinel_2_file = Sentinel2Tile(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1, band='B02')
        self.assertEqual(sentinel_2_file.date_str, '2020_1_1')
    
    def test_archive_path(self):
        sentinel_2_file = Sentinel2Tile(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1, band='B02')
        self.assertEqual(sentinel_2_file.archive_path(region='Uganda', district='Kafefe'), os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kafefe', '35MRV',
                                                                                                         '2020_1_1', '35_MR_V_2020_1_1_0_B02.jp2'))
        
    def test_create(self):
        sentinel_2_file = Sentinel2Tile(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1, band='B02')
        sentinel_2_file = Sentinel2Tile.create(sentinel_2_file.name)
        self.assertIsInstance(sentinel_2_file, Sentinel2Tile)
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.band, 'B02')
        self.assertEqual(sentinel_2_file.sequence, 0)

        sentinel_2_file = Sentinel2Tile.create(sentinel_2_file.archive_path('Uganda', 'Kafefe'))
        self.assertIsInstance(sentinel_2_file, Sentinel2Tile)
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.band, 'B02')
        self.assertEqual(sentinel_2_file.sequence, 0)

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kibaale')
        os.makedirs(district_1)
        for file in [
            '35_MR_V_2020_1_1_0_B01.jp2',
            '36_MR_V_2020_1_1_0_B01.jp2',
            '35_MJ_V_2020_1_1_0_B01.jp2',
            '35_MR_K_2020_1_1_0_B01.jp2',
            '35_MR_V_2021_1_1_0_B01.jp2',
            '35_MR_V_2020_2_1_0_B01.jp2',
            '35_MR_V_2020_1_2_0_B01.jp2',
            '35_MR_V_2020_1_1_2_B01.jp2',
            '35_MR_V_2020_1_1_0_B02.jp2',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))
        district_2 = os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kabarole')
        os.makedirs(district_1, exist_ok=True)
        for file in [
            '35_MR_V_2020_1_1_0_B01.jp2',
            '36_MR_V_2020_1_1_0_B01.jp2',
            '35_MJ_V_2020_1_1_0_B01.jp2',
            '35_MR_K_2020_1_1_0_B01.jp2',
            '35_MR_V_2021_1_1_0_B01.jp2',
            '35_MR_V_2020_2_1_0_B01.jp2',
            '35_MR_V_2020_1_2_0_B01.jp2',
            '35_MR_V_2020_1_1_2_B01.jp2',
            '35_MR_V_2020_1_1_0_B02.jp2',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))
        self.assertListEqual(Sentinel2Tile.find_files(), sorted([
            os.path.join(district_1, '35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, '36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, '35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, '35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2020_1_1_0_B02.jp2'),

            os.path.join(district_2, '35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_1_0_B02.jp2')
        ]))

        self.assertEqual(Sentinel2Tile.find_files(region='Uganda'), sorted([
            os.path.join(district_1, '35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, '36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, '35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, '35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_1, '35_MR_V_2020_1_1_0_B02.jp2'),

            os.path.join(district_2, '35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_1_0_B02.jp2')
        ]))

        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole'), sorted([
            os.path.join(district_2, '35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_2, '35_MR_V_2020_1_1_0_B02.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', utm_code='36'), sorted([
            os.path.join(district_2, '36_MR_V_2020_1_1_0_B01.jp2'),
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', latitude_band='MJ'), sorted([
            os.path.join(district_2, '35_MJ_V_2020_1_1_0_B01.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', square='K'), sorted([
            os.path.join(district_2, '35_MR_K_2020_1_1_0_B01.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', year=2021), sorted([
            os.path.join(district_2, '35_MR_V_2021_1_1_0_B01.jp2'),
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', month=2), sorted([
            os.path.join(district_2, '35_MR_V_2020_2_1_0_B01.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', day=2), sorted([
            os.path.join(district_2, '35_MR_V_2020_1_2_0_B01.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', sequence=2), sorted([
            os.path.join(district_2, '35_MR_V_2020_1_1_2_B01.jp2'),
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', band='B02'), sorted([
            os.path.join(district_2, '35_MR_V_2020_1_1_0_B02.jp2'),
        ]))


class TestSentinel2Cloud(TestFileTypes):

    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseSentinel2File._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'sentinel2')

    def test_init(self):
        sentinel_2_file = Sentinel2Cloud(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1)
        self.assertIsNotNone(sentinel_2_file)
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.sequence, 0)
    
    def test_name(self):
        sentinel_2_file = Sentinel2Cloud(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1)
        self.assertEqual(sentinel_2_file.name, '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml')

        sentinel_2_file = Sentinel2Cloud(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1, sequence=1)
        self.assertEqual(sentinel_2_file.name, '35_MR_V_2020_1_1_1_qi_MSK_CLOUDS_B00.gml')

    def test_mgrs_grid(self):
        sentinel_2_file = Sentinel2Cloud(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1)
        self.assertEqual(sentinel_2_file.mgrs_grid, '35MRV')
    
    def test_date_str(self):
        sentinel_2_file = Sentinel2Cloud(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1)
        self.assertEqual(sentinel_2_file.date_str, '2020_1_1')
    
    def test_archive_path(self):
        sentinel_2_file = Sentinel2Cloud(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1)
        self.assertEqual(sentinel_2_file.archive_path(region='Uganda', district='Kafefe'), os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kafefe', '35MRV',
                                                                                                         '2020_1_1', '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'))
        
    def test_create(self):
        sentinel_2_file = Sentinel2Cloud(utm_code='35', latitude_band='MR', square='V', year=2020, month=1, day=1)
        sentinel_2_file = Sentinel2Cloud.create(sentinel_2_file.name)
        self.assertIsInstance(sentinel_2_file, Sentinel2Cloud)
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.sequence, 0)

        sentinel_2_file = Sentinel2Cloud.create(sentinel_2_file.archive_path('Uganda', 'Kafefe'))
        self.assertIsInstance(sentinel_2_file, Sentinel2Cloud)
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.sequence, 0)

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kibaale')
        os.makedirs(district_1)
        for file in [
            '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))
        district_2 = os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kabarole')
        os.makedirs(district_1, exist_ok=True)
        for file in [
            '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml',
            '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))
        self.assertListEqual(Sentinel2Cloud.find_files(), sorted([
            os.path.join(district_1, '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml'),

            os.path.join(district_2, '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml')
        ]))

        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda'), sorted([
            os.path.join(district_1, '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, '35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml'),

            os.path.join(district_2, '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml')
        ]))

        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole'), sorted([
            os.path.join(district_2, '35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, '35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', utm_code='36'), sorted([
            os.path.join(district_2, '36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', latitude_band='MJ'), sorted([
            os.path.join(district_2, '35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', square='K'), sorted([
            os.path.join(district_2, '35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', year=2021), sorted([
            os.path.join(district_2, '35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', month=2), sorted([
            os.path.join(district_2, '35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', day=2), sorted([
            os.path.join(district_2, '35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', sequence=2), sorted([
            os.path.join(district_2, '35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml'),
        ]))
