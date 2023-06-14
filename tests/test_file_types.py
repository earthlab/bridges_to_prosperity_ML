import os
import shutil
from unittest import TestCase
from file_types import *
from definitions import *
from pathlib import Path


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
    def test_init(self):
        optical_composite = OpticalComposite(region='Uganda', district='Kibaale', military_grid='35MRV',
                                             bands=['B03', 'B02', 'B04'])
        self.assertIsNotNone(optical_composite)
        self.assertEqual(optical_composite.region, 'Uganda')
        self.assertEqual(optical_composite.district, 'Kibaale')
        self.assertEqual(optical_composite.mgrs, '35MRV')
        self.assertEqual(optical_composite.bands, ['B02', 'B03', 'B04'])
        self.assertTrue(optical_composite._s3)

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
                         os.path.join(COMPOSITE_DIR, 'Uganda', 'Kibaale',
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
        self.assertTrue(optical_composite_create_name._s3)
        self.assertEqual(optical_composite_create_name.name, optical_composite.name)

        optical_composite_create_path = OpticalComposite.create(optical_composite.archive_path())
        self.assertIsInstance(optical_composite_create_path, OpticalComposite)
        self.assertEqual(optical_composite_create_path.region, 'Uganda')
        self.assertEqual(optical_composite_create_path.district, 'Kibaale')
        self.assertEqual(optical_composite_create_path.mgrs, '35MRV')
        self.assertEqual(optical_composite_create_path.bands, ['B02', 'B03', 'B04'])
        self.assertTrue(optical_composite_create_path._s3)
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

        self.assertEqual(OpticalComposite.find_files(in_dir=self.TEST_DATA_DIR), [])
        self.assertEqual(OpticalComposite.find_files(in_dir=self.TEST_DATA_DIR, recursive=True), sorted([
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
        self.assertEqual(OpticalComposite.find_files(in_dir=composite_dir, recursive=True, region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif'),
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_36MGR_B02_B03_B04.tif'),
                             os.path.join(district_2, 'optical_composite_Uganda_Kasese_35MRV_B02_B03_B04.tif'),
                             os.path.join(district_2, 'optical_composite_Uganda_Kasese_35MRV_B04.tif'),
                             os.path.join(district_2, 'optical_composite_Uganda_Kasese_36MGR_B02_B03_B04.tif'),
                         ]))
        self.assertEqual(OpticalComposite.find_files(in_dir=composite_dir, recursive=True, region='Uganda',
                                                     district='Kibaale'), sorted([
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_36MGR_B02_B03_B04.tif')
        ]))
        self.assertEqual(OpticalComposite.find_files(in_dir=composite_dir, recursive=True, region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV', '36MGR']), sorted([
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_36MGR_B02_B03_B04.tif')
        ]))
        self.assertEqual(OpticalComposite.find_files(in_dir=composite_dir, recursive=True, region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV']), sorted([
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif'),
            os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif')
        ]))
        self.assertEqual(OpticalComposite.find_files(in_dir=composite_dir, recursive=True, region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV'], bands=['B03', 'B02', 'B04']),
                         sorted([
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B02_B03_B04.tif')
                         ]))
        self.assertEqual(OpticalComposite.find_files(in_dir=composite_dir, recursive=True, region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV'], bands=['B04']),
                         sorted([
                             os.path.join(district_1, 'optical_composite_Uganda_Kibaale_35MRV_B04.tif')
                         ]))
        self.assertEqual(OpticalComposite.find_files(in_dir=composite_dir, recursive=True, region='Uganda',
                                                     district='Kibaale', mgrs=['35MRV'], bands=['B03', 'B02']), [])
