import shutil
from unittest import TestCase
from file_types import _BaseCompositeFile, _BaseSentinel2File, _BaseNonOpticalBand, _BaseInferenceFiles, _BaseTileFile, \
    _BaseDatasetSplit

from file_types import *
from definitions import B2P_DIR


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

        self.assertEqual(optical_composite.archive_path,
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

        optical_composite_create_path = OpticalComposite.create(optical_composite.archive_path)
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


class TestOpticalCompositeSlice(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseCompositeFile._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'composites')

    def test_init(self):
        optical_composite_slice = OpticalCompositeSlice(region='Uganda', district='Kibaale', military_grid='35MRV',
                                                        band='B03', left_bound=0, right_bound=500)
        self.assertIsNotNone(optical_composite_slice)

        self.assertEqual(optical_composite_slice.region, 'Uganda')
        self.assertEqual(optical_composite_slice.district, 'Kibaale')
        self.assertEqual(optical_composite_slice.mgrs, '35MRV')
        self.assertEqual(optical_composite_slice.band, 'B03')
        self.assertEqual(optical_composite_slice.left_bound, 0)
        self.assertEqual(optical_composite_slice.right_bound, 500)

    def test_name(self):
        optical_composite_slice = OpticalCompositeSlice(region='Uganda', district='Kibaale', military_grid='35MRV',
                                                        band='B03', left_bound=0, right_bound=500)
        self.assertEqual(optical_composite_slice.name, 'optical_composite_slice_Uganda_Kibaale_35MRV_B03_0_500.tif')

        optical_composite_slice = OpticalCompositeSlice(region='Rwanda', district='all', military_grid='36MRK',
                                                        band='B01', left_bound=1000, right_bound=1500)
        self.assertEqual(optical_composite_slice.name, 'optical_composite_slice_Rwanda_all_36MRK_B01_1000_1500.tif')

    def test_archive_path(self):
        optical_composite_slice = OpticalCompositeSlice(region='Uganda', district='Kibaale', military_grid='35MRV',
                                                        band='B03', left_bound=0, right_bound=500)
        self.assertEqual(optical_composite_slice.archive_path, os.path.join(
            self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kibaale', '35MRV',
            'optical_composite_slice_Uganda_Kibaale_35MRV_B03_0_500.tif'))

    def test_create(self):
        # Round trip
        optical_composite_slice = OpticalCompositeSlice(region='Uganda', district='Kibaale', military_grid='35MRV',
                                                        band='B03', left_bound=0, right_bound=500)
        optical_composite_create_name = OpticalCompositeSlice.create(optical_composite_slice.name)
        self.assertIsInstance(optical_composite_create_name, OpticalCompositeSlice)
        self.assertEqual(optical_composite_create_name.region, 'Uganda')
        self.assertEqual(optical_composite_create_name.district, 'Kibaale')
        self.assertEqual(optical_composite_create_name.mgrs, '35MRV')
        self.assertEqual(optical_composite_create_name.band, 'B03')
        self.assertEqual(optical_composite_create_name.left_bound, 0)
        self.assertEqual(optical_composite_create_name.right_bound, 500)
        self.assertEqual(optical_composite_create_name.name, optical_composite_slice.name)

        optical_composite_create_path = OpticalCompositeSlice.create(optical_composite_slice.archive_path)
        self.assertIsInstance(optical_composite_create_path, OpticalCompositeSlice)
        self.assertEqual(optical_composite_create_path.region, 'Uganda')
        self.assertEqual(optical_composite_create_path.district, 'Kibaale')
        self.assertEqual(optical_composite_create_path.mgrs, '35MRV')
        self.assertEqual(optical_composite_create_path.band, 'B03')
        self.assertEqual(optical_composite_create_path.left_bound, 0)
        self.assertEqual(optical_composite_create_path.right_bound, 500)
        self.assertEqual(optical_composite_create_path.name, optical_composite_slice.name)

        self.assertIsNone(OpticalComposite.create('off_nominal.tif'))

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kibaale')
        os.makedirs(district_1)
        for file in [
            'optical_composite_slice_Uganda_Kibaale_35MRV_B02_0_500.tif',
            'optical_composite_slice_Uganda_Kibaale_35MRV_B01_0_500.tif',
            'optical_composite_slice_Uganda_Kibaale_35MRK_B02_500_1000.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kasese')
        for file in [
            'optical_composite_slice_Uganda_Kasese_35MRV_B02_0_500.tif',
            'optical_composite_slice_Uganda_Kasese_35MRV_B04_0_500.tif',
            'optical_composite_slice_Uganda_Kasese_36MGR_B02_500_1000.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Rwanda', 'all')
        for file in [
            'optical_composite_slice_Rwanda_all_35MRV_B02_0_500.tif',
            'optical_composite_slice_Rwanda_all_35MRV_B04_500_1000.tif',
            'optical_composite_slice_Rwanda_all_36MGR_B02_1000_1500.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(OpticalCompositeSlice.find_files(), sorted([
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B02_0_500.tif'),
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B01_0_500.tif'),
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRK_B02_500_1000.tif'),
            os.path.join(district_2, 'optical_composite_slice_Uganda_Kasese_35MRV_B02_0_500.tif'),
            os.path.join(district_2, 'optical_composite_slice_Uganda_Kasese_35MRV_B04_0_500.tif'),
            os.path.join(district_2, 'optical_composite_slice_Uganda_Kasese_36MGR_B02_500_1000.tif'),
            os.path.join(district_3, 'optical_composite_slice_Rwanda_all_35MRV_B02_0_500.tif'),
            os.path.join(district_3, 'optical_composite_slice_Rwanda_all_35MRV_B04_500_1000.tif'),
            os.path.join(district_3, 'optical_composite_slice_Rwanda_all_36MGR_B02_1000_1500.tif')
        ]))
        self.assertEqual(OpticalCompositeSlice.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B02_0_500.tif'),
                             os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B01_0_500.tif'),
                             os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRK_B02_500_1000.tif'),
                             os.path.join(district_2, 'optical_composite_slice_Uganda_Kasese_35MRV_B02_0_500.tif'),
                             os.path.join(district_2, 'optical_composite_slice_Uganda_Kasese_35MRV_B04_0_500.tif'),
                             os.path.join(district_2, 'optical_composite_slice_Uganda_Kasese_36MGR_B02_500_1000.tif'),
                         ]))
        self.assertEqual(OpticalCompositeSlice.find_files(region='Uganda',
                                                          district='Kibaale'), sorted([
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B02_0_500.tif'),
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B01_0_500.tif'),
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRK_B02_500_1000.tif'),
        ]))
        self.assertEqual(OpticalCompositeSlice.find_files(region='Uganda',
                                                          district='Kibaale', mgrs=['35MRV', '35MRK']), sorted([
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B02_0_500.tif'),
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B01_0_500.tif'),
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRK_B02_500_1000.tif'),
        ]))
        self.assertEqual(OpticalCompositeSlice.find_files(region='Uganda',
                                                          district='Kibaale', mgrs=['35MRV']), sorted([
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B02_0_500.tif'),
            os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B01_0_500.tif'),
        ]))
        self.assertEqual(OpticalCompositeSlice.find_files(region='Uganda',
                                                          district='Kibaale', mgrs=['35MRV'], band='B02'),
                         sorted([
                             os.path.join(district_1, 'optical_composite_slice_Uganda_Kibaale_35MRV_B02_0_500.tif'),
                         ]))
        self.assertEqual(OpticalCompositeSlice.find_files(region='Uganda',
                                                          district='Kibaale', mgrs=['35MRV'], band='B04'),
                         sorted([]))


class TestMultiVariateComposite(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseCompositeFile._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'composites')

    def test_init(self):
        multivariate_composite = MultiVariateComposite(region='Uganda', district='Kibaale', military_grid='35MRV')
        self.assertIsNotNone(multivariate_composite)

        self.assertEqual(multivariate_composite.region, 'Uganda')
        self.assertEqual(multivariate_composite.district, 'Kibaale')
        self.assertEqual(multivariate_composite.mgrs, '35MRV')

    def test_name(self):
        multivariate_composite = MultiVariateComposite(region='Uganda', district='Kibaale', military_grid='35MRV')
        self.assertEqual(multivariate_composite.name, 'multivariate_composite_Uganda_Kibaale_35MRV.tif')

        multivariate_composite = MultiVariateComposite(region='Rwanda', district='all', military_grid='36MRK')
        self.assertEqual(multivariate_composite.name, 'multivariate_composite_Rwanda_all_36MRK.tif')

    def test_archive_path(self):
        multivariate_composite = MultiVariateComposite(region='Uganda', district='Kibaale', military_grid='35MRV')
        self.assertEqual(multivariate_composite.archive_path,
                         os.path.join(self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kibaale',
                                      'multivariate_composite_Uganda_Kibaale_35MRV.tif')
                         )

        multivariate_composite = MultiVariateComposite(region='Rwanda', district='all', military_grid='36MRK')
        self.assertEqual(multivariate_composite.archive_path,
                         os.path.join(self.TEST_DATA_DIR, 'composites', 'Rwanda', 'all',
                                      'multivariate_composite_Rwanda_all_36MRK.tif')
                         )

    def test_create(self):
        multivariate_file = MultiVariateComposite(region='Uganda', district='Kibaale', military_grid='35MRV')
        multivariate_create_name = MultiVariateComposite.create(multivariate_file.name)
        self.assertIsInstance(multivariate_create_name, MultiVariateComposite)
        self.assertEqual(multivariate_create_name.region, 'Uganda')
        self.assertEqual(multivariate_create_name.district, 'Kibaale')
        self.assertEqual(multivariate_create_name.mgrs, '35MRV')

        multivariate_create_path = MultiVariateComposite.create(multivariate_create_name.archive_path)
        self.assertIsInstance(multivariate_create_path, MultiVariateComposite)
        self.assertEqual(multivariate_create_path.region, 'Uganda')
        self.assertEqual(multivariate_create_path.district, 'Kibaale')
        self.assertEqual(multivariate_create_path.mgrs, '35MRV')

        self.assertIsNone(OpticalComposite.create('off_nominal.tif'))

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kibaale')
        os.makedirs(district_1)
        for file in [
            'multivariate_composite_Uganda_Kibaale_35MRV.tif',
            'multivariate_composite_Uganda_Kibaale_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Uganda', 'Kasese')
        for file in [
            'multivariate_composite_Uganda_Kasese_35MRV.tif',
            'multivariate_composite_Uganda_Kasese_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'composites', 'Rwanda', 'all')
        for file in [
            'multivariate_composite_Rwanda_all_35MRV.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(MultiVariateComposite.find_files(), sorted([
            os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_36MGR.tif'),
            os.path.join(district_2, 'multivariate_composite_Uganda_Kasese_35MRV.tif'),
            os.path.join(district_2, 'multivariate_composite_Uganda_Kasese_36MGR.tif'),
            os.path.join(district_3, 'multivariate_composite_Rwanda_all_35MRV.tif'),
        ]))
        self.assertEqual(MultiVariateComposite.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_35MRV.tif'),
                             os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_36MGR.tif'),
                             os.path.join(district_2, 'multivariate_composite_Uganda_Kasese_35MRV.tif'),
                             os.path.join(district_2, 'multivariate_composite_Uganda_Kasese_36MGR.tif')
                         ]))
        self.assertEqual(MultiVariateComposite.find_files(region='Uganda',
                                                          district='Kibaale'), sorted([
            os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_36MGR.tif')
        ]))
        self.assertEqual(MultiVariateComposite.find_files(region='Uganda',
                                                          district='Kibaale', mgrs=['35MRV', '36MGR']), sorted([
            os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_36MGR.tif')
        ]))
        self.assertEqual(MultiVariateComposite.find_files(region='Uganda',
                                                          district='Kibaale', mgrs=['35MRV']), sorted([
            os.path.join(district_1, 'multivariate_composite_Uganda_Kibaale_35MRV.tif')
        ]))
        self.assertEqual(MultiVariateComposite.find_files(region='Rwanda',
                                                          district='all', mgrs=['35MRV']),
                         sorted([
                             os.path.join(district_3, 'multivariate_composite_Rwanda_all_35MRV.tif'),
                         ]))
        self.assertEqual(MultiVariateComposite.find_files(region='Rwanda', district='all', mgrs=['37MRV']), [])


class TestSentinel2Tile(TestFileTypes):

    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseSentinel2File._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'sentinel2')

    def test_init(self):
        sentinel_2_file = Sentinel2Tile(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                        square='V', year=2020, month=1, day=1, band='B02')
        self.assertIsNotNone(sentinel_2_file)
        self.assertEqual(sentinel_2_file.region, 'Uganda')
        self.assertEqual(sentinel_2_file.district, 'Kafefe')
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.band, 'B02')
        self.assertEqual(sentinel_2_file.sequence, 0)

    def test_name(self):
        sentinel_2_file = Sentinel2Tile(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                        square='V', year=2020, month=1, day=1, band='B02')
        self.assertEqual(sentinel_2_file.name, 'Uganda_Kafefe_35_MR_V_2020_1_1_0_B02.jp2')

        sentinel_2_file = Sentinel2Tile(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                        square='V', year=2020, month=1, day=1, band='B02', sequence=1)
        self.assertEqual(sentinel_2_file.name, 'Uganda_Kafefe_35_MR_V_2020_1_1_1_B02.jp2')

    def test_mgrs_grid(self):
        sentinel_2_file = Sentinel2Tile(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                        square='V', year=2020, month=1, day=1, band='B02')
        self.assertEqual(sentinel_2_file.mgrs_grid, '35MRV')

    def test_date_str(self):
        sentinel_2_file = Sentinel2Tile(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                        square='V', year=2020, month=1, day=1, band='B02')
        self.assertEqual(sentinel_2_file.date_str, '2020_1_1')

    def test_archive_path(self):
        sentinel_2_file = Sentinel2Tile(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                        square='V', year=2020, month=1, day=1, band='B02')
        self.assertEqual(sentinel_2_file.archive_path, os.path.join(
            self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kafefe', '35MRV', '2020_1_1',
            'Uganda_Kafefe_35_MR_V_2020_1_1_0_B02.jp2'))

    def test_create(self):
        sentinel_2_file = Sentinel2Tile(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                        square='V', year=2020, month=1, day=1, band='B02')
        sentinel_2_file = Sentinel2Tile.create(sentinel_2_file.name)
        self.assertIsInstance(sentinel_2_file, Sentinel2Tile)
        self.assertEqual(sentinel_2_file.region, 'Uganda')
        self.assertEqual(sentinel_2_file.district, 'Kafefe')
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.band, 'B02')
        self.assertEqual(sentinel_2_file.sequence, 0)

        sentinel_2_file = Sentinel2Tile.create(sentinel_2_file.archive_path)
        self.assertIsInstance(sentinel_2_file, Sentinel2Tile)
        self.assertEqual(sentinel_2_file.region, 'Uganda')
        self.assertEqual(sentinel_2_file.district, 'Kafefe')
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
            'Uganda_Kibaale_35_MR_V_2020_1_1_0_B01.jp2',
            'Uganda_Kibaale_36_MR_V_2020_1_1_0_B01.jp2',
            'Uganda_Kibaale_35_MJ_V_2020_1_1_0_B01.jp2',
            'Uganda_Kibaale_35_MR_K_2020_1_1_0_B01.jp2',
            'Uganda_Kibaale_35_MR_V_2021_1_1_0_B01.jp2',
            'Uganda_Kibaale_35_MR_V_2020_2_1_0_B01.jp2',
            'Uganda_Kibaale_35_MR_V_2020_1_2_0_B01.jp2',
            'Uganda_Kibaale_35_MR_V_2020_1_1_2_B01.jp2',
            'Uganda_Kibaale_35_MR_V_2020_1_1_0_B02.jp2',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))
        district_2 = os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kabarole')
        os.makedirs(district_1, exist_ok=True)
        for file in [
            'Uganda_Kabarole_35_MR_V_2020_1_1_0_B01.jp2',
            'Uganda_Kabarole_36_MR_V_2020_1_1_0_B01.jp2',
            'Uganda_Kabarole_35_MJ_V_2020_1_1_0_B01.jp2',
            'Uganda_Kabarole_35_MR_K_2020_1_1_0_B01.jp2',
            'Uganda_Kabarole_35_MR_V_2021_1_1_0_B01.jp2',
            'Uganda_Kabarole_35_MR_V_2020_2_1_0_B01.jp2',
            'Uganda_Kabarole_35_MR_V_2020_1_2_0_B01.jp2',
            'Uganda_Kabarole_35_MR_V_2020_1_1_2_B01.jp2',
            'Uganda_Kabarole_35_MR_V_2020_1_1_0_B02.jp2',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))
        self.assertListEqual(Sentinel2Tile.find_files(), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_1_1_0_B02.jp2'),

            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_B02.jp2')
        ]))

        self.assertEqual(Sentinel2Tile.find_files(region='Uganda'), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_1, 'Uganda_Kibaale_35_MR_V_2020_1_1_0_B02.jp2'),

            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_B02.jp2')
        ]))

        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_36_MR_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MJ_V_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_K_2020_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2021_1_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_2_1_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_2_0_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_2_B01.jp2'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_B02.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', mgrs='36MRV'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_36_MR_V_2020_1_1_0_B01.jp2'),
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', mgrs='35MJV'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MJ_V_2020_1_1_0_B01.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', mgrs='35mrk'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_K_2020_1_1_0_B01.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', year=2021), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2021_1_1_0_B01.jp2'),
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', month=2), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_2_1_0_B01.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', day=2), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_2_0_B01.jp2')
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', sequence=2), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_2_B01.jp2'),
        ]))
        self.assertEqual(Sentinel2Tile.find_files(region='Uganda', district='Kabarole', band='B02'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_B02.jp2'),
        ]))


class TestSentinel2Cloud(TestFileTypes):

    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseSentinel2File._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'sentinel2')

    def test_init(self):
        sentinel_2_file = Sentinel2Cloud(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                         square='V', year=2020, month=1, day=1)
        self.assertIsNotNone(sentinel_2_file)
        self.assertEqual(sentinel_2_file.region, 'Uganda')
        self.assertEqual(sentinel_2_file.district, 'Kafefe')
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.sequence, 0)

    def test_name(self):
        sentinel_2_file = Sentinel2Cloud(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                         square='V', year=2020, month=1, day=1)
        self.assertEqual(sentinel_2_file.name, 'Uganda_Kafefe_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml')

        sentinel_2_file = Sentinel2Cloud(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                         square='V', year=2020, month=1, day=1, sequence=1)
        self.assertEqual(sentinel_2_file.name, 'Uganda_Kafefe_35_MR_V_2020_1_1_1_qi_MSK_CLOUDS_B00.gml')

    def test_mgrs_grid(self):
        sentinel_2_file = Sentinel2Cloud(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                         square='V', year=2020, month=1, day=1)
        self.assertEqual(sentinel_2_file.mgrs_grid, '35MRV')

    def test_date_str(self):
        sentinel_2_file = Sentinel2Cloud(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                         square='V', year=2020, month=1, day=1)
        self.assertEqual(sentinel_2_file.date_str, '2020_1_1')

    def test_archive_path(self):
        sentinel_2_file = Sentinel2Cloud(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                         square='V', year=2020, month=1, day=1)
        self.assertEqual(sentinel_2_file.archive_path, os.path.join(
            self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kafefe', '35MRV', '2020_1_1',
            'Uganda_Kafefe_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'))

    def test_create(self):
        sentinel_2_file = Sentinel2Cloud(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                         square='V', year=2020, month=1, day=1)
        sentinel_2_file = Sentinel2Cloud.create(sentinel_2_file.name)
        self.assertIsInstance(sentinel_2_file, Sentinel2Cloud)
        self.assertEqual(sentinel_2_file.region, 'Uganda')
        self.assertEqual(sentinel_2_file.district, 'Kafefe')
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.sequence, 0)

        sentinel_2_file = Sentinel2Cloud.create(sentinel_2_file.archive_path)
        self.assertIsInstance(sentinel_2_file, Sentinel2Cloud)
        self.assertEqual(sentinel_2_file.region, 'Uganda')
        self.assertEqual(sentinel_2_file.district, 'Kafefe')
        self.assertEqual(sentinel_2_file.utm_code, '35')
        self.assertEqual(sentinel_2_file.latitude_band, 'MR')
        self.assertEqual(sentinel_2_file.square, 'V')
        self.assertEqual(sentinel_2_file.year, 2020)
        self.assertEqual(sentinel_2_file.month, 1)
        self.assertEqual(sentinel_2_file.day, 1)
        self.assertEqual(sentinel_2_file.sequence, 0)

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kafefe')
        os.makedirs(district_1)
        for file in [
            'Uganda_Kafefe_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kafefe_36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kafefe_35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kafefe_35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kafefe_35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kafefe_35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kafefe_35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kafefe_35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kafefe_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))
        district_2 = os.path.join(self.TEST_DATA_DIR, 'sentinel2', 'Uganda', 'Kabarole')
        os.makedirs(district_1, exist_ok=True)
        for file in [
            'Uganda_Kabarole_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kabarole_36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kabarole_35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kabarole_35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kabarole_35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kabarole_35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kabarole_35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kabarole_35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml',
            'Uganda_Kabarole_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))
        self.assertListEqual(Sentinel2Cloud.find_files(), sorted([
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml'),

            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml')
        ]))

        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda'), sorted([
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_1, 'Uganda_Kafefe_35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml'),

            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml')
        ]))

        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml'),
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', mgrs='36MRV'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_36_MR_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml'),
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', mgrs='35MJV'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MJ_V_2020_1_1_0_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', mgrs='35MRK'), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_K_2020_1_1_0_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', year=2021), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2021_1_1_0_qi_MSK_CLOUDS_B00.gml'),
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', month=2), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_2_1_0_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', day=2), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_2_0_qi_MSK_CLOUDS_B00.gml')
        ]))
        self.assertEqual(Sentinel2Cloud.find_files(region='Uganda', district='Kabarole', sequence=2), sorted([
            os.path.join(district_2, 'Uganda_Kabarole_35_MR_V_2020_1_1_2_qi_MSK_CLOUDS_B00.gml'),
        ]))


class TestElevationFile(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        Elevation._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'elevation')

    def test_init(self):
        elevation_file = Elevation(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertIsNotNone(elevation_file)
        self.assertEqual(elevation_file.region, 'Uganda')
        self.assertEqual(elevation_file.district, 'Kasese')
        self.assertEqual(elevation_file.mgrs, '35MRV')

        elevation_file = Elevation(region='Rwanda', district='all', mgrs='36MRV')
        self.assertIsNotNone(elevation_file)
        self.assertEqual(elevation_file.region, 'Rwanda')
        self.assertEqual(elevation_file.district, 'all')
        self.assertEqual(elevation_file.mgrs, '36MRV')

    def test_name(self):
        elevation_file = Elevation(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(elevation_file.name, 'Uganda_Kasese_35MRV.tif')

        elevation_file = Elevation(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(elevation_file.name, 'Rwanda_all_36MRV.tif')

    def test_archive_path(self):
        elevation_file = Elevation(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(elevation_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'elevation', 'Uganda', 'Kasese',
                                                                   'Uganda_Kasese_35MRV.tif'))

        elevation_file = Elevation(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(elevation_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'elevation', 'Rwanda', 'all',
                                                                   'Rwanda_all_36MRV.tif'))

    def test_create(self):
        elevation_file = Elevation(region='Uganda', district='Kasese', mgrs='35MRV')
        elevation_file_create = Elevation.create(elevation_file.name)
        self.assertIsInstance(elevation_file_create, Elevation)
        self.assertEqual(elevation_file_create.region, 'Uganda')
        self.assertEqual(elevation_file_create.district, 'Kasese')
        self.assertEqual(elevation_file_create.mgrs, '35MRV')

        elevation_file_create = Elevation.create(elevation_file.archive_path)
        self.assertIsInstance(elevation_file_create, Elevation)
        self.assertEqual(elevation_file_create.region, 'Uganda')
        self.assertEqual(elevation_file_create.district, 'Kasese')
        self.assertEqual(elevation_file_create.mgrs, '35MRV')

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'elevation', 'Uganda', 'Kibaale')
        os.makedirs(district_1)
        for file in [
            'Uganda_Kibaale_35MRV.tif',
            'Uganda_Kibaale_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'elevation', 'Uganda', 'Kasese')
        for file in [
            'Uganda_Kasese_35MRV.tif',
            'Uganda_Kasese_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'elevation', 'Rwanda', 'all')
        for file in [
            'Rwanda_all_35MRV.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(Elevation.find_files(), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif'),
            os.path.join(district_2, 'Uganda_Kasese_35MRV.tif'),
            os.path.join(district_2, 'Uganda_Kasese_36MGR.tif'),
            os.path.join(district_3, 'Rwanda_all_35MRV.tif'),
        ]))
        self.assertEqual(Elevation.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
                             os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif'),
                             os.path.join(district_2, 'Uganda_Kasese_35MRV.tif'),
                             os.path.join(district_2, 'Uganda_Kasese_36MGR.tif')
                         ]))
        self.assertEqual(Elevation.find_files(region='Uganda',
                                              district='Kibaale'), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif')
        ]))
        self.assertEqual(Elevation.find_files(region='Uganda',
                                              district='Kibaale', mgrs='35MRV'), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif')
        ]))
        self.assertEqual(Elevation.find_files(region='Rwanda',
                                              district='all', mgrs='35MRV'),
                         sorted([
                             os.path.join(district_3, 'Rwanda_all_35MRV.tif'),
                         ]))
        self.assertEqual(Elevation.find_files(region='Rwanda', district='all', mgrs='37MRV'), [])


class TestSlopeFile(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        Slope._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'slope')

    def test_init(self):
        slope_file = Slope(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertIsNotNone(slope_file)
        self.assertEqual(slope_file.region, 'Uganda')
        self.assertEqual(slope_file.district, 'Kasese')
        self.assertEqual(slope_file.mgrs, '35MRV')

        slope_file = Slope(region='Rwanda', district='all', mgrs='36MRV')
        self.assertIsNotNone(slope_file)
        self.assertEqual(slope_file.region, 'Rwanda')
        self.assertEqual(slope_file.district, 'all')
        self.assertEqual(slope_file.mgrs, '36MRV')

    def test_name(self):
        slope_file = Slope(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(slope_file.name, 'Uganda_Kasese_35MRV.tif')

        slope_file = Slope(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(slope_file.name, 'Rwanda_all_36MRV.tif')

    def test_archive_path(self):
        slope_file = Slope(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(slope_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'slope', 'Uganda', 'Kasese',
                                                               'Uganda_Kasese_35MRV.tif'))

        slope_file = Slope(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(slope_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'slope', 'Rwanda', 'all',
                                                               'Rwanda_all_36MRV.tif'))

    def test_create(self):
        slope_file = Slope(region='Uganda', district='Kasese', mgrs='35MRV')
        slope_file_create = Slope.create(slope_file.name)
        self.assertIsInstance(slope_file_create, Slope)
        self.assertEqual(slope_file_create.region, 'Uganda')
        self.assertEqual(slope_file_create.district, 'Kasese')
        self.assertEqual(slope_file_create.mgrs, '35MRV')

        slope_file_create = Slope.create(slope_file.archive_path)
        self.assertIsInstance(slope_file_create, Slope)
        self.assertEqual(slope_file_create.region, 'Uganda')
        self.assertEqual(slope_file_create.district, 'Kasese')
        self.assertEqual(slope_file_create.mgrs, '35MRV')

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'slope', 'Uganda', 'Kibaale')
        os.makedirs(district_1)
        for file in [
            'Uganda_Kibaale_35MRV.tif',
            'Uganda_Kibaale_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'slope', 'Uganda', 'Kasese')
        for file in [
            'Uganda_Kasese_35MRV.tif',
            'Uganda_Kasese_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'slope', 'Rwanda', 'all')
        for file in [
            'Rwanda_all_35MRV.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(Slope.find_files(), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif'),
            os.path.join(district_2, 'Uganda_Kasese_35MRV.tif'),
            os.path.join(district_2, 'Uganda_Kasese_36MGR.tif'),
            os.path.join(district_3, 'Rwanda_all_35MRV.tif'),
        ]))
        self.assertEqual(Slope.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
                             os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif'),
                             os.path.join(district_2, 'Uganda_Kasese_35MRV.tif'),
                             os.path.join(district_2, 'Uganda_Kasese_36MGR.tif')
                         ]))
        self.assertEqual(Slope.find_files(region='Uganda',
                                          district='Kibaale'), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif')
        ]))
        self.assertEqual(Slope.find_files(region='Uganda',
                                          district='Kibaale', mgrs='35MRV'), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif')
        ]))
        self.assertEqual(Slope.find_files(region='Rwanda',
                                          district='all', mgrs='35MRV'),
                         sorted([
                             os.path.join(district_3, 'Rwanda_all_35MRV.tif'),
                         ]))
        self.assertEqual(Slope.find_files(region='Rwanda', district='all', mgrs='37MRV'), [])


class TestOSMFile(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        OSM._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'osm')

    def test_init(self):
        osm_file = OSM(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertIsNotNone(osm_file)
        self.assertEqual(osm_file.region, 'Uganda')
        self.assertEqual(osm_file.district, 'Kasese')
        self.assertEqual(osm_file.mgrs, '35MRV')

        osm_file = OSM(region='Rwanda', district='all', mgrs='36MRV')
        self.assertIsNotNone(osm_file)
        self.assertEqual(osm_file.region, 'Rwanda')
        self.assertEqual(osm_file.district, 'all')
        self.assertEqual(osm_file.mgrs, '36MRV')

    def test_name(self):
        osm_file = OSM(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(osm_file.name, 'Uganda_Kasese_35MRV.tif')

        osm_file = OSM(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(osm_file.name, 'Rwanda_all_36MRV.tif')

    def test_archive_path(self):
        osm_file = OSM(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(osm_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'osm', 'Uganda', 'Kasese',
                                                             'Uganda_Kasese_35MRV.tif'))

        osm_file = OSM(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(osm_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'osm', 'Rwanda', 'all',
                                                             'Rwanda_all_36MRV.tif'))

    def test_create(self):
        osm_file = OSM(region='Uganda', district='Kasese', mgrs='35MRV')
        osm_file_create = OSM.create(osm_file.name)
        self.assertIsInstance(osm_file_create, OSM)
        self.assertEqual(osm_file_create.region, 'Uganda')
        self.assertEqual(osm_file_create.district, 'Kasese')
        self.assertEqual(osm_file_create.mgrs, '35MRV')

        osm_file_create = OSM.create(osm_file.archive_path)
        self.assertIsInstance(osm_file_create, OSM)
        self.assertEqual(osm_file_create.region, 'Uganda')
        self.assertEqual(osm_file_create.district, 'Kasese')
        self.assertEqual(osm_file_create.mgrs, '35MRV')

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'osm', 'Uganda', 'Kibaale')
        os.makedirs(district_1)
        for file in [
            'Uganda_Kibaale_35MRV.tif',
            'Uganda_Kibaale_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'osm', 'Uganda', 'Kasese')
        for file in [
            'Uganda_Kasese_35MRV.tif',
            'Uganda_Kasese_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'osm', 'Rwanda', 'all')
        for file in [
            'Rwanda_all_35MRV.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(OSM.find_files(), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif'),
            os.path.join(district_2, 'Uganda_Kasese_35MRV.tif'),
            os.path.join(district_2, 'Uganda_Kasese_36MGR.tif'),
            os.path.join(district_3, 'Rwanda_all_35MRV.tif'),
        ]))
        self.assertEqual(OSM.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
                             os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif'),
                             os.path.join(district_2, 'Uganda_Kasese_35MRV.tif'),
                             os.path.join(district_2, 'Uganda_Kasese_36MGR.tif')
                         ]))
        self.assertEqual(OSM.find_files(region='Uganda',
                                        district='Kibaale'), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'Uganda_Kibaale_36MGR.tif')
        ]))
        self.assertEqual(OSM.find_files(region='Uganda',
                                        district='Kibaale', mgrs='35MRV'), sorted([
            os.path.join(district_1, 'Uganda_Kibaale_35MRV.tif')
        ]))
        self.assertEqual(OSM.find_files(region='Rwanda',
                                        district='all', mgrs='35MRV'),
                         sorted([
                             os.path.join(district_3, 'Rwanda_all_35MRV.tif'),
                         ]))
        self.assertEqual(OSM.find_files(region='Rwanda', district='all', mgrs='37MRV'), [])


class TestSingleRegionTileMatch(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        SingleRegionTileMatch._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'tiles')

    def test_init(self):
        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400)
        self.assertIsNotNone(tile_match)
        self.assertEqual(tile_match.region, 'Uganda')
        self.assertEqual(tile_match.tile_size, 400)
        self.assertIsNone(tile_match.district)
        self.assertIsNone(tile_match.mgrs)

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole')
        self.assertIsNotNone(tile_match)
        self.assertEqual(tile_match.region, 'Uganda')
        self.assertEqual(tile_match.tile_size, 400)
        self.assertEqual(tile_match.district, 'Kabarole')
        self.assertIsNone(tile_match.mgrs)

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole', military_grid='35MRV')
        self.assertIsNotNone(tile_match)
        self.assertEqual(tile_match.region, 'Uganda')
        self.assertEqual(tile_match.tile_size, 400)
        self.assertEqual(tile_match.district, 'Kabarole')
        self.assertEqual(tile_match.mgrs, '35MRV')

    def test_name(self):
        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400)
        self.assertEqual(tile_match.name, 'tile_match_Uganda_400.csv')

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole')
        self.assertEqual(tile_match.name, 'tile_match_Uganda_Kabarole_400.csv')

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole', military_grid='35MRV')
        self.assertEqual(tile_match.name, 'tile_match_Uganda_Kabarole_35MRV_400.csv')

    def test_archive_path(self):
        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400)
        self.assertEqual(tile_match.archive_path, os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda',
                                                               'tile_match_Uganda_400.csv'))

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole')
        self.assertEqual(tile_match.archive_path, os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kabarole',
                                                               'tile_match_Uganda_Kabarole_400.csv'))

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole', military_grid='35MRV')
        self.assertEqual(tile_match.archive_path, os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kabarole',
                                                               '35MRV', 'tile_match_Uganda_Kabarole_35MRV_400.csv'))

    def test_create(self):
        pass

    def test_find_files(self):
        pass
