import shutil
from unittest import TestCase
from file_types import _BaseCompositeFile, _BaseSentinel2File, _BaseTileFile, \
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


class TestFileType(TestFileTypes):
    file_type_inits = [
            OpticalComposite(region='Uganda', district='Kibaale', military_grid='35MRV',
                                                    bands=['B03', 'B02', 'B04']),
            OpticalCompositeSlice(region='Uganda', district='Kibaale', military_grid='35MRV',
                    band='B03', left_bound=0, right_bound=500),
            MultiVariateComposite(region='Uganda', district='Kibaale', military_grid='35MRV'),
            Sentinel2Tile(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                                                square='V', year=2020, month=1, day=1, band='B02'),
            Sentinel2Cloud(region='Uganda', district='Kafefe', utm_code='35', latitude_band='MR',
                square='V', year=2020, month=1, day=1),
            Elevation(region='Uganda', district='Kasese', mgrs='35MRV'),
            Slope(region='Uganda', district='Kasese', mgrs='35MRV'),
            OSM(region='Uganda', district='Kasese', mgrs='35MRV'),
            SingleRegionTileMatch(region='Uganda', tile_size=400),
            MultiRegionTileMatch(regions=['Uganda', 'Rwanda'], tile_size=300),
            Tile(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300),
            PyTorch(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300),
            TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                                layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False),
            InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                                layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False),
            InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                                layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False),
            TrainSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300),
            ValidateSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        ]
    
    def test_create(self):
        for file_type in self.file_type_inits:
            created_file = File.create(file_type.name)
            self.assertIsInstance(created_file, type(file_type))
    

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
        self.assertEqual(elevation_file.name, 'elevation_Uganda_Kasese_35MRV.tif')

        elevation_file = Elevation(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(elevation_file.name, 'elevation_Rwanda_all_36MRV.tif')

    def test_archive_path(self):
        elevation_file = Elevation(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(elevation_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'elevation', 'Uganda', 'Kasese',
                                                                   'elevation_Uganda_Kasese_35MRV.tif'))

        elevation_file = Elevation(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(elevation_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'elevation', 'Rwanda', 'all',
                                                                   'elevation_Rwanda_all_36MRV.tif'))

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
            'elevation_Uganda_Kibaale_35MRV.tif',
            'elevation_Uganda_Kibaale_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'elevation', 'Uganda', 'Kasese')
        for file in [
            'elevation_Uganda_Kasese_35MRV.tif',
            'elevation_Uganda_Kasese_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'elevation', 'Rwanda', 'all')
        for file in [
            'elevation_Rwanda_all_35MRV.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(Elevation.find_files(), sorted([
            os.path.join(district_1, 'elevation_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'elevation_Uganda_Kibaale_36MGR.tif'),
            os.path.join(district_2, 'elevation_Uganda_Kasese_35MRV.tif'),
            os.path.join(district_2, 'elevation_Uganda_Kasese_36MGR.tif'),
            os.path.join(district_3, 'elevation_Rwanda_all_35MRV.tif'),
        ]))
        self.assertEqual(Elevation.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'elevation_Uganda_Kibaale_35MRV.tif'),
                             os.path.join(district_1, 'elevation_Uganda_Kibaale_36MGR.tif'),
                             os.path.join(district_2, 'elevation_Uganda_Kasese_35MRV.tif'),
                             os.path.join(district_2, 'elevation_Uganda_Kasese_36MGR.tif')
                         ]))
        self.assertEqual(Elevation.find_files(region='Uganda',
                                              district='Kibaale'), sorted([
            os.path.join(district_1, 'elevation_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'elevation_Uganda_Kibaale_36MGR.tif')
        ]))
        self.assertEqual(Elevation.find_files(region='Uganda',
                                              district='Kibaale', mgrs='35MRV'), sorted([
            os.path.join(district_1, 'elevation_Uganda_Kibaale_35MRV.tif')
        ]))
        self.assertEqual(Elevation.find_files(region='Rwanda',
                                              district='all', mgrs='35MRV'),
                         sorted([
                             os.path.join(district_3, 'elevation_Rwanda_all_35MRV.tif'),
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
        self.assertEqual(slope_file.name, 'slope_Uganda_Kasese_35MRV.tif')

        slope_file = Slope(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(slope_file.name, 'slope_Rwanda_all_36MRV.tif')

    def test_archive_path(self):
        slope_file = Slope(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(slope_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'slope', 'Uganda', 'Kasese',
                                                               'slope_Uganda_Kasese_35MRV.tif'))

        slope_file = Slope(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(slope_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'slope', 'Rwanda', 'all',
                                                               'slope_Rwanda_all_36MRV.tif'))

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
            'slope_Uganda_Kibaale_35MRV.tif',
            'slope_Uganda_Kibaale_36MGR.tif',
            'slope_off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'slope', 'Uganda', 'Kasese')
        for file in [
            'slope_Uganda_Kasese_35MRV.tif',
            'slope_Uganda_Kasese_36MGR.tif',
            'slope_off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'slope', 'Rwanda', 'all')
        for file in [
            'slope_Rwanda_all_35MRV.tif',
            'slope_off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(Slope.find_files(), sorted([
            os.path.join(district_1, 'slope_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'slope_Uganda_Kibaale_36MGR.tif'),
            os.path.join(district_2, 'slope_Uganda_Kasese_35MRV.tif'),
            os.path.join(district_2, 'slope_Uganda_Kasese_36MGR.tif'),
            os.path.join(district_3, 'slope_Rwanda_all_35MRV.tif'),
        ]))
        self.assertEqual(Slope.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'slope_Uganda_Kibaale_35MRV.tif'),
                             os.path.join(district_1, 'slope_Uganda_Kibaale_36MGR.tif'),
                             os.path.join(district_2, 'slope_Uganda_Kasese_35MRV.tif'),
                             os.path.join(district_2, 'slope_Uganda_Kasese_36MGR.tif')
                         ]))
        self.assertEqual(Slope.find_files(region='Uganda',
                                          district='Kibaale'), sorted([
            os.path.join(district_1, 'slope_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'slope_Uganda_Kibaale_36MGR.tif')
        ]))
        self.assertEqual(Slope.find_files(region='Uganda',
                                          district='Kibaale', mgrs='35MRV'), sorted([
            os.path.join(district_1, 'slope_Uganda_Kibaale_35MRV.tif')
        ]))
        self.assertEqual(Slope.find_files(region='Rwanda',
                                          district='all', mgrs='35MRV'),
                         sorted([
                             os.path.join(district_3, 'slope_Rwanda_all_35MRV.tif'),
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
        self.assertEqual(osm_file.name, 'osm_Uganda_Kasese_35MRV.tif')

        osm_file = OSM(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(osm_file.name, 'osm_Rwanda_all_36MRV.tif')

    def test_archive_path(self):
        osm_file = OSM(region='Uganda', district='Kasese', mgrs='35MRV')
        self.assertEqual(osm_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'osm', 'Uganda', 'Kasese',
                                                             'osm_Uganda_Kasese_35MRV.tif'))

        osm_file = OSM(region='Rwanda', district='all', mgrs='36MRV')
        self.assertEqual(osm_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'osm', 'Rwanda', 'all',
                                                             'osm_Rwanda_all_36MRV.tif'))

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
            'osm_Uganda_Kibaale_35MRV.tif',
            'osm_Uganda_Kibaale_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'osm', 'Uganda', 'Kasese')
        for file in [
            'osm_Uganda_Kasese_35MRV.tif',
            'osm_Uganda_Kasese_36MGR.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'osm', 'Rwanda', 'all')
        for file in [
            'osm_Rwanda_all_35MRV.tif',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(OSM.find_files(), sorted([
            os.path.join(district_1, 'osm_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'osm_Uganda_Kibaale_36MGR.tif'),
            os.path.join(district_2, 'osm_Uganda_Kasese_35MRV.tif'),
            os.path.join(district_2, 'osm_Uganda_Kasese_36MGR.tif'),
            os.path.join(district_3, 'osm_Rwanda_all_35MRV.tif'),
        ]))
        self.assertEqual(OSM.find_files(region='Uganda'),
                         sorted([
                             os.path.join(district_1, 'osm_Uganda_Kibaale_35MRV.tif'),
                             os.path.join(district_1, 'osm_Uganda_Kibaale_36MGR.tif'),
                             os.path.join(district_2, 'osm_Uganda_Kasese_35MRV.tif'),
                             os.path.join(district_2, 'osm_Uganda_Kasese_36MGR.tif')
                         ]))
        self.assertEqual(OSM.find_files(region='Uganda',
                                        district='Kibaale'), sorted([
            os.path.join(district_1, 'osm_Uganda_Kibaale_35MRV.tif'),
            os.path.join(district_1, 'osm_Uganda_Kibaale_36MGR.tif')
        ]))
        self.assertEqual(OSM.find_files(region='Uganda',
                                        district='Kibaale', mgrs='35MRV'), sorted([
            os.path.join(district_1, 'osm_Uganda_Kibaale_35MRV.tif')
        ]))
        self.assertEqual(OSM.find_files(region='Rwanda',
                                        district='all', mgrs='35MRV'),
                         sorted([
                             os.path.join(district_3, 'osm_Rwanda_all_35MRV.tif'),
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
        self.assertEqual(tile_match.name, 'sr_tile_match_Uganda_400.csv')

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole')
        self.assertEqual(tile_match.name, 'sr_tile_match_Uganda_Kabarole_400.csv')

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole', military_grid='35MRV')
        self.assertEqual(tile_match.name, 'sr_tile_match_Uganda_Kabarole_35MRV_400.csv')

    def test_archive_path(self):
        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400)
        self.assertEqual(tile_match.archive_path, os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda',
                                                               'sr_tile_match_Uganda_400.csv'))

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole')
        self.assertEqual(tile_match.archive_path, os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kabarole',
                                                               'sr_tile_match_Uganda_Kabarole_400.csv'))

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kabarole', military_grid='35MRV')
        self.assertEqual(tile_match.archive_path, os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kabarole',
                                                               '35MRV', 'sr_tile_match_Uganda_Kabarole_35MRV_400.csv'))

    def test_create(self):
        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400)
        tile_match_create = SingleRegionTileMatch.create(tile_match.name)
        self.assertIsInstance(tile_match_create, SingleRegionTileMatch)
        self.assertEqual(tile_match_create.region, 'Uganda')
        self.assertEqual(tile_match_create.tile_size, 400)
        self.assertIsNone(tile_match_create.district)
        self.assertIsNone(tile_match_create.mgrs)

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kafefe')
        tile_match_create = SingleRegionTileMatch.create(tile_match.name)
        self.assertIsInstance(tile_match_create, SingleRegionTileMatch)
        self.assertEqual(tile_match_create.region, 'Uganda')
        self.assertEqual(tile_match_create.tile_size, 400)
        self.assertEqual(tile_match_create.district, 'Kafefe')
        self.assertIsNone(tile_match_create.mgrs)

        tile_match = SingleRegionTileMatch(region='Uganda', tile_size=400, district='Kafefe', military_grid='35MRV')
        tile_match_create = SingleRegionTileMatch.create(tile_match.name)
        self.assertIsInstance(tile_match_create, SingleRegionTileMatch)
        self.assertEqual(tile_match_create.region, 'Uganda')
        self.assertEqual(tile_match_create.tile_size, 400)
        self.assertEqual(tile_match_create.district, 'Kafefe')
        self.assertEqual(tile_match_create.mgrs, '35MRV')

        tile_match = SingleRegionTileMatch(region="Cote d'Ivoire", tile_size=400, district='Kafefe', military_grid='35MRV')
        tile_match_create = SingleRegionTileMatch.create(tile_match.name)
        self.assertIsInstance(tile_match_create, SingleRegionTileMatch)
        self.assertEqual(tile_match_create.region, "Cote d'Ivoire")
        self.assertEqual(tile_match_create.tile_size, 400)
        self.assertEqual(tile_match_create.district, 'Kafefe')
        self.assertEqual(tile_match_create.mgrs, '35MRV')

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda')
        os.makedirs(district_1)
        for file in [
            'sr_tile_match_Uganda_300.csv',
            'sr_tile_match_Uganda_400.csv',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kibaale')
        os.makedirs(district_2)
        for file in [
            'sr_tile_match_Uganda_Kibaale_300.csv',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kibaale', '35MRV')
        os.makedirs(district_3)
        for file in [
            'sr_tile_match_Uganda_Kibaale_35MRV_300.csv',
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))

        self.assertEqual(SingleRegionTileMatch.find_files(tile_size=300), sorted([
            os.path.join(district_1, 'sr_tile_match_Uganda_300.csv',),
            os.path.join(district_2, 'sr_tile_match_Uganda_Kibaale_300.csv'),
            os.path.join(district_3, 'sr_tile_match_Uganda_Kibaale_35MRV_300.csv')
        ]))

        self.assertEqual(SingleRegionTileMatch.find_files(tile_size=400), sorted([
            os.path.join(district_1, 'sr_tile_match_Uganda_400.csv')
        ]))

        self.assertEqual(SingleRegionTileMatch.find_files(tile_size=300, region='Uganda'), sorted([
            os.path.join(district_1, 'sr_tile_match_Uganda_300.csv'),
            os.path.join(district_2, 'sr_tile_match_Uganda_Kibaale_300.csv'),
            os.path.join(district_3, 'sr_tile_match_Uganda_Kibaale_35MRV_300.csv')
        ]))

        self.assertEqual(SingleRegionTileMatch.find_files(tile_size=300, region='Uganda', district='Kibaale'), sorted([
            os.path.join(district_2, 'sr_tile_match_Uganda_Kibaale_300.csv'),
            os.path.join(district_3, 'sr_tile_match_Uganda_Kibaale_35MRV_300.csv')
        ]))

        self.assertEqual(SingleRegionTileMatch.find_files(tile_size=300, region='Uganda', district='Kibaale',
                                                          military_grid='35MRV'), sorted([
            os.path.join(district_3, 'sr_tile_match_Uganda_Kibaale_35MRV_300.csv')
        ]))


class TestMultiRegionTileMatch(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        MultiRegionTileMatch._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'multi_region_tile_match')

    def test_init(self):
        tile_match = MultiRegionTileMatch(regions=['Uganda', 'Rwanda'], tile_size=300)
        self.assertIsNotNone(tile_match)
        self.assertEqual(tile_match.regions, ['Rwanda', 'Uganda'])
        self.assertEqual(tile_match.tile_size, 300)

    def test_name(self):
        tile_match = MultiRegionTileMatch(regions=['Uganda', 'Rwanda'], tile_size=300)
        self.assertEqual(tile_match.name, 'mr_tile_match_Rwanda_Uganda_300.csv')

    def test_archive_path(self):
        tile_match = MultiRegionTileMatch(regions=['Uganda', 'Rwanda'], tile_size=300)
        self.assertEqual(tile_match.archive_path, os.path.join(self.TEST_DATA_DIR, 'multi_region_tile_match',
                                                               'mr_tile_match_Rwanda_Uganda_300.csv'))

    def test_create(self):
        tile_match = MultiRegionTileMatch(regions=['Uganda', 'Rwanda'], tile_size=300)
        tile_match_created = MultiRegionTileMatch.create(tile_match.name)
        self.assertIsInstance(tile_match_created, MultiRegionTileMatch)
        self.assertEqual(tile_match_created.regions, ['Rwanda', 'Uganda'])
        self.assertEqual(tile_match_created.tile_size, 300)

        tile_match = MultiRegionTileMatch(regions=['Uganda', 'Rwanda'], tile_size=300)
        tile_match_created = MultiRegionTileMatch.create(tile_match.archive_path)
        self.assertIsInstance(tile_match_created, MultiRegionTileMatch)
        self.assertEqual(tile_match_created.regions, ['Rwanda', 'Uganda'])
        self.assertEqual(tile_match_created.tile_size, 300)

    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'multi_region_tile_match')
        os.makedirs(district_1)
        for file in [
            'mr_tile_match_Rwanda_Uganda_300.csv',
            "mr_tile_match_Cote d'Ivoire_Rwanda_Uganda_300.csv",
            "mr_tile_match_Cote d'Ivoire_Rwanda_Uganda_400.csv",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))
        self.assertEqual(MultiRegionTileMatch.find_files(tile_size=300), sorted([
            os.path.join(district_1, 'mr_tile_match_Rwanda_Uganda_300.csv'),
            os.path.join(district_1, "mr_tile_match_Cote d'Ivoire_Rwanda_Uganda_300.csv"),
        ]))


class TestTile(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseTileFile._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'tiles')
    
    def test_init(self):
        tile = Tile(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300)
        self.assertIsNotNone(tile)
        self.assertEqual(tile.region, "Cote D'Ivoire")
        self.assertEqual(tile.district, 'Kafefe')
        self.assertEqual(tile.mgrs, "35MRV")
        self.assertEqual(tile.tile_size, 350)
        self.assertEqual(tile.x_min, 200)
        self.assertEqual(tile.y_min, 300)
    
    def test_name(self):
        tile = Tile(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300)
        self.assertEqual(tile.name, "Cote D'Ivoire_Kafefe_35MRV_350_200_300.tif")

    def test_archive_path(self):
        tile = Tile(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300)
        self.assertEqual(tile.archive_path, os.path.join(self.TEST_DATA_DIR, 'tiles', "Cote D'Ivoire", 'Kafefe', '35MRV', '350',
                                                          "Cote D'Ivoire_Kafefe_35MRV_350_200_300.tif"))
    
    def test_create(self):
        tile = Tile(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300)
        tile_created = Tile.create(tile.name)
        self.assertIsInstance(tile_created, Tile)
        self.assertEqual(tile_created.region, "Cote D'Ivoire")
        self.assertEqual(tile_created.district, 'Kafefe')
        self.assertEqual(tile_created.mgrs, '35MRV')
        self.assertEqual(tile_created.tile_size, 350)
        self.assertEqual(tile_created.x_min, 200)
        self.assertEqual(tile_created.y_min, 300)
    
    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kafefe', '35MRV', '350')
        os.makedirs(district_1)
        for file in [
            "Uganda_Kafefe_35MRV_350_200_300.tif",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kafefe', '35MRV', '400')
        os.makedirs(district_2)
        for file in [
            "Uganda_Kafefe_35MRV_400_200_300.tif",
            "Uganda_Kafefe_35MRV_400_300_400.tif",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'tiles', "Cote D'Ivoire", 'Kibaale', '36MRV', '300')
        os.makedirs(district_3)
        for file in [
            "Cote D'Ivoire_Kibaale_36MRV_300_200_300.tif",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))
        
        self.assertEqual(Tile.find_files(), sorted([
            os.path.join(district_1, "Uganda_Kafefe_35MRV_350_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.tif"),
            os.path.join(district_3, "Cote D'Ivoire_Kibaale_36MRV_300_200_300.tif")
        ]))

        self.assertEqual(Tile.find_files(region='Uganda'), sorted([
            os.path.join(district_1, "Uganda_Kafefe_35MRV_350_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.tif")
        ]))

        self.assertEqual(Tile.find_files(region='Uganda', district='Kafefe'), sorted([
            os.path.join(district_1, "Uganda_Kafefe_35MRV_350_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.tif")
        ]))

        self.assertEqual(Tile.find_files(region='Uganda', district='Kafefe', military_grid='35MRV'), sorted([
            os.path.join(district_1, "Uganda_Kafefe_35MRV_350_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.tif")
        ]))

        self.assertEqual(Tile.find_files(region='Uganda', district='Kafefe', military_grid='35MRV', tile_size=400), sorted([
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.tif"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.tif")
        ]))


class TestPyTorch(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseTileFile._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'tiles')

    def test_init(self):
        tile = PyTorch(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300)
        self.assertIsNotNone(tile)
        self.assertEqual(tile.region, "Cote D'Ivoire")
        self.assertEqual(tile.district, 'Kafefe')
        self.assertEqual(tile.mgrs, "35MRV")
        self.assertEqual(tile.tile_size, 350)
        self.assertEqual(tile.x_min, 200)
        self.assertEqual(tile.y_min, 300)
    
    def test_name(self):
        tile = PyTorch(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300)
        self.assertEqual(tile.name, "Cote D'Ivoire_Kafefe_35MRV_350_200_300.pt")

    def test_archive_path(self):
        tile = PyTorch(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300)
        self.assertEqual(tile.archive_path, os.path.join(self.TEST_DATA_DIR, 'tiles', "Cote D'Ivoire", 'Kafefe', '35MRV', '350',
                                                         "Cote D'Ivoire_Kafefe_35MRV_350_200_300.pt"))
    
    def test_create(self):
        tile = PyTorch(region="Cote D'Ivoire", district='Kafefe', military_grid='35MRV', tile_size=350, x_min=200, y_min=300)
        tile_created = PyTorch.create(tile.name)
        self.assertIsInstance(tile_created, PyTorch)
        self.assertEqual(tile_created.region, "Cote D'Ivoire")
        self.assertEqual(tile_created.district, 'Kafefe')
        self.assertEqual(tile_created.mgrs, '35MRV')
        self.assertEqual(tile_created.tile_size, 350)
        self.assertEqual(tile_created.x_min, 200)
        self.assertEqual(tile_created.y_min, 300)
    
    def test_find_files(self):
        district_1 = os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kafefe', '35MRV', '350')
        os.makedirs(district_1)
        for file in [
            "Uganda_Kafefe_35MRV_350_200_300.pt",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_1, file))

        district_2 = os.path.join(self.TEST_DATA_DIR, 'tiles', 'Uganda', 'Kafefe', '35MRV', '400')
        os.makedirs(district_2)
        for file in [
            "Uganda_Kafefe_35MRV_400_200_300.pt",
            "Uganda_Kafefe_35MRV_400_300_400.pt",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_2, file))

        district_3 = os.path.join(self.TEST_DATA_DIR, 'tiles', "Cote D'Ivoire", 'Kibaale', '36MRV', '300')
        os.makedirs(district_3)
        for file in [
            "Cote D'Ivoire_Kibaale_36MRV_300_200_300.pt",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(district_3, file))
        
        self.assertEqual(PyTorch.find_files(), sorted([
            os.path.join(district_1, "Uganda_Kafefe_35MRV_350_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.pt"),
            os.path.join(district_3, "Cote D'Ivoire_Kibaale_36MRV_300_200_300.pt")
        ]))

        self.assertEqual(PyTorch.find_files(region='Uganda'), sorted([
            os.path.join(district_1, "Uganda_Kafefe_35MRV_350_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.pt")
        ]))

        self.assertEqual(PyTorch.find_files(region='Uganda', district='Kafefe'), sorted([
            os.path.join(district_1, "Uganda_Kafefe_35MRV_350_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.pt")
        ]))

        self.assertEqual(PyTorch.find_files(region='Uganda', district='Kafefe', military_grid='35MRV'), sorted([
            os.path.join(district_1, "Uganda_Kafefe_35MRV_350_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.pt")
        ]))

        self.assertEqual(PyTorch.find_files(region='Uganda', district='Kafefe', military_grid='35MRV', tile_size=400), sorted([
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_200_300.pt"),
            os.path.join(district_2, "Uganda_Kafefe_35MRV_400_300_400.pt")
        ]))


class TestTrainedModel(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        TrainedModel._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'trained_models')
    
    def test_init(self):
        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertIsNotNone(trained_model_file)
        self.assertEqual(trained_model_file.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(trained_model_file.architecture, 'resnet18')
        self.assertEqual(trained_model_file.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(trained_model_file.epoch, 5)
        self.assertEqual(trained_model_file.ratio, 5.0)
        self.assertEqual(trained_model_file.tile_size, 300)
        self.assertFalse(trained_model_file.best)

        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertIsNotNone(trained_model_file)
        self.assertEqual(trained_model_file.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(trained_model_file.architecture, 'resnet18')
        self.assertEqual(trained_model_file.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(trained_model_file.epoch, 5)
        self.assertEqual(trained_model_file.ratio, 5.0)
        self.assertEqual(trained_model_file.tile_size, 300)
        self.assertTrue(trained_model_file.best)

    def test_name(self):
        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertEqual(trained_model_file.name, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar")

        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertEqual(trained_model_file.name, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5_best.tar")

    def test_archive_path(self):
        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertEqual(trained_model_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'trained_models', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0',
                         "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar"))

        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertEqual(trained_model_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'trained_models', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0',
                                                               "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5_best.tar"))
        
    def test_create(self):
        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        trained_model_created = TrainedModel.create(trained_model_file.name)
        self.assertIsInstance(trained_model_created, TrainedModel)
        self.assertEqual(trained_model_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(trained_model_created.architecture, 'resnet18')
        self.assertEqual(trained_model_created.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(trained_model_created.epoch, 5)
        self.assertEqual(trained_model_created.ratio, 5.0)
        self.assertEqual(trained_model_created.tile_size, 300)
        self.assertFalse(trained_model_created.best)

        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['red'], epoch=5, ratio=5.0, tile_size=300, best=True)
        trained_model_created = TrainedModel.create(trained_model_file.name)
        self.assertIsInstance(trained_model_created, TrainedModel)
        self.assertEqual(trained_model_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(trained_model_created.architecture, 'resnet18')
        self.assertEqual(trained_model_created.layers, ['red'])
        self.assertEqual(trained_model_created.epoch, 5)
        self.assertEqual(trained_model_created.ratio, 5.0)
        self.assertEqual(trained_model_created.tile_size, 300)
        self.assertTrue(trained_model_created.best)

        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['blue', 'osm-boundary'], epoch=5, ratio=5.0, tile_size=300, best=True)
        trained_model_created = TrainedModel.create(trained_model_file.name)
        self.assertIsInstance(trained_model_created, TrainedModel)
        self.assertEqual(trained_model_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(trained_model_created.architecture, 'resnet18')
        self.assertEqual(trained_model_created.layers, ['blue', 'osm-boundary'])
        self.assertEqual(trained_model_created.epoch, 5)
        self.assertEqual(trained_model_created.ratio, 5.0)
        self.assertEqual(trained_model_created.tile_size, 300)
        self.assertTrue(trained_model_created.best)

        trained_model_file = TrainedModel(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['elevation', 'slope'], epoch=5, ratio=5.0, tile_size=300, best=True)
        trained_model_created = TrainedModel.create(trained_model_file.name)
        self.assertIsInstance(trained_model_created, TrainedModel)
        self.assertEqual(trained_model_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(trained_model_created.architecture, 'resnet18')
        self.assertEqual(trained_model_created.layers, ['elevation', 'slope'])
        self.assertEqual(trained_model_created.epoch, 5)
        self.assertEqual(trained_model_created.ratio, 5.0)
        self.assertEqual(trained_model_created.tile_size, 300)
        self.assertTrue(trained_model_created.best)

    def test_find_files(self):
        dir_1 = os.path.join(self.TEST_DATA_DIR, 'trained_models', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0')
        os.makedirs(dir_1)
        for file in [
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar",
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.tar",
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_epoch5.tar",
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.tar",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_1, file))

        dir_2 = os.path.join(self.TEST_DATA_DIR, 'trained_models', "Cote D'Ivoire", 'resnet18', 'r5.0')
        os.makedirs(dir_2)
        for file in [
            "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar",
            "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.tar",
            "Cote D'Ivoire_resnet18_r5.0_ts300_green_epoch5.tar",
            "Cote D'Ivoire_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.tar",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_2, file))

        dir_3 = os.path.join(self.TEST_DATA_DIR, 'trained_models', "Cote D'Ivoire", 'resnet50', 'r5.0')
        os.makedirs(dir_3)
        for file in [
            "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch5.tar",
            "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch10.tar",
            "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.tar",
            "Cote D'Ivoire_resnet50_r5.0_ts300_elevation_nir_slope_epoch5.tar",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_3, file))
        
        dir_4 = os.path.join(self.TEST_DATA_DIR, 'trained_models', "Cote D'Ivoire", 'resnet50', 'r10.0')
        os.makedirs(dir_4)
        for file in [
            "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.tar",
            "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.tar",
            "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.tar",
            "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.tar",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_4, file))

        self.assertEqual(TrainedModel.find_files(), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_epoch5.tar"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.tar"),

            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_epoch5.tar"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.tar"),

            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_elevation_nir_slope_epoch5.tar"),

            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.tar")
        ]))

        self.assertEqual(TrainedModel.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"]), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_epoch5.tar"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.tar")
        ]))

        self.assertEqual(TrainedModel.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['osm-water', 'nir', 'green']), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.tar")
        ]))

        self.assertEqual(TrainedModel.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['elevation', 'nir', 'slope']), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.tar")
        ]))

        self.assertEqual(TrainedModel.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['green']), sorted([]))

        self.assertEqual(TrainedModel.find_files(regions=[ "Cote D'Ivoire"]), sorted([
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_epoch5.tar"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.tar"),

            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_elevation_nir_slope_epoch5.tar"),

            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.tar")

        ]))

        self.assertEqual(TrainedModel.find_files(architecture='resnet50'), sorted([
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.tar"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_elevation_nir_slope_epoch5.tar"),

            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.tar")
        ]))

        self.assertEqual(TrainedModel.find_files(ratio=10), sorted([
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.tar"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.tar")
        ]))

class TestInferenceResultsCSV(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        InferenceResultsCSV._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'inference_results')

    def test_init(self):
        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertIsNotNone(ir_file)
        self.assertEqual(ir_file.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file.architecture, 'resnet18')
        self.assertEqual(ir_file.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(ir_file.epoch, 5)
        self.assertEqual(ir_file.ratio, 5.0)
        self.assertEqual(ir_file.tile_size, 300)
        self.assertFalse(ir_file.best)

        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertIsNotNone(ir_file)
        self.assertEqual(ir_file.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file.architecture, 'resnet18')
        self.assertEqual(ir_file.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(ir_file.epoch, 5)
        self.assertEqual(ir_file.ratio, 5.0)
        self.assertEqual(ir_file.tile_size, 300)
        self.assertTrue(ir_file.best)

    def test_name(self):
        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertEqual(ir_file.name, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv")

        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertEqual(ir_file.name, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5_best.csv")

    def test_archive_path(self):
        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertEqual(ir_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0',
                         "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv"))

        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertEqual(ir_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0',
                                                               "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5_best.csv"))
        
    def test_create(self):
        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        ir_file_created = InferenceResultsCSV.create(ir_file.name)
        self.assertIsInstance(ir_file_created, InferenceResultsCSV)
        self.assertEqual(ir_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file_created.architecture, 'resnet18')
        self.assertEqual(ir_file_created.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(ir_file_created.epoch, 5)
        self.assertEqual(ir_file_created.ratio, 5.0)
        self.assertEqual(ir_file_created.tile_size, 300)
        self.assertFalse(ir_file_created.best)

        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['red'], epoch=5, ratio=5.0, tile_size=300, best=True)
        ir_file_created = InferenceResultsCSV.create(ir_file.name)
        self.assertIsInstance(ir_file_created, InferenceResultsCSV)
        self.assertEqual(ir_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file_created.architecture, 'resnet18')
        self.assertEqual(ir_file_created.layers, ['red'])
        self.assertEqual(ir_file_created.epoch, 5)
        self.assertEqual(ir_file_created.ratio, 5.0)
        self.assertEqual(ir_file_created.tile_size, 300)
        self.assertTrue(ir_file_created.best)

        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['blue', 'osm-boundary'], epoch=5, ratio=5.0, tile_size=300, best=True)
        ir_file_created = InferenceResultsCSV.create(ir_file.name)
        self.assertIsInstance(ir_file_created, InferenceResultsCSV)
        self.assertEqual(ir_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file_created.architecture, 'resnet18')
        self.assertEqual(ir_file_created.layers, ['blue', 'osm-boundary'])
        self.assertEqual(ir_file_created.epoch, 5)
        self.assertEqual(ir_file_created.ratio, 5.0)
        self.assertEqual(ir_file_created.tile_size, 300)
        self.assertTrue(ir_file_created.best)

        ir_file = InferenceResultsCSV(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['elevation', 'slope'], epoch=5, ratio=5.0, tile_size=300, best=True)
        ir_file_created = InferenceResultsCSV.create(ir_file.name)
        self.assertIsInstance(ir_file_created, InferenceResultsCSV)
        self.assertEqual(ir_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file_created.architecture, 'resnet18')
        self.assertEqual(ir_file_created.layers, ['elevation', 'slope'])
        self.assertEqual(ir_file_created.epoch, 5)
        self.assertEqual(ir_file_created.ratio, 5.0)
        self.assertEqual(ir_file_created.tile_size, 300)
        self.assertTrue(ir_file_created.best)

    def test_find_files(self):
        dir_1 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0')
        os.makedirs(dir_1)
        for file in [
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv",
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.csv",
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_epoch5.csv",
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.csv",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_1, file))

        dir_2 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire", 'resnet18', 'r5.0')
        os.makedirs(dir_2)
        for file in [
            "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv",
            "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.csv",
            "Cote D'Ivoire_resnet18_r5.0_ts300_green_epoch5.csv",
            "Cote D'Ivoire_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.csv",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_2, file))

        dir_3 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire", 'resnet50', 'r5.0')
        os.makedirs(dir_3)
        for file in [
            "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch5.csv",
            "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch10.csv",
            "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.csv",
            "Cote D'Ivoire_resnet50_r5.0_ts300_elevation_nir_slope_epoch5.csv",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_3, file))
        
        dir_4 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire", 'resnet50', 'r10.0')
        os.makedirs(dir_4)
        for file in [
            "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.csv",
            "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.csv",
            "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.csv",
            "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.csv",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_4, file))

        self.assertEqual(InferenceResultsCSV.find_files(), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_epoch5.csv"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.csv"),

            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_epoch5.csv"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.csv"),

            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_elevation_nir_slope_epoch5.csv"),

            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.csv")
        ]))

        self.assertEqual(InferenceResultsCSV.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"]), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_epoch5.csv"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.csv")
        ]))

        self.assertEqual(InferenceResultsCSV.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['osm-water', 'nir', 'green']), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.csv")
        ]))

        self.assertEqual(InferenceResultsCSV.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['elevation', 'nir', 'slope']), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.csv")
        ]))

        self.assertEqual(InferenceResultsCSV.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['green']), sorted([]))

        self.assertEqual(InferenceResultsCSV.find_files(regions=[ "Cote D'Ivoire"]), sorted([
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_epoch5.csv"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_elevation_nir_slope_epoch5.csv"),

            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_elevation_nir_slope_epoch5.csv"),

            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.csv")

        ]))

        self.assertEqual(InferenceResultsCSV.find_files(architecture='resnet50'), sorted([
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.csv"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_elevation_nir_slope_epoch5.csv"),

            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.csv")
        ]))

        self.assertEqual(InferenceResultsCSV.find_files(ratio=10), sorted([
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch5.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_nir_osm-water_epoch10.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_green_epoch5.csv"),
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5.csv")
        ]))


class TestInferenceResultsShapefile(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        InferenceResultsShapefile._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'inference_results')

    def tearDown(self) -> None:
        shutil.rmtree(os.path.join(self.TEST_DATA_DIR, 'inference_results'))
        return super().tearDown()
    
    def setUp(self) -> None:
        Path(os.path.join(self.TEST_DATA_DIR, 'inference_results')).mkdir(parents=True, exist_ok=True)
        return super().setUp()

    def test_init(self):
        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertIsNotNone(ir_file)
        self.assertEqual(ir_file.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file.architecture, 'resnet18')
        self.assertEqual(ir_file.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(ir_file.epoch, 5)
        self.assertEqual(ir_file.ratio, 5.0)
        self.assertEqual(ir_file.tile_size, 300)
        self.assertFalse(ir_file.best)

        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertIsNotNone(ir_file)
        self.assertEqual(ir_file.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file.architecture, 'resnet18')
        self.assertEqual(ir_file.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(ir_file.epoch, 5)
        self.assertEqual(ir_file.ratio, 5.0)
        self.assertEqual(ir_file.tile_size, 300)
        self.assertTrue(ir_file.best)

    def test_name(self):
        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertEqual(ir_file.name, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.shp")

        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertEqual(ir_file.name, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5_best.shp")

    def test_archive_path(self):
        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertEqual(ir_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0',
                                                            'green_nir_osm-water_epoch5_shapefile',
                         "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.shp"))

        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertEqual(ir_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0',
                                                            'green_nir_osm-water_epoch5_best_shapefile',
                                                               "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5_best.shp"))
        
    def test_create(self):
        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        ir_file_created = InferenceResultsShapefile.create(ir_file.name)
        self.assertIsInstance(ir_file_created, InferenceResultsShapefile)
        self.assertEqual(ir_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file_created.architecture, 'resnet18')
        self.assertEqual(ir_file_created.layers, ['green', 'nir', 'osm-water'])
        self.assertEqual(ir_file_created.epoch, 5)
        self.assertEqual(ir_file_created.ratio, 5.0)
        self.assertEqual(ir_file_created.tile_size, 300)
        self.assertFalse(ir_file_created.best)

        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['red'], epoch=5, ratio=5.0, tile_size=300, best=True)
        ir_file_created = InferenceResultsShapefile.create(ir_file.name)
        self.assertIsInstance(ir_file_created, InferenceResultsShapefile)
        self.assertEqual(ir_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file_created.architecture, 'resnet18')
        self.assertEqual(ir_file_created.layers, ['red'])
        self.assertEqual(ir_file_created.epoch, 5)
        self.assertEqual(ir_file_created.ratio, 5.0)
        self.assertEqual(ir_file_created.tile_size, 300)
        self.assertTrue(ir_file_created.best)

        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['blue', 'osm-boundary'], epoch=5, ratio=5.0, tile_size=300, best=True)
        ir_file_created = InferenceResultsShapefile.create(ir_file.name)
        self.assertIsInstance(ir_file_created, InferenceResultsShapefile)
        self.assertEqual(ir_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file_created.architecture, 'resnet18')
        self.assertEqual(ir_file_created.layers, ['blue', 'osm-boundary'])
        self.assertEqual(ir_file_created.epoch, 5)
        self.assertEqual(ir_file_created.ratio, 5.0)
        self.assertEqual(ir_file_created.tile_size, 300)
        self.assertTrue(ir_file_created.best)

        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['elevation', 'slope'], epoch=5, ratio=5.0, tile_size=300, best=True)
        ir_file_created = InferenceResultsShapefile.create(ir_file.name)
        self.assertIsInstance(ir_file_created, InferenceResultsShapefile)
        self.assertEqual(ir_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(ir_file_created.architecture, 'resnet18')
        self.assertEqual(ir_file_created.layers, ['elevation', 'slope'])
        self.assertEqual(ir_file_created.epoch, 5)
        self.assertEqual(ir_file_created.ratio, 5.0)
        self.assertEqual(ir_file_created.tile_size, 300)
        self.assertTrue(ir_file_created.best)

    def test_find_files(self):
        dir_1 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0', 'green_nir_osm-water_epoch5_shapefile')
        os.makedirs(dir_1)
        for file in [
            "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.shp",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_1, file))

        dir_2 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire", 'resnet18', 'r5.0', 'green_nir_osm-water_epoch10_shapefile')
        os.makedirs(dir_2)
        for file in [
            "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.shp",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_2, file))

        dir_3 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire", 'resnet50', 'r5.0', 'green_epoch5_shapefile')
        os.makedirs(dir_3)
        for file in [
            "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.shp",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_3, file))
        
        dir_4 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire", 'resnet50', 'r10.0', 'elevation_nir_slope_epoch5_best_shapefile')
        os.makedirs(dir_4)
        for file in [
            "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5_best.shp",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_4, file))

        self.assertEqual(InferenceResultsShapefile.find_files(), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.shp"),
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.shp"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.shp")
        ]))

        self.assertEqual(InferenceResultsShapefile.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"]), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.shp"),
        ]))

        self.assertEqual(InferenceResultsShapefile.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['osm-water', 'nir', 'green']), sorted([
            os.path.join(dir_1, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.shp")
        ]))

        self.assertEqual(InferenceResultsShapefile.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['elevation', 'nir', 'slope']), sorted([]))

        self.assertEqual(InferenceResultsShapefile.find_files(regions=['Uganda', 'rwanda', "Cote D'Ivoire"], layers=['green']), sorted([]))

        self.assertEqual(InferenceResultsShapefile.find_files(regions=[ "Cote D'Ivoire"]), sorted([
            os.path.join(dir_2, "Cote D'Ivoire_resnet18_r5.0_ts300_green_nir_osm-water_epoch10.shp"),
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.shp")
        ]))

        self.assertEqual(InferenceResultsShapefile.find_files(architecture='resnet50'), sorted([
            os.path.join(dir_3, "Cote D'Ivoire_resnet50_r5.0_ts300_green_epoch5.shp")
        ]))

        self.assertEqual(InferenceResultsShapefile.find_files(ratio=10, best=True), sorted([
            os.path.join(dir_4, "Cote D'Ivoire_resnet50_r10.0_ts300_elevation_nir_slope_epoch5_best.shp")
        ]))

    def test_tarfile_name(self):
        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertEqual(ir_file.tarfile_name, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar")

        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertEqual(ir_file.tarfile_name, "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5_best.tar")
    
    def test_tarfile_archive_path(self):
        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertEqual(ir_file.tarfile_archive_path, os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0',
                         "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar"))

        ir_file = InferenceResultsShapefile(regions=['Rwanda', 'Uganda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=True)
        self.assertEqual(ir_file.tarfile_archive_path, os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda_Uganda", 'resnet18', 'r5.0',
                                                               "Cote D'Ivoire_Rwanda_Uganda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5_best.tar"))
    
    def test_create_tar_file(self):
        dir_1 = os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda", 'resnet18', 'r5.0', 'green_nir_osm-water_epoch5_shapefile')
        os.makedirs(dir_1)
        for file in [
            "Cote D'Ivoire_Rwanda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.shp",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_1, file))
        
        file = InferenceResultsShapefile(regions=['Rwanda', "Cote D'Ivoire"], architecture='resnet18',
                                           layers=['osm-water', 'nir', 'green'], epoch=5, ratio=5.0, tile_size=300, best=False)
        self.assertFalse(os.path.exists(os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda", 'resnet18', 'r5.0',
                                                      "Cote D'Ivoire_Rwanda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar")))
        file.create_tar_file()
        self.assertTrue(os.path.exists(os.path.join(self.TEST_DATA_DIR, 'inference_results', "Cote D'Ivoire_Rwanda", 'resnet18', 'r5.0',
                                                      "Cote D'Ivoire_Rwanda_resnet18_r5.0_ts300_green_nir_osm-water_epoch5.tar")))


class TestTrainSplit(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseDatasetSplit._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'train_validate_splits')

    def test_init(self):
        dataset_file = TrainSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        self.assertIsNotNone(dataset_file)
        self.assertEqual(dataset_file.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(dataset_file.ratio, 70)
        self.assertEqual(dataset_file.tile_size, 300)
    
    def test_name(self):
        dataset_file = TrainSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        self.assertEqual(dataset_file.name, "train_Cote D'Ivoire_Rwanda_Uganda_70_ts300.csv")
    
    def test_archive_path(self):
        dataset_file = TrainSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        self.assertEqual(dataset_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'train_validate_splits', "train_Cote D'Ivoire_Rwanda_Uganda_70_ts300.csv"))
    
    def test_create(self):
        dataset_file = TrainSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        dataset_file_created = TrainSplit.create(dataset_file.name)
        self.assertIsInstance(dataset_file_created, TrainSplit)
        self.assertEqual(dataset_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(dataset_file_created.ratio, 70)
        self.assertEqual(dataset_file_created.tile_size, 300)

        dataset_file_created = TrainSplit.create(dataset_file.archive_path)
        self.assertIsInstance(dataset_file_created, TrainSplit)
        self.assertEqual(dataset_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(dataset_file_created.ratio, 70)
        self.assertEqual(dataset_file_created.tile_size, 300)

    def test_find_files(self):
        dir_1 = os.path.join(self.TEST_DATA_DIR, 'train_validate_splits')
        os.makedirs(dir_1)
        for file in [
            "train_Cote D'Ivoire_Rwanda_Uganda_70_ts300.csv",
            "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts300.csv",
            "train_Cote D'Ivoire_Rwanda_Uganda_70_ts500.csv",
            "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts500.csv",
            "train_Uganda_70_ts500.csv",
            "validate_Uganda_30_ts500.csv",
            "train_Uganda_60_ts500.csv",
            "validate_Uganda_40_ts500.csv",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_1, file))


        self.assertEqual(TrainSplit.find_files(), sorted([
            os.path.join(dir_1, "train_Cote D'Ivoire_Rwanda_Uganda_70_ts300.csv"),
            os.path.join(dir_1, "train_Cote D'Ivoire_Rwanda_Uganda_70_ts500.csv"),
            os.path.join(dir_1, "train_Uganda_70_ts500.csv"),
            os.path.join(dir_1, "train_Uganda_60_ts500.csv")
        ]))

        self.assertEqual(TrainSplit.find_files(regions=["Cote D'Ivoire", 'Rwanda', 'Uganda']), sorted([
            os.path.join(dir_1, "train_Cote D'Ivoire_Rwanda_Uganda_70_ts300.csv"),
            os.path.join(dir_1, "train_Cote D'Ivoire_Rwanda_Uganda_70_ts500.csv")
        ]))

        self.assertEqual(TrainSplit.find_files(regions=["Cote D'Ivoire", 'Rwanda', 'Uganda'], tile_size=500), sorted([
            os.path.join(dir_1, "train_Cote D'Ivoire_Rwanda_Uganda_70_ts500.csv")
        ]))

        self.assertEqual(TrainSplit.find_files(regions=['Uganda'], tile_size=500, ratio=60), sorted([
            os.path.join(dir_1, "train_Uganda_60_ts500.csv")
        ]))


class TestValidateSplit(TestFileTypes):
    @classmethod
    def setUpClass(cls):
        TestFileTypes.setUpClass()
        _BaseDatasetSplit._ROOT_DATA_DIR = os.path.join(cls.TEST_DATA_DIR, 'train_validate_splits')

    def test_init(self):
        dataset_file = ValidateSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        self.assertIsNotNone(dataset_file)
        self.assertEqual(dataset_file.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(dataset_file.ratio, 70)
        self.assertEqual(dataset_file.tile_size, 300)
    
    def test_name(self):
        dataset_file = ValidateSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        self.assertEqual(dataset_file.name, "validate_Cote D'Ivoire_Rwanda_Uganda_70_ts300.csv")
    
    def test_archive_path(self):
        dataset_file = ValidateSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        self.assertEqual(dataset_file.archive_path, os.path.join(self.TEST_DATA_DIR, 'train_validate_splits', "validate_Cote D'Ivoire_Rwanda_Uganda_70_ts300.csv"))
    
    def test_create(self):
        dataset_file = ValidateSplit(regions=["Cote D'Ivoire", 'Uganda', 'Rwanda'], ratio=70, tile_size=300)
        dataset_file_created = ValidateSplit.create(dataset_file.name)
        self.assertIsInstance(dataset_file_created, ValidateSplit)
        self.assertEqual(dataset_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(dataset_file_created.ratio, 70)
        self.assertEqual(dataset_file_created.tile_size, 300)

        dataset_file_created = ValidateSplit.create(dataset_file.archive_path)
        self.assertIsInstance(dataset_file_created, ValidateSplit)
        self.assertEqual(dataset_file_created.regions, ["Cote D'Ivoire", 'Rwanda', 'Uganda'])
        self.assertEqual(dataset_file_created.ratio, 70)
        self.assertEqual(dataset_file_created.tile_size, 300)

    def test_find_files(self):
        dir_1 = os.path.join(self.TEST_DATA_DIR, 'train_validate_splits')
        os.makedirs(dir_1)
        for file in [
            "train_Cote D'Ivoire_Rwanda_Uganda_70_ts300.csv",
            "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts300.csv",
            "train_Cote D'Ivoire_Rwanda_Uganda_70_ts500.csv",
            "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts500.csv",
            "train_Uganda_70_ts500.csv",
            "validate_Uganda_30_ts500.csv",
            "train_Uganda_60_ts500.csv",
            "validate_Uganda_40_ts500.csv",
            'off_nominal.txt'
        ]:
            self.create_blank_file(os.path.join(dir_1, file))


        self.assertEqual(ValidateSplit.find_files(), sorted([
            os.path.join(dir_1, "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts300.csv"),
            os.path.join(dir_1, "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts500.csv"),
            os.path.join(dir_1, "validate_Uganda_30_ts500.csv"),
            os.path.join(dir_1, "validate_Uganda_40_ts500.csv")
        ]))

        self.assertEqual(ValidateSplit.find_files(regions=["Cote D'Ivoire", 'Rwanda', 'Uganda']), sorted([
            os.path.join(dir_1, "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts300.csv"),
            os.path.join(dir_1, "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts500.csv")
        ]))

        self.assertEqual(ValidateSplit.find_files(regions=["Cote D'Ivoire", 'Rwanda', 'Uganda'], tile_size=500), sorted([
            os.path.join(dir_1, "validate_Cote D'Ivoire_Rwanda_Uganda_30_ts500.csv")
        ]))

        self.assertEqual(ValidateSplit.find_files(regions=['Uganda'], tile_size=500, ratio=40), sorted([
            os.path.join(dir_1, "validate_Uganda_40_ts500.csv")
        ]))
