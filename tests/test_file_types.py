from unittest import TestCase
from file_types import OpticalComposite


class TestFileTypes(TestCase):
    def setUp(self) -> None:
        pass
    def tearDown(self) -> None:
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



