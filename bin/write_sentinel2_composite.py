import os
import re
import shutil
from argparse import ArgumentParser
from datetime import datetime
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window

from src.utilities.imaging import scale


def decode_coordinate_grid(dirname: str):
    directory_pattern = r'(?P<utm_code>\d+)(?P<latitude_band>\S{1})(?P<square>\S{2})'
    match = re.match(directory_pattern, dirname)

    return match


def get_median_slices(indir, outdir, number_of_slices):
    band_length = 10980
    slice_width = band_length / number_of_slices
    slice_bounds = [int(i) for i in np.arange(0, band_length + slice_width, slice_width)]

    coordinate_match = decode_coordinate_grid(os.path.basename(indir))  # Should be a 5 character string like 36LUK
    if coordinate_match:
        coords = coordinate_match.groupdict()
        utm_code = coords['utm_code']
        latitude_band = coords['latitude_band']
        square = coords['square']

    else:
        raise ValueError('Input directories should be named after the military grid, i.e. 36LUK')

    for i in range(len(slice_bounds) - 1):
        left_bound = slice_bounds[i]
        right_bound = slice_bounds[i + 1]

        for band in ['B02', 'B03', 'B04']:

            # This list will hold the slice from each day's data that is available
            dates_for_slice = []
            transform = None
            crs = None

            for dated_directory in os.listdir(indir):
                date = datetime.strptime(dated_directory, '%Y%m%d')

                # The same cloud file is used for every band so read it in here
                cloud_path = os.path.join(indir, dated_directory,
                                          f'tiles_{utm_code}_{latitude_band}_{square}_{date.year}_'
                                          f'{date.month}_{date.day}_0_qi_MSK_CLOUDS_B00.gml')
                if not os.path.exists(cloud_path):
                    raise FileNotFoundError(f'Could not find cloud file at {cloud_path}')

                cloud_file = None
                if os.path.getsize(cloud_path) > 500:
                    cloud_file = gpd.read_file(cloud_path)

                base_file_name = f'tiles_{utm_code}_{latitude_band}_{square}_{date.year}_{date.month}_{date.day}'

                image_path = os.path.join(indir, dated_directory, base_file_name + f'_{band}.jp2')
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f'Could not find image file at {image_path}')

                image = rasterio.open(image_path, driver='JP2OpenJPEG')
                image_read = image.read(1)

                if crs is None:
                    crs = image.crs
                if transform is None:
                    transform = image.transform

                # Read in the file... going to try and do this so the file is only read in once
                image_slice = image.read(1, window=Window.from_slices(slice(left_bound, right_bound),
                                                                      slice(0, band_length)))

                try:
                    if cloud_file is not None:
                        cloud_file.crs = ({'init': str(image_slice.crs)})

                        # convert the cloud mask data to a raster that has the same shape and transformation as the
                        # image raster data
                        cloud_image = features.rasterize(
                            ((g['geometry'], 1) for v, g in cloud_file.iterrows()),
                            out_shape=image_read.shape,
                            transform=image.transform, all_touched=True)
                        cloud_channels = np.where(cloud_image == 0, 1, 0)

                        # find the indices in the cloud mask raster data where the red channel is 0, the green channel
                        # is 1, and the blue channel is 0
                    else:
                        cloud_image = np.empty_like(image_read.shape)
                        cloud_channels = np.where(cloud_image == 0, 1, 1)
                except:
                    cloud_image = np.empty_like(image_read.shape)
                    cloud_channels = np.where(cloud_image == 0, 1, 1)

                # select the image data from the current slice and index
                image_slice = image_slice * cloud_channels[left_bound: right_bound, 0:10980]

                # filter out cloud pixels in the current slice by finding pixels that have a brightness of 0 or whose
                # brightness value is too high (> 4000) and set them to np.nan
                mask = np.where(np.logical_or(image_slice == 0, image_slice > 4000))
                image_slice[mask] = np.nan

                dates_for_slice.append(image_slice)

            # Write the file for each band and slice, combining the data for each avaiable day of data
            shape = dates_for_slice[0].shape
            y2 = np.vstack(dates_for_slice)

            # Releasing memory
            dates_for_slice = []
            del dates_for_slice

            z2 = np.nanmedian(y2, axis=0, overwrite_input=True)

            # Releasing memory
            y2 = []
            del y2

            z2 = np.uint16(z2.reshape(shape))
            outpath = os.path.join(outdir, f'{os.path.basename(indir)}', band, f'{left_bound}_{right_bound}.tiff')
            true_color = rasterio.open(outpath, 'w', driver='Gtiff', width=shape[1], height=shape[0], count=1, crs=crs,
                                       transform=transform, dtype=z2.dtype)

            true_color.write(z2, 1)  # green
            true_color.close()

            z2 = []
            del z2


def write_out(indir, outdir, region_name: str):
    coordinate = os.path.basename(indir)

    final_truecolor = []  # used to catch each of the bands in order to write in disk the overall Truecolo picture

    for band_dir in os.listdir(indir):
        over_bnd = {}
        transform = None
        crs = None
        dtype = None
        for cloud_cleaned_file in os.listdir(os.path.join(indir, band_dir)):
            file = rasterio.open(os.path.join(indir, band_dir, cloud_cleaned_file))
            over_bnd[cloud_cleaned_file] = file

            if transform is None:
                transform = file.transform
            if crs is None:
                crs = file.crs
            if dtype is None:
                dtype = file.dtypes[0]

        out_path = os.path.join(outdir, f'Multiband_median_corrected_{coordinate}_{band_dir}.tiff')
        with rasterio.open(out_path, 'w', driver='GTiff', width=10980, height=10980, count=1, crs=crs,
                           transform=transform, dtype=dtype) as dst:
            for file_name, file in over_bnd.items():
                left_bound = file_name.split('_')[0]
                right_bound = file_name.split('_')[1].replace('.tiff', '')
                dst.write(file.read(1), window=Window.from_slices(slice(left_bound, right_bound), slice(0, 10980)),
                          indexes=1)

        file2 = rasterio.open('Multiband_median_corrected' + coordinate + band_dir + '.tiff')

        final_truecolor.append(file2)

        file2.close()

    out_path = os.path.join(outdir, f'{region_name}_multiband_cld_NAN_median_corrected_{coordinate}.tiff')
    true = rasterio.open(out_path, 'w', driver='Gtiff', width=final_truecolor[0].width,
                         height=final_truecolor[0].height, count=3, crs=final_truecolor[0].crs,
                         transform=final_truecolor[0].transform, dtype=final_truecolor[0].dtypes[0])

    true.write(final_truecolor[0].read(1), 3)  # green
    true.write(final_truecolor[1].read(1), 2)
    true.write(final_truecolor[2].read(1), 1)

    true.close()

    ds = gdal.Open(out_path)
    r = ds.GetRasterBand(3).ReadAsArray()
    g = ds.GetRasterBand(2).ReadAsArray()
    b = ds.GetRasterBand(1).ReadAsArray()

    ds = None
    del ds

    r = scale(r).astype('uint8')
    g = scale(g).astype('uint8')
    b = scale(b).astype('uint8')

    dss = rasterio.open(out_path)

    output_scaled = os.path.join(outdir, f'multiband_scaled_corrected_{coordinate}.tiff')

    true = rasterio.open(str(output_scaled), 'w', driver='Gtiff',
                         width=dss.width, height=dss.height,
                         count=3,
                         crs=dss.crs,
                         transform=dss.transform,
                         dtype='uint8'
                         )

    true.write(r, 3)
    true.write(g, 2)
    true.write(b, 1)

    true.close()

    del r, g, b, dss, true


def create_composite(input_dir, out_dir, region_name, slices):
    slice_outdir = tempfile.mkdtemp(prefix='b2p')

    slices = 1 if slices is None else slices
    get_median_slices(input_dir, slice_outdir, slices)
    write_out(slice_outdir, out_dir, region_name)
    shutil.rmtree(slice_outdir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True, help="Path to the input directory contianing each"
                                                                           " date's sentinel2 files")
    parser.add_argument('--out_dir', '-o', type=str, required=True, help='Path to where composites will be written')
    parser.add_argument('--region_name', '-r', type=str, required=True, help='Name of the composite region')
    parser.add_argument('--slices', '-s', type=int, required=False, default=1,
                        help='The number of slices to break the sentinel2 tiles up into before cloud-correcting. '
                             'Default is 1')
    args = parser.parse_args()
    create_composite(args.input_dir, args.out_dir, args.region_name, args.slices)
