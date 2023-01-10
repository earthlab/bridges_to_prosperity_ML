import argparse
import multiprocessing
import json
import logging
import os
import time
from argparse import Namespace
from multiprocessing import Pool

from utilities.imaging import scale

import numpy as np
import rasterio
from osgeo import gdal


def make_tiff_files_task(namespace: Namespace):
    dem = gdal.Open(namespace.output_scaled)
    gt = dem.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]

    xlen = res * dem.RasterXSize
    ylen = res * dem.RasterYSize
    div = 366
    xsize = xlen / div
    ysize = ylen / div
    xsteps = [xmin + xsize * i for i in range(div + 1)]
    ysteps = [ymax - ysize * i for i in range(div + 1)]

    del gt, xmin, ymax, res, xlen, ylen
    geom_lookup = {}
    filenames = []
    for i in range(namespace.tile_start, namespace.tile_stop):
        for j in range(div):
            xmin = xsteps[i]
            xmax = xsteps[i + 1]
            ymax = ysteps[j]
            ymin = ysteps[j + 1]
            tiff_filename = os.path.join(namespace.output_dir, 'dem' + namespace.composite_path.split('.')[0][-3:] +
                                         str(i) + '_' + str(j) + '.tif')
            filenames.append(tiff_filename)
            gdal.Warp(tiff_filename, dem, outputBounds=(xmin, ymin, xmax, ymax), dstNodata=-999)
            if namespace.geom_lookup_outfile is not None:
                geom_lookup[os.path.basename(tiff_filename)] = ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                                                                (xmin, ymin))

            del xmin, xmax, ymax, ymin

    if namespace.geom_lookup_outfile is not None:
        with open(namespace.geom_lookup_outfile, 'w+') as f:
            json.dump(geom_lookup, f)


def create_tiles(composite_path: str, output_dir: str, cores: int, geometry_lookup_path: str = None):
    """

    """
    logging.info('Starting tiff file creation')
    t1 = time.time()
    ds = gdal.Open(composite_path)
    r = scale(ds.GetRasterBand(3).ReadAsArray()).astype('uint8')
    g = scale(ds.GetRasterBand(2).ReadAsArray()).astype('uint8')
    b = scale(ds.GetRasterBand(1).ReadAsArray()).astype('uint8')
    ds = None
    del ds

    dss = rasterio.open(composite_path)

    output_scaled = os.path.join(output_dir,
                                 'multiband_scaled_corrected' + os.path.basename(composite_path).split('.')[0][
                                                                -3:] + '.tiff')
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

    del r, g, b, true

    div = 366
    batch_space = np.linspace(0, div, cores + 1)
    batches = []
    for i in range(1, len(batch_space)):
        batches.append((int(batch_space[i-1]), int(batch_space[i])))

    output_parallel_files = os.path.join(output_dir, 'tiff_parallel_output')
    os.makedirs(output_parallel_files, exist_ok=True)

    with Pool(len(batches)) as p:
        args = []
        for i, batch in enumerate(batches):
            geom_lookup_outfile = None if geometry_lookup_path is None else os.path.join(output_parallel_files,
                                                                                         f'geom_lookup_{i}.json')
            args.append(Namespace(output_scaled=output_scaled, output_dir=output_dir, composite_path=composite_path,
                                  tile_start=batch[0], tile_stop=batch[1], geom_lookup_outfile=geom_lookup_outfile))

        # Block until all processes are done
        for result in p.map(make_tiff_files_task, args):
            print(result)

    # Combine the parallel output
    if geometry_lookup_path is not None:
        geom_lookup = {}
        for file in os.listdir(output_parallel_files):
            file_path = os.path.join(output_parallel_files, file)
            if file.startswith('geom_lookup'):
                with open(file_path, 'r') as f:
                    geoms = json.load(f)
                    for geom in geoms:
                        geom_lookup[geom] = geoms[geom]

        with open(geometry_lookup_path, 'w+') as f:
            json.dump({'geom_lookup': geom_lookup}, f)

    logging.info(f'Wrote tiff files in {time.time() - t1}s')
    print(f'Wrote tiff files to {output_dir}')

    # TODO: Do some parallel file cleanup


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument('--composite_path', '-f', type=str, required=True, help='Path to the color corrected composite'
                                                                                ' s2 tile for the region')
    parser.add_argument('--out_dir', '-o', type=str, required=True, help='Path to directory where output files will be'
                                                                         ' written')

    # Optional args
    parser.add_argument('--geom_lookup_path', '-g', type=str, required=True,
                        help='If specified the a tiff geometry lookup file will be written here')
    parser.add_argument('--cores', '-c', type=int, required=False, help='Number of cores to use in parallel for tiling')

    args = parser.parse_args()

    create_tiles(composite_path=args.composite_path, output_dir=args.out_dir, cores=args.cores)
