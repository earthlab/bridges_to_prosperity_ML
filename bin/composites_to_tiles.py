import argparse
import multiprocessing as mp
import json
import logging
import os
import time
import glob
import tqdm
from argparse import Namespace
from multiprocessing import Pool

from src.utilities.imaging import scale

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
            tile_basename = 'dem' + namespace.composite_path.split('.')[0][-5:] + str(i) + '_' + str(j) + '.tif'
            tiff_filename = os.path.join(namespace.output_dir, tile_basename)
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
                                 'multiband_scaled_corrected' + os.path.basename(composite_path).split('.')[0][-3:] + '.tiff')
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
            args.append(
                Namespace(
                    output_scaled=output_scaled, 
                    output_dir=output_dir, 
                    composite_path=composite_path,
                    tile_start=batch[0], 
                    tile_stop=batch[1], 
                    geom_lookup_outfile=geom_lookup_outfile
                )
            )

        # Block until all processes are done
        for result in p.map(make_tiff_files_task, args):
            pass 
            # print(result)

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

def composite_to_tiles(s2_dir, output_dir, cores, geom_lookup_path):
    mp.set_start_method('spawn')
    files = glob.glob(os.path.join(s2_dir, "**/*multiband_cld_NAN_median_corrected*.tiff"),recursive=True)
    pbar = tqdm.tqdm(files)
    for composite_path in pbar:
        prts = os.path.basename(composite_path).split('.')
        abbrev = prts[0][-5:]
        pbar.set_description(f"Processing {abbrev}")
        pbar.refresh()
        this_outdir = os.path.join(output_dir, abbrev)
        if not os.path.isdir(this_outdir):
            os.makedirs(this_outdir)
        create_tiles(
            composite_path,
            this_outdir,
            cores,
            geom_lookup_path
        )

if __name__ == '__main__':
    
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)
            ), 
            '..'
        )
    )
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--region', 
        '-r', 
        type=str, 
        help='Region for tiling'
    )
    parser.add_argument(
        '--in_dir', 
        '-i', 
        type=str, 
        required=False, 
        default=os.path.join(base_dir, 
        "data", 
        "sentinel2"), 
        help='path to inpt directory where s2 path is'
    )
    parser.add_argument(
        '--out_dir', 
        '-o', 
        type=str, 
        required=False, 
        default=os.path.join(base_dir, "data", "tiles"),
         help='Path to directory where output files will be written'
    )
    parser.add_argument(
        '--cores', 
        '-c', 
        type=int, 
        default=mp.cpu_count() - 1,
        required=False, 
        help='Number of cores to use in parallel for tiling'
    )
    parser.add_argument(
        '--geom_lookup_path', 
        '-g', 
        type=str, 
        required=False,
        default=None,
        help='If specified the a tiff geometry lookup file will be written here'
    )
    
    args = parser.parse_args()
    # make paths to the region 
    s2_dir = os.path.join(args.in_dir, args.region)
    out_dir = os.path.join(args.out_dir, args.region)
    # call function 
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    composite_to_tiles(
        s2_dir,
        out_dir, 
        args.cores,
        args.geom_lookup_path
    )
