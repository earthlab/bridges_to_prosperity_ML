#!/usr/bin/env python

import logging
import time

import geopandas as gpd
import argparse
from shapely.geometry import Polygon
import rasterio
import gdal
import rioxarray
from contextlib import redirect_stdout, redirect_stderr, ExitStack
from fastai.vision import *
from subprocess import Popen
import random
import multiprocessing
import boto3


def scale(x):
    return (x - np.nanmin(x)) * (1 / (np.nanmax(x) - np.nanmin(x)) * 255)


def progress_tiff_list(filename: str):
    outlist = []
    with open(filename, 'r') as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if first:
                first = False
                continue
            outlist.append(row[1])

    return outlist


@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


def progress_to_list(filename: str, tiff_dir: str):
    outlist = []
    with open(filename, 'r') as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if first:
                first = False
                continue
            polygon = row[2:-3]
            coords = []
            for j in polygon:
                coords.append(tuple([int(p) for p in j.strip('POLYGON').strip('(').strip(' (').strip(')').split(' ')]))

            outlist.append((os.path.join(tiff_dir, row[1]), Polygon(coords), row[-3], int(row[-2]), float(row[-1])))
    return outlist


def batch_list(input_list, batches):
    f = len(input_list) // batches
    for i in range(0, len(input_list), f):
        yield input_list[i:i + f]


def generate_jobfile_name(parallel_dir: str):
    ls = os.listdir(parallel_dir)

    if not ls:
        return f'{random.randint(1, 10000)}.json'

    job_file = ls[0]
    while job_file in ls:
        job_file = f'{random.randint(1, 10000)}.json'

    return job_file


HOME_DIR = os.path.join('/home', 'jovyan')
N_CORES = multiprocessing.cpu_count()


def tiling(input_rstr: str, name: str, home_dir: str, progress_file: str = None,  cores: int = N_CORES - 1):
    logging.info('Starting function workflow')
    start = time.time()
    ds = gdal.Open(input_rstr)
    r = ds.GetRasterBand(3).ReadAsArray()
    g = ds.GetRasterBand(2).ReadAsArray()
    b = ds.GetRasterBand(1).ReadAsArray()

    ds = None
    del ds

    r = scale(r)
    g = scale(g)
    b = scale(b)

    r = r.astype('uint8')
    g = g.astype('uint8')
    b = b.astype('uint8')

    dss = rasterio.open(input_rstr)

    t1 = time.time()
    logging.info('Writing tiff')
    output_scaled = os.path.join(
        home_dir, 'multiband_scaled_corrected' + os.path.basename(input_rstr).split('.')[0][-3:] + '.tiff')
    true = rasterio.open(str(output_scaled), 'w', driver='Gtiff',
                         width=dss.width, height=dss.height,
                         count=3,
                         crs=dss.crs,
                         transform=dss.transform,
                         dtype='uint8'
                         )
    logging.info(f'Wrote tiff {time.time() - t1}s')
    logging.info('Wrote tiff')
    true.write(r, 3)
    true.write(g, 2)
    true.write(b, 1)
    true.close()

    # s3.Bucket(bucket_name).upload_file(output_scaled, name + '_' + output_scaled)

    del r, g, b, dss, true

    dem = gdal.Open(output_scaled)
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

    aoi_name = 'folder_' + input_rstr.split('.')[0][-3:]
    tilling_dir = os.path.join(home_dir, f'Tilling_tiff_{aoi_name}_{name}')
    os.makedirs(tilling_dir, exist_ok=True)

    for i in range(div):
        for j in range(div):
            xmin = xsteps[i]
            xmax = xsteps[i + 1]
            ymax = ysteps[j]
            ymin = ysteps[j + 1]
            gdal.Warp(os.path.join(tilling_dir, 'dem' + input_rstr.split('.')[0][-3:] + str(i) + str(j) + '.tif'), dem,
                      outputBounds=(xmin, ymin, xmax, ymax), dstNodata=-999)

            del xmin, xmax, ymax, ymin

    del dem, div, xsteps, ysteps

    shape_dir = os.path.join(home_dir, f'Tilling_shp_{aoi_name}_{name}')
    os.makedirs(shape_dir, exist_ok=True)

    for i in os.listdir(tilling_dir)[0:4]:
        rds = rioxarray.open_rasterio(os.path.join(tilling_dir, i))
        gdf_crs = rds.rio.crs
    del rds

    # Create a temp working directory for parallelization i/o
    temp_dir = tempfile.mkdtemp(prefix='b2p')

    input_parallel_files = os.path.join(temp_dir, 'input_parallel')
    os.makedirs(input_parallel_files, exist_ok=True)

    # Make a unique directory in parallel directory
    output_parallel_files = os.path.join(temp_dir, 'output_parallel', f'{int(time.time())}')
    os.makedirs(output_parallel_files, exist_ok=True)

    logging.info('Starting tiling and shape file creation')

    tilling_dir_list = [os.path.join(tilling_dir, file) for file in os.listdir(tilling_dir)]
    logging.info(f'Length of tilling dir: {len(os.listdir(tilling_dir))}')
    batches = list(batch_list(tilling_dir_list, cores))
    t1 = time.time()
    for batch in batches:
        job_file = generate_jobfile_name(input_parallel_files)
        job_path = os.path.join(input_parallel_files, job_file)
        with open(job_path, 'w+') as f:
            json.dump({'tiling_files': batch}, f)
        outpath = os.path.join(output_parallel_files, generate_jobfile_name(output_parallel_files))
        Popen([sys.executable, os.path.join(home_dir, 'make_tile_tiff_files.py'), '--input_file', job_path,
               '--shape_dir', shape_dir, '--crs', gdf_crs.to_string(), '--outpath', outpath])
    # Wait until all the job files have been written out
    while len(os.listdir(shape_dir)) < len(tilling_dir_list) * 5:
        print('SLEEPING')
        time.sleep(3 * 60)

    file_bounds = []
    for file in os.listdir(output_parallel_files):
        with open(os.path.join(output_parallel_files, file), 'r') as f:
            f_data = json.load(f)
            for bound in f_data['file_bounds']:
                file_bounds.append((bound[0], Polygon(bound[1])))

    logging.info(f'Made tiff and shape files in {time.time() - t1}s using {len(batches)} cores')
    logging.info('Finished making tile tiff and shape files')

    names, bounds = zip(*file_bounds)
    geoseries = gpd.GeoSeries(data=bounds)
    rdf = gpd.GeoDataFrame(data=names, columns=['name_shp'], geometry=geoseries, crs=gdf_crs)

    shape_tiles = os.path.join(home_dir, 'shp_tiles_' + str(input_rstr.split('.')[0][-3:]))
    rdf.to_file(shape_tiles)
    logging.info('Finished making tile shp files')

    for obj in os.listdir(shape_tiles):
        s3.Bucket(bucket_name).upload_file(os.path.join(shape_tiles, obj), name + '_' + obj)

    del rdf
    logging.info('Finished uploading tile shp files to the S3 bucket')

    logging.info('Starting Inference')
    test_gdf = gpd.read_file(
        os.path.join(home_dir, 'shp_tiles_' + str(input_rstr.split('.')[0][-3:]),
                     'shp_tiles_' + str(input_rstr.split('.')[0][-3:]) + '.shp')
    )
    np.random.seed(42)

    # Create a new unique output directory to store subprocess output
    output_parallel_files = os.path.join(temp_dir, 'output_parallel', f'{int(time.time())}')
    os.makedirs(output_parallel_files, exist_ok=True)

    t1 = time.time()

    if progress_file is not None:
        completed_tiles = progress_tiff_list(progress_file)
        uncompleted_tiling_dir_list = [i for i in tilling_dir_list if os.path.basename(i) not in completed_tiles]
        batches = list(batch_list(uncompleted_tiling_dir_list, cores))

    count = 0
    for batch in batches:
        count += len(batch)
        job_file = generate_jobfile_name(input_parallel_files)
        job_path = os.path.join(input_parallel_files, job_file)
        with open(job_path, 'w+') as f:
            json.dump({'tiling_files': batch}, f)

        outpath = os.path.join(output_parallel_files, generate_jobfile_name(output_parallel_files))
        args = [sys.executable, os.path.join(home_dir, 'inference.py'), '--input_file', job_path,
               '--model_path', os.path.join(home_dir, 'learn_res50.pkl'),
               '--shape_path', os.path.join(shape_tiles, 'shp_tiles_' + str(input_rstr.split('.')[0][-3:]) + '.shp'),
               '--tiling_dir', tilling_dir, '--input_tiff', input_rstr, '--location_name', name,
               '--outpath', outpath]

        Popen(args)

    logging.info(f"Performing inference on {count} files")

    # There are three files (output, progress csv, and logfile) being written for each subprocess
    while len(os.listdir(output_parallel_files)) < len(batches) * 3:
        time.sleep(15 * 60)

    logging.info(f'Finished creating geoms list {time.time() - t1}s')
    logging.info(f'Finished creating geoms list')

    geoms = []
    for file in os.listdir(output_parallel_files):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(output_parallel_files, file), 'r') as f:
            file_data = json.load(f)
            for geom in file_data['geoms']:
                geometry = Polygon(geom[1])
                geoms.append((geom[0], geometry, geom[2], geom[3], geom[4]))

    if progress_file is not None:
        geoms += progress_to_list(progress_file, tiff_dir=tilling_dir)

    t1 = time.time()
    file_names, geom_vals, pred_labels, pred_values, fl_values = zip(*geoms)

    geoseries_data = gpd.GeoSeries(data=geom_vals)

    test_gdff = gpd.GeoDataFrame(data=file_names, columns=['name_shp'], geometry=geoseries_data,
                                 crs=test_gdf.crs)

    test_gdff['label'] = pred_labels
    test_gdff['value'] = pred_values
    test_gdff['fl_val'] = fl_values

    logging.info(f'Finished performing inference {time.time() - t1}.s')
    logging.info('Finished performing inference')

    inference_dir = os.path.join(home_dir, 'RESNET_' + str(50) + '_Inference_' + str(input_rstr.split('.')[0][-3:]))
    test_gdff.to_file(inference_dir)
    for obj in os.listdir(inference_dir):
        s3.Bucket(bucket_name).upload_file(os.path.join(inference_dir, obj), name + '_' + obj)

    logging.info('Finished uploading inference files to the S3 bucket')

    del test_gdf, test_gdff, pred_labels, pred_values, fl_values

    shutil.rmtree(tilling_dir)
    shutil.rmtree(shape_dir)
    end = time.time()
    logging.info(f'Time of Code Execution: {end - start}s')

    return


if __name__ == '__main__':
    logdir = tempfile.mkdtemp(prefix='b2p_logs')
    logfile = os.path.join(logdir, f'{int(time.time())}.log')
    print(f'Writing log file to {logfile}')

    logging.basicConfig(filename=logfile, level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True, help='Name of the region')
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to tiff file')
    parser.add_argument('--cores', '-c', type=int, required=False, default=N_CORES - 1)
    parser.add_argument('--progress_file', '-p', type=str, required=False, help='Path to progress csv file')
    parser.add_argument('--bucket_name', '-b', type=str, required=True, help='Path to AWS s3 bucket i.e. b2p.erve')
    args = parser.parse_args()

    bucket_name = args.bucket_name
    s3 = boto3.resource('s3')

    tiling(args.file, args.name, HOME_DIR, args.progress_file, args.cores)
