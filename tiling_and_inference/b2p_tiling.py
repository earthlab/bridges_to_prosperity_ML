import argparse
import csv
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import time
from argparse import Namespace
from multiprocessing import Pool
from typing import List, Any
import subprocess as sp

import boto3
import geopandas as gpd
import numpy as np
import rasterio
from fastai.vision import ImageList, get_transforms, load_learner, imagenet_stats
from geojson import Polygon
from shapely.geometry import Polygon


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))


# Utility functions
def progress_tiff_list(filename: str) -> List[str]:
    """
    Parses tiff file names out of a csv file with inference results.
    Args:
        filename (str): Path to inference results csv file
    Returns:
        completed_tiff_files (list): List of tiff file names with inference results in input file
    """
    completed_tiff_files = []
    with open(filename, 'r') as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if first:
                first = False
                continue
            completed_tiff_files.append(row[1])

    return completed_tiff_files


def polygon_inference_file_to_list(filename: str, tiff_dir: str):
    """
    Parses and typecasts csv file with inference results so that it can be read in as a list of Python objects. This
    function is for inference result files that include the string representation of the polygon object.
    Args:
        filename (str): Path to csv file with inference results
        tiff_dir (str): Path to directory containing tiff files that are listed in the input file
    Returns:
        inference_file_to_list (list): List of inference results typecast from strings to their respective Python
         objects
    """
    inference_results = []
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
            inference_results.append((os.path.join(tiff_dir, row[1]), Polygon(coords), row[-3], int(row[-2]),
                                      float(row[-1])))
    return inference_results


def batch_list(input_list: List[Any], batches: int) -> List[List[Any]]:
    """
    Splits up an input list into a list of sub-lists. The number of sub-lists is determined by the batches argument.
    This is useful when creating batches for parallelization.
    Args:
        input_list (list): List to be broken up into sub-lists
        batches (int): Number of sub-lists to split the input_list into
    """
    f = len(input_list) // batches
    for i in range(0, len(input_list), f):
        yield input_list[i:i + f]


# Parallel task

def do_inference_task(namespace: Namespace):
    job_id = os.path.basename(namespace.output_file).split('.json')[0]
    work_dir = os.path.dirname(namespace.output_file)

    t1 = time.time()
    # pkl file path
    learn_infer = load_learner(path=os.path.dirname(namespace.model_path), file=os.path.basename(namespace.model_path))

    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    np.random.seed(42)

    # tiling dir path
    test_data = ImageList.from_folder(namespace.tiling_dir).split_none().label_empty().transform(
        tfms, size=224, tfm_y=False).databunch().normalize(imagenet_stats)
    test_data.train_dl.new(shuffle=False)
    val_dict = {1: 'yes', 0: 'no'}

    # input tiff file
    sent_indx = str(namespace.input_tiff_path.split('.')[0][-3:])
    with open(namespace.input_file, 'r') as f:
        file_data = json.load(f)
    ls_names = file_data['tiling_files']
    print(f"{time.time() - t1}s setup time")

    # unique name for logfile
    log_path = os.path.join(work_dir, f'{namespace.region_name}_{sent_indx}_Inference_Progress_{job_id}.log')
    with open(log_path, 'w+') as f:
        f.write(f'Length of tiling files for job {len(ls_names)}\n')

    # unique name for progress file
    progress_path = os.path.join(work_dir, f'{namespace.region_name}_{sent_indx}_Inference_Progress_{job_id}.csv')
    with open(progress_path, 'w+') as f:
        f.write('Index,Filename,Geometry,Predicted_Label,Predicted_Value,fl_value\n')

    with open(namespace.tiff_geom_path, 'r') as f:
        geom_lookup = json.load(f)['geom_lookup']

    geoms = []
    # tiling file paths
    for i, tiff_path in enumerate(ls_names):
        t0 = time.time()
        diff = None

        try:
            t1 = time.time()
            im = test_data.train_ds[i][0]
            prediction = learn_infer.predict(im)

            pred_val = prediction[1].data.item()
            pred_label = val_dict[pred_val]
            fl_val = prediction[2].data[pred_val].item()

            coords = geom_lookup[os.path.basename(tiff_path)]
            geoms.append((tiff_path, coords, pred_label, pred_val, fl_val))

            outline = f'{i},{tiff_path},{coords},{pred_label},{pred_val},{fl_val}\n'
            with open(progress_path, 'a') as f:
                f.write(outline)

            diff = time.time() - t1

            del im, prediction, pred_val, pred_label, fl_val
        except Exception as e:
            outline = str(e)

        # Write info to log file
        if diff is not None:
            outline += f" inference time: {diff}s"
        outline += f' Total time: {time.time() - t0}s'
        with open(log_path, 'a') as f:
            outline += '\n'
            f.write(outline)

    with open(namespace.output_file, 'w+') as f:
        json.dump({'geoms': geoms}, f, indent=1)


# Parent process

def do_inference(input_rstr: str, region_name: str, tiling_dir: str, output_dir: str, model_path: str,
                 crs: rasterio.crs.CRS, geom_lookup_path: str, progress_file: str = None, bucket_name: str = None,
                 s3=None):
    logging.info('Starting Inference')
    np.random.seed(42)

    cores = 2 if multiprocessing.cpu_count() > 2 else 1

    # Create input and output directories for parallelization i/o
    input_parallel_files = os.path.join(output_dir, 'inference_parallel_input')
    os.makedirs(input_parallel_files, exist_ok=True)

    output_parallel_files = os.path.join(output_dir, 'inference_parallel_output')
    os.makedirs(output_parallel_files, exist_ok=True)

    t1 = time.time()

    # Create batches for parallelization
    tilling_dir_list = [os.path.join(tiling_dir, i) for i in os.listdir(tiling_dir)]
    if progress_file is not None:
        completed_tiles = progress_tiff_list(progress_file)
        uncompleted_tiling_dir_list = [i for i in tilling_dir_list if os.path.basename(i) not in completed_tiles]
        batches = list(batch_list(uncompleted_tiling_dir_list, cores))
    else:
        batches = list(batch_list(tilling_dir_list, cores))

    count = 0
    args = []
    for i, batch in enumerate(batches):
        count += len(batch)
        job_path = os.path.join(input_parallel_files, f'batch_{i}_input.json')
        with open(job_path, 'w+') as f:
            json.dump({'tiling_files': batch}, f)
        out_path = os.path.join(output_parallel_files, f'batch_{i}_output.json')
        args.append(Namespace(input_file=job_path, model_path=model_path, tiling_dir=tiling_dir,
                              input_tiff_path=input_rstr, tiff_geom_path=geom_lookup_path, region_name=region_name,
                              output_file=out_path))
    logging.info(f"Performing inference on {count} files")

    with Pool(cores) as p:
        for result in p.map(do_inference_task, args):
            print(result)

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
        geoms += polygon_inference_file_to_list(progress_file, tiff_dir=tiling_dir)

    t1 = time.time()
    file_names, geom_vals, pred_labels, pred_values, fl_values = zip(*geoms)

    geoseries_data = gpd.GeoSeries(data=geom_vals)

    test_gdff = gpd.GeoDataFrame(data=file_names, columns=['name_shp'], geometry=geoseries_data,
                                 crs=str(crs))

    test_gdff['label'] = pred_labels
    test_gdff['value'] = pred_values
    test_gdff['fl_val'] = fl_values

    logging.info(f'Finished performing inference {time.time() - t1}.s')
    logging.info('Finished performing inference')

    inference_dir = os.path.join(output_dir, 'RESNET_' + str(50) + '_Inference_' +
                                 str(input_rstr.split('.')[0][-3:]))
    test_gdff.to_file(inference_dir)

    if bucket_name is not None and s3 is not None:
        for obj in os.listdir(inference_dir):
            s3.Bucket(bucket_name).upload_file(os.path.join(inference_dir, obj), region_name + '_' + obj)

    logging.info('Finished uploading inference files to the S3 bucket')

    del test_gdff, pred_labels, pred_values, fl_values


def main(composite_path: str, region_name: str, model_path: str, progress_file: str = None, cores: int = None,
         output_dir: str = None, s3_bucket_name: str = None):

    # Create output file directories
    aoi_name = 'folder_' + composite_path.split('.')[0][-3:]
    output_dir = output_dir if output_dir is not None else tempfile.mkdtemp(prefix=f'b2p_{aoi_name}')
    tiling_dir = os.path.join(output_dir, f'Tilling_tiff_{aoi_name}_{region_name}')
    os.makedirs(tiling_dir, exist_ok=True)
    rss = rasterio.open(composite_path)

    # Configure logging
    logfile = os.path.join(output_dir, f'output.log')
    logging.basicConfig(filename=logfile, level=logging.INFO)

    # Configure AWS s3
    s3 = boto3.resource('s3')

    geom_lookup = os.path.join(output_dir, 'tiff_geom_lookup.json')
    cores = cores if cores is not None else multiprocessing.cpu_count() - 2

    # Call parent processes for parallel tasks
    sp.call([sys.executable, os.path.join(PROJECT_DIR, 'tiling', 'make_tiles_from_composite.py'),
             '--composite_path', composite_path, '--out_dir', tiling_dir, '--geom_lookup_path', geom_lookup, '--cores',
             cores])

    do_inference(input_rstr=composite_path, region_name=region_name, tiling_dir=tiling_dir, output_dir=output_dir,
                 model_path=model_path, progress_file=progress_file, crs=rss.crs, geom_lookup_path=geom_lookup,
                 bucket_name=s3_bucket_name, s3=s3)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument('--composite_path', '-f', type=str, required=True,
                        help='Path to the color corrected composite s2'
                             ' tile for the region')
    parser.add_argument('--region_name', '-n', type=str, required=True, help='Name of the region')
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to inference model file')

    # Optional args
    parser.add_argument('--cores', '-c', type=int, required=False, help='Number of cores to use in parallel for tiling')
    parser.add_argument('--progress_file', '-p', type=str, required=False, help='Path to progress csv file')
    parser.add_argument('--bucket_name', '-b', type=str, required=False, help='Path to AWS s3 bucket i.e. b2p.erve')
    parser.add_argument('--out_dir', '-o', type=str, required=False, help='Path to directory where output files will be'
                                                                          ' written')

    args = parser.parse_args()

    main(composite_path=args.file, region_name=args.name, model_path=args.model, progress_file=args.progress_file,
         cores=args.cores, output_dir=args.out_dir, s3_bucket_name=args.bucket_name)
