"""
Downloads files from s3 storage. The root s3 bucket name can be specified but defaults to project configuration value.
Regions, districts, mgrs tiles, bands, and other parameters can be specified to only get specific location's composites. Use the --help flag to see a full list. 
Downloads are done in parallel by default but can be changed with cores parameter.
"""
import argparse
import multiprocessing as mp

import numpy as np
from tqdm.contrib.concurrent import process_map

from src.utilities.config_reader import CONFIG
from src.utilities.aws import list_s3_files, s3_download_task
from file_types import OpticalComposite, MultiVariateComposite, InferenceResultsCSV, InferenceResultsTarfile, TrainedModel, File



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Base arguments
    parser.add_argument('--s3_bucket_name', '-b', type=str, required=False, default=CONFIG.AWS.BUCKET,
                        help='Name of the s3 bucket to upload the tiles to. Default is bucket specified in config.yaml')
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='Number of cores to use when making tiles in parallel. Default is cpu_count - 1')
    
    subparsers = parser.add_subparsers(help='File type to upload. Can be composites, models, or inference_results', dest='file_type')

    # Composite file arguments
    composite_parser = subparsers.add_parser('composites')
    composite_parser.add_argument('--region', '-r', type=str, required=False, help='Name of the region to upload '
                                                                                   'composites for. Defualt'
                                  'is all regions in the archive')
    composite_parser.add_argument('--district', '-d', type=str, required=False,
                                  help='Name of the district to upload the composites for. Defualt is all districts '
                                       'per input region.')
    composite_parser.add_argument('--mgrs', '-m', type=str, nargs='+', required=False,
                                  help='Name of the mgrs tile(s) to download for regions and districts. Default is all'
                                       ' tiles')
    composite_parser.add_argument('--bands', '-b', type=str, nargs='+', required=False,
                                  help='Optical band combination (B02, B03, B08) etc to download. If not specifed then'
                                       ' all combinations are downloaded')

    # Inference file arguments
    inference_parser = argparse.ArgumentParser(add_help=False)
    inference_parser.add_argument('--regions', nargs='+', required=False, help='If specified, only files made from these regions will be uploaded, otherwise all region combinations will be found')
    inference_parser.add_argument('--architecture', required=False, type=str, help='If specified, only files of this architecture will be uploaded, otherwise all architectures will be found')
    inference_parser.add_argument('--layers', required=False, nargs='+', help='If specified, only files made from these layers will be uploaded, otherwise all layer combinations will be found')
    inference_parser.add_argument('--epoch', required=False, type=int, help='If specified, only files from this epoch will be uploaded, otherwise all epochs will be found')
    inference_parser.add_argument('--ratio', required=False, type=float, help='If specified only files of this no bridge / bridge ratio will be uploaded, otherwise all ratios will be found')
    inference_parser.add_argument('--tile_size', required=False, type=int, help='If specified only files of this tile size will be uploaded, otherwise all tile sizes will be found')
    inference_parser.add_argument('--best', required=False, action='store_true', help='If set, only files marked as best will be uploaded')
    
    models_parser = subparsers.add_parser('models', parents=[inference_parser])
    inference_results_parser = subparsers.add_parser('inference_results', parents=[inference_parser])

    args = parser.parse_args()

    files_to_download = []
    if args.file_type == 'composites':
        s3_files = list_s3_files(MultiVariateComposite.ROOT_S3_DIR, args.s3_bucket_name)
        multivariate_files = MultiVariateComposite.filter_files(s3_files, args.region, args.district, args.mgrs)
        optical_files = OpticalComposite.filter_files(s3_files, args.region, args.district, args.bands, args.mgrs)
        files_to_download = multivariate_files + optical_files
    elif args.file_type == 'models':
        s3_files = list_s3_files(TrainedModel.ROOT_S3_DIR, args.s3_bucket)
        files_to_download = TrainedModel.filter_files(s3_files, args.regions, args.architecture, args.layers, args.epoch, args.ration, args.tile_size, args.best)
    elif args.inference_results == 'inference_results':
        s3_files = list_s3_files(InferenceResultsCSV.ROOT_S3_DIR, args.s3_bucket)
        csv_files = InferenceResultsCSV.filter_files(s3_files, args.regions, args.architecture, args.layers, args.epoch, args.ration, args.tile_size, args.best)
        tar_files = InferenceResultsTarfile.filter_files(s3_files, args.regions, args.architecture, args.layers, args.epoch, args.ration, args.tile_size, args.best)
        files_to_download = csv_files + tar_files
    else:
        raise ValueError('Missing first positional argument for file type. Must be one of [composites, models, inference_results]')
    
    print(f'Found {len(files_to_download)} files to download')

    parallel_inputs = []
    for i, s3_file_path in enumerate(np.array_split(files_to_download, args.cores)):
        file_class = File.create(s3_file_path)
        file_class.create_archive_dir()
        parallel_inputs.append((s3_file_path, file_class.archive_path, str(i + 1), args.s3_bucket_name))
    process_map(
        s3_download_task,
        parallel_inputs,
        max_workers=args.cores
    )
