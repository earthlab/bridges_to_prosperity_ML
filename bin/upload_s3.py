"""
Uploads different file types to s3 storage including composite, trained model, and inference result files. Several
parameters can be specified in order to make the uploaded file set more specific.
"""
import argparse

from src.utilities.config_reader import CONFIG

import os

from src.utilities.aws import upload_files
from file_types import MultiVariateComposite, TrainedModel, InferenceResultsCSV, InferenceResultsShapefile, File


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Base arguments
    parser.add_argument('--s3_bucket_name', '-b', type=str, required=False, default=CONFIG.AWS.BUCKET,
                        help='Name of the s3 bucket to upload the tiles to. Default is bucket specified in config.yaml')
    subparsers = parser.add_subparsers(help='File type to upload. Can be composites, models, or inference_results',
                                       dest='file_type')

    # Composite file arguments
    composite_parser = subparsers.add_parser('composites')
    composite_parser.add_argument('--region', '-r', type=str, required=False, help='Name of the region to upload'
                                                                                   ' composites for. Default'
                                  'is all regions in the archive')
    composite_parser.add_argument('--district', '-d', type=str, required=False,
                                  help='Name of the district to upload the composites for. Default is all districts per'
                                       ' input region.')
    composite_parser.add_argument('--mgrs', '-m', type=str, nargs='+', required=False,
                                  help='Name of the mgrs tile(s) to download for regions and districts. Default is all'
                                       ' tiles')

    # Inference file arguments
    inference_parser = argparse.ArgumentParser(add_help=False)
    inference_parser.add_argument('--regions', nargs='+', required=False,
                                  help='If specified, only files made from these regions will be uploaded, otherwise'
                                       ' all region combinations will be found')
    inference_parser.add_argument('--architecture', required=False, type=str,
                                  help='If specified, only files of this architecture will be uploaded, otherwise all '
                                       'architectures will be found')
    inference_parser.add_argument('--layers', required=False, nargs='+',
                                  help='If specified, only files made from these layers will be uploaded, otherwise '
                                       'all layer combinations will be found')
    inference_parser.add_argument('--epoch', required=False, type=int,
                                  help='If specified, only files from this epoch will be uploaded, otherwise all epochs'
                                       ' will be found')
    inference_parser.add_argument('--ratio', required=False, type=float,
                                  help='If specified only files of this no bridge / bridge ratio will be uploaded,'
                                       ' otherwise all ratios will be found')
    inference_parser.add_argument('--tile_size', required=False, type=int,
                                  help='If specified only files of this tile size will be uploaded, otherwise all tile'
                                       ' sizes will be found')
    inference_parser.add_argument('--best', required=False, action='store_true',
                                  help='If set, only files marked as best will be uploaded')
    
    models_parser = subparsers.add_parser('models', parents=[inference_parser])
    inference_results_parser = subparsers.add_parser('inference_results', parents=[inference_parser])

    args = parser.parse_args()

    if args.file_type == 'composites':
        files = MultiVariateComposite.find_files(region=args.region, district=args.district, mgrs=args.mgrs)
    elif args.file_type == 'models':
        files = TrainedModel.find_files(regions=args.regions, architecture=args.architecture, layers=args.layers,
                                        epoch=args.epoch, ratio=args.ratio, tile_size=args.tile_size, best=args.best)
    elif args.file_type == 'inference_results':
        shape_files = InferenceResultsShapefile.find_files(regions=args.regions, architecture=args.architecture,
                                                           layers=args.layers, epoch=args.epoch, ratio=args.ratio,
                                                           tile_size=args.tile_size, best=args.best)
        files = []
        for shape_file in shape_files:
            shape_file_object = InferenceResultsShapefile.create(shape_file)
            if not shape_file_object.tar_file.exists:
                shape_file_object.create_tar_file()
            files.append(shape_file_object.tar_file.archive_path)
        files += InferenceResultsCSV.find_files(regions=args.regions, architecture=args.architecture,
                                                layers=args.layers, epoch=args.epoch, ratio=args.ratio,
                                                tile_size=args.tile_size, best=args.best)
    else:
        raise ValueError('Missing first positional argument for file type. Must be one of [composites, models,'
                         ' inference_results]')

    print(f'Found {len(files)} files to upload')

    # TODO: Parallelize this

    upload_files([File.create(f) for f in files], args.s3_bucket_name)
