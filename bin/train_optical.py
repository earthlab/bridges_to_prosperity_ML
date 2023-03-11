import os
from argparse import ArgumentParser
from src.ml.train import _torch_train_optical

if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument(
    #     '--groundtruth', 
    #     '-g', 
    #     type=str, 
    #     default=None,
    #     required=False, 
    #     help='Path to csv files with ground truth'
    # )
    # parser.add_argument(
    #     '--tiles', 
    #     '-t', 
    #     type=str, 
    #     default=None,
    #     required=False, 
    #     help='Path to tiff files'
    # )
    # parser.add_argument(
    #     '--outdir', 
    #     '-o', 
    #     type=str, 
    #     required=False, 
    #     default=None,
    #     help='Where to write pkl files containing params'
    # )
    # parser.add_argument(
    #     '--resnt', 
    #     '-r', 
    #     type=int, 
    #     required=False, 
    #     default=RESNT_DEFAULT,
    #     help='List of resnt defaults'
    # )
    # parser.add_argument(
    #     '--countries', 
    #     '-c', 
    #     type=str, 
    #     required=True, 
    #     help='List of resnt defaults',
    #     nargs='+'
    # )
    # args = parser.parse_args()

    # train_models(
    #     args.groundtruth, 
    #     args.tiles, 
    #     args.outdir, 
    #     args.country,
    #     args.resnt,
    # )
    _torch_train_optical()
   