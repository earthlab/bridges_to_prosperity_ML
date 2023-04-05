import os

from src.ml.train import train_torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
TILE_DIR = os.path.join(BASE_DIR, 'data', 'tiles')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'torch')
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
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    assert os.path.isdir(TILE_DIR), f"Please create tiles prior to training {TILE_DIR} DNE"
    for arch in ['resnet18', 'resnet34', 'resnet50']:
        for ratio in [.5, 1, 1.5, 2, 2.5, 3, 5]:
            train_torch(
                os.path.join(DATA_DIR, f'{arch}', f'ratio_{ratio}'), 
                TILE_DIR, 
                arch,
                ratio=ratio
            )
   