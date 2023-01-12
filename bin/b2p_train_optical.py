import os
from argparse import ArgumentParser
from src import train_ML

RESNT_DEFAULT=[18, 34, 50, 101, 152]
def train_models(
    truth_dir: str,
    tiles_dir: str,
    outdir: str, 
    countries: list,
    resnt=RESNT_DEFAULT):
    # set truth_dir dir 
    if truth_dir is None: 
        truth_dir = os.path.dirname(os.path.realpath(__file__))
        truth_dir = os.path.abspath( 
            os.path.join(
                indir,
                '..', 
                'data', 
                'truth_dir'
            )
        )
    
    # set tiles_dir dir 
    if tiles_dir is None: 
        tiles_dir = os.path.dirname(os.path.realpath(__file__))
        tiles_dir = os.path.abspath( 
            os.path.join(
                indir,
                '..', 
                'data', 
                'tiles_dir'
            )
        )
    
    # set/make out dir if
    if outdir is None: 
        outdir = os.path.dirname(os.path.realpath(__file__))
        outdir = os.path.abspath( 
            os.path.join(
                outdir,
                '..', 
                'data', 
                'trained_models'
            )
        )
    if not os.path.isdir(outdir): 
        os.mkdir(outdir)
        
    # build dicts for functions 
    csv_files = {}
    for item in os.listdir(truth_dir):
        for c in countries:
            if c in item:
                csv_files[c] = os.path.join(truth_dir, item)

    tiff_dirs = {}
    for item in os.listdir(tiles_dir):
        for c in countries:
            if c in item:
                csv_files[c] = os.path.join(tiles_dir, item)

    # seperate the training data from the val data
    train_df, val_df = train_ML.format_inputs(
        csv_files, 
        tiff_dirs
    )

    interps = train_ML.train_optical(train_df)
    
    train_ML.metrics(val_df, interps, resnt)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--groundtruth', 
        '-g', 
        type=str, 
        default=None,
        required=False, 
        help='Path to csv files with ground truth'
    )
    parser.add_argument(
        '--tiles', 
        '-t', 
        type=str, 
        default=None,
        required=False, 
        help='Path to tiff files'
    )
    parser.add_argument(
        '--outdir', 
        '-o', 
        type=str, 
        required=False, 
        default=None,
        help='Where to write pkl files containing params'
    )
    parser.add_argument(
        '--resnt', 
        '-r', 
        type=int, 
        required=False, 
        default=RESNT_DEFAULT,
        help='List of resnt defaults'
    )
    parser.add_argument(
        '--countries', 
        '-c', 
        type=str, 
        required=True, 
        help='List of resnt defaults',
        nargs='+'
    )
    args = parser.parse_args()

    train_models(
        args.groundtruth, 
        args.tiles, 
        args.outdir, 
        args.country,
        args.resnt,
    )