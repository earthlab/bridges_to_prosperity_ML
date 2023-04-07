import argparse
import os
from typing import List, Union

from definitions import TORCH_DIR
from src.ml.train import train_torch

ARCHITECTURES = ('resnet18', 'resnet34', 'resnet50')


def train_optical(training_csv_path: str, test_csv_path: str, architectures: List[str] = ARCHITECTURES,
                  no_bridge_to_bridge_ratios: Union[None, List[float]] = None, results_dir: str = TORCH_DIR):
    os.makedirs(results_dir, exist_ok=True)
    no_bridge_to_bridge_ratios = [None] if no_bridge_to_bridge_ratios is None else no_bridge_to_bridge_ratios

    for architecture in architectures:
        for no_bridge_to_bridge_ratio in no_bridge_to_bridge_ratios:
            train_torch(
                os.path.join(results_dir, f'{architecture}', f'ratio_{no_bridge_to_bridge_ratio}'),
                training_csv_path,
                test_csv_path,
                architecture,
                bridge_no_bridge_ratio=no_bridge_to_bridge_ratio
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_csv',
        '-tr',
        type=str,
        default=None,
        required=False,
        help='Path to csv file with training portion of dataset'
    )
    parser.add_argument(
        '--test_csv',
        '-te',
        type=str,
        default=None,
        required=False,
        help='Path to csv file with testing / validation portion of dataset'
    )
    parser.add_argument(
        '--results_dir',
        '-o',
        type=str,
        required=False,
        default=TORCH_DIR,
        help='Where to write pkl files containing training and validation results'
    )
    parser.add_argument(
        '--architectures',
        '-a',
        type=str,
        nargs='+',
        required=False,
        default=ARCHITECTURES,
        help='List of or single Resnet model architecture(s) to use for training i.e. resnet18,resnet35,resnet50'
    )
    parser.add_argument(
        '--ratios',
        '-r',
        type=float,
        nargs='+',
        required=False,
        help='List of or single ratio(s) of no_bridge to bridge data to fix class balance in test / validation set. '
             'Model will be trained for each ratio specified i.e. 0.5,1,1.5,5. If none then no class balancing will be '
             'done '
    )

    args = parser.parse_args()

    train_optical(
        training_csv_path=args.train_csv,
        test_csv_path=args.train_csv,
        results_dir=args.results_dir,
        architectures=args.architectures,
        no_bridge_to_bridge_ratios=args.ratios
    )

   