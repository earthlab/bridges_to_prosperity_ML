import argparse
from typing import List, Union

from definitions import LAYER_TO_IX
from src.ml.train import train_torch
from file_types import TrainedModel

ARCHITECTURES = ('resnet18', 'resnet34', 'resnet50')


def train_optical(training_csv_path: str, validate_csv_path: str, architectures: List[str] = ARCHITECTURES,
                  no_bridge_to_bridge_ratios: Union[None, List[float]] = None, layers: Union[None, List[str]] = None):
    no_bridge_to_bridge_ratios = [None] if no_bridge_to_bridge_ratios is None else no_bridge_to_bridge_ratios

    for architecture in architectures:
        for no_bridge_to_bridge_ratio in no_bridge_to_bridge_ratios:
            model_file = TrainedModel(architecture, layers, 0, no_bridge_to_bridge_ratio)
            print(f'Writing model files to {model_file.archive_dir}')
            train_torch(
                training_csv_path,
                validate_csv_path,
                architecture,
                bridge_no_bridge_ratio=no_bridge_to_bridge_ratio,
                layers=layers
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_csv',
        '-tr',
        type=str,
        required=True,
        help='Path to csv file with training portion of dataset'
    )
    parser.add_argument(
        '--validate_csv',
        '-val',
        type=str,
        required=True,
        help='Path to csv file with testing / validation portion of dataset'
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
    parser.add_argument(
        '--layers',
        type=list,
        nargs='+',
        required=True,
        default=None,
        help='List of (3) data layers to be used. Ex red,green,blue choose from: '
             ' - red'
             ' - blue'
             ' - green'
             ' - nir'
             ' - osm-water'
             ' - osm-boundary'
             ' - elevation'
             ' - slope'
    )
    args = parser.parse_args()

    if not all([layer in LAYER_TO_IX for layer in args.layers]):
        raise ValueError(f'Invalid layer(s). Valid layers to choose from are: {LAYER_TO_IX}')

    train_optical(
        training_csv_path=args.train_csv,
        validate_csv_path=args.validate_csv,
        architectures=args.architectures,
        no_bridge_to_bridge_ratios=args.ratios,
        layers=args.layers
    )
