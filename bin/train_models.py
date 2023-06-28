"""
Trains models to infer bridge locations and outputs the models to archive. The architecture, no bridge / bridge ratio, and layers used to train the model
can be specified. 
"""
import os
import argparse
from typing import List, Union

from definitions import LAYER_TO_IX, TRAIN_VALIDATE_SPLIT_DIR
from src.ml.train import train_torch
from file_types import TrainedModel, TrainSplit, ValidateSplit

ARCHITECTURES = ('resnet18', 'resnet34', 'resnet50')


def train_models(regions: List[str], training_ratio: int, layers: List[str], tile_size: int,
                 architectures: List[str] = ARCHITECTURES,
                 no_bridge_to_bridge_ratios: Union[None, List[float]] = None) -> None:
    """
    Trains models to infer bridge locations and outputs the models to archive. The architecture, no bridge / bridge
    ratio, and layers used to train the model can be specified.
    Args:
        regions (list): List of regions to use to train each model
        training_ratio (int): The ratio of training to validation data used to train the model
        layers (list): The names of the layers to use from the multivariate composites to train the model. Can pick up
         to 3 layers (red blue green nir osm-water osm-boundary elevation slope)
        tile_size (int): Size of the tiles in meters to use to train the models. These tiles and the corresponding tile
         match file should already exist for each input region.
        architectures (list): List of model architectures to use when training. The default is resnet18, resnet34, and
         resnet50
        no_bridge_to_bridge_ratios (list): List of class ratios (no_bridge / bridge) to use when training the models
    """
    if not 100 < training_ratio < 0:
        raise ValueError('Training ratio must be between 0 and 100')

    training_csv_file = TrainSplit(regions=regions, ratio=training_ratio, tile_size=tile_size)
    training_csv_file.create_archive_dir()
    if not training_csv_file.exists:
        raise LookupError(f'Could not find training split csv in {TRAIN_VALIDATE_SPLIT_DIR}. Run '
                          f'train_validate_split.py with the specified regions, training ratio, and tile size to create'
                          f' it')

    validate_csv_file = ValidateSplit(regions=regions, ratio=100-training_ratio, tile_size=tile_size)
    validate_csv_file.create_archive_dir()
    if not validate_csv_file.exists:
        raise LookupError(f'Could not find validate split csv in {TRAIN_VALIDATE_SPLIT_DIR}. Run '
                          f'train_validate_split.py with the specified regions, training ratio, and tile size to create'
                          f' it')

    no_bridge_to_bridge_ratios = [None] if no_bridge_to_bridge_ratios is None else no_bridge_to_bridge_ratios

    for architecture in architectures:
        for no_bridge_to_bridge_ratio in no_bridge_to_bridge_ratios:
            model_file = TrainedModel(regions, architecture, layers, 0, no_bridge_to_bridge_ratio, tile_size=100)
            model_file.create_archive_dir()
            print(f'Writing model files to {os.path.dirname(model_file.archive_path)}')
            train_torch(
                training_csv_file.archive_path,
                validate_csv_file.archive_path,
                regions,
                architecture,
                bridge_no_bridge_ratio=no_bridge_to_bridge_ratio,
                layers=layers,
                tile_size=tile_size
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--regions',
        '-r',
        type=str,
        required=True,
        nargs='+',
        help='Name of the regions whose tiles should be used to train the model. These will be used along with the '
             'input training ratio in order to find the corresponding train / validation dataset'
    )
    parser.add_argument(
        '--class_ratios',
        '-r',
        type=float,
        nargs='+',
        required=False,
        help='List of or single ratio(s) of no_bridge to bridge data to fix class balance in test / validation set. '
             'Model will be trained for each ratio specified i.e. 0.5,1,1.5,5.'
    )
    parser.add_argument(
        '--tile_size',
        type=int,
        nargs='+',
        required=False,
        default=300,
        help='Size of the tiles to be used for training the model. The size is in meters and the tiles and tile match '
             'file for the input tile size should already exist'
    )
    parser.add_argument(
        '--training_ratio',
        type=int,
        required=False,
        default=70,
        help='The ratio of test to validation data used to train. This will be used along with the input regions in '
             'order to find the corresponding train / validation dataset'
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

    if not 0 < args.layers < 4:
        raise ValueError('Must pick between between 1 and 3 layers')

    if not all([layer in LAYER_TO_IX for layer in args.layers]):
        raise ValueError(f'Invalid layer(s). Valid layers to choose from are: {LAYER_TO_IX}')

    train_models(
        regions=args.regions,
        no_bridge_to_bridge_ratios=args.class_ratios,
        tile_size=args.tile_size,
        training_ratio=args.training_ratio,
        architectures=args.architectures,
        layers=args.layers
    )
