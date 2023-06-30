"""
Creates a train / validation csv set from a single tile match file for a given ratio of training to validation data. The tile match file 
for each region and tile size must already exist.
"""
import argparse
import multiprocessing as mp
import random
from typing import List

import numpy as np
import pandas as pd
from file_types import TrainSplit, ValidateSplit, SingleRegionTileMatch, MultiRegionTileMatch
from src.utilities.files import filter_non_unique_bridge_locations

CORES = mp.cpu_count() - 1


def create_dset_csv(regions: List[str], ratio: int, tile_size: int) -> None:
    """
    Creates a train / validation csv set from a single tile match file for a given ratio of training to validation data. The tile match file 
    for each region and tile size must already exist.
    Args:
        regions (list): List of regions to create the data set from. The tile match file for each of these regions will be concatenated into a single
            multi-region tile match file
        ratio (int): Ratio of training to validation data to use. A ratio of 70 will split the tile match file into a file with 70% training and 30% validation data
        tile_size (int): The size of the tiles in meters to find the tile match file for in each region
    """
    assert 100 > ratio > 0

    train_dfs = []
    val_dfs = []
    matched_dfs = []
    for region in regions:
        df = SingleRegionTileMatch(region=region, tile_size=tile_size)
        if not df.exists:
            raise LookupError(
                f'Could not find tile match file at {df.archive_path} for region {region} and tile size {tile_size}.'
                f' Run tiles_from_composites.py with the region set to {region} and tile size set to {tile_size} to create this file')
        matched_df = pd.read_csv(df.archive_path)

        # Create training sets and validation sets separate the training and validation into separate files
        b_ix = matched_df.index[matched_df['is_bridge']].tolist()
        nb_ix = matched_df.index[~matched_df['is_bridge']].tolist()
        b_train_ix = random.sample(b_ix, int(round(ratio / 100 * len(b_ix))))
        nb_train_ix = random.sample(nb_ix, int(round(ratio / 100 * len(nb_ix))))
        b_val_ix = np.setdiff1d(b_ix, b_train_ix)
        nb_val_ix = np.setdiff1d(nb_ix, nb_train_ix)
        train_df = pd.concat(
            [
                matched_df.iloc[b_train_ix],
                matched_df.iloc[nb_train_ix]
            ],
            ignore_index=True
        )
        val_df = pd.concat(
            [
                matched_df.iloc[b_val_ix],
                matched_df.iloc[nb_val_ix]
            ],
            ignore_index=True
        )
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        matched_dfs.append(matched_df)

    joined_train_df = pd.concat(train_dfs, ignore_index=True)
    joined_val_df = pd.concat(val_dfs, ignore_index=True)
    joined_matched_df = filter_non_unique_bridge_locations(pd.concat(matched_dfs, ignore_index=True))

    train_csv = TrainSplit(regions, ratio, tile_size)
    val_csv = ValidateSplit(regions, int(100-ratio), tile_size)
    matched_df = MultiRegionTileMatch(regions, tile_size)
    train_csv.create_archive_dir()
    val_csv.create_archive_dir()
    matched_df.create_archive_dir()

    joined_train_df.to_csv(train_csv.archive_path)
    joined_val_df.to_csv(val_csv.archive_path)
    joined_matched_df.to_csv(matched_df.archive_path)
    print(f'Saving to {train_csv.archive_path}, {val_csv.archive_path}, and {matched_df.archive_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', '-r', type=str, required=True, nargs='+',
                        help='Name of the region(s) to create the train / validate split file for. Tile match files ' \
                             'for each of the regions will be combined before splitting.'
                        )
    parser.add_argument('--training_ratio', type=int, required=False, default=70, help='Percentage of data to use as'
                                                                                       ' training data. Default is 70')
    parser.add_argument('--tile_size', type=int, nargs='+', required=False, default=300,
                        help='Size of the tiles to be used for training the model. The size is in meters and the tiles'
                             ' and tile match file for the input tile size should already exist')
    args = parser.parse_args()

    create_dset_csv(args.regions, args.training_ratio, args.tile_size)
