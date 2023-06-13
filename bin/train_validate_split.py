import argparse
import multiprocessing as mp
import os
import random
from typing import List

import numpy as np
import pandas as pd
from file_types import TrainSplit, ValidateSplit, TileMatch
from definitions import TILE_DIR

CORES = mp.cpu_count() - 1


def create_dset_csv(regions: List[str], ratio: int) -> None:
    assert 100 > ratio > 0

    train_dfs = []
    val_dfs = []
    matched_dfs = []
    for region in regions:
        df = TileMatch.find_files(os.path.join(TILE_DIR, region))
        if not df:
            raise LookupError(f'Could not find tile match file for region {region} in {os.path.join(TILE_DIR, region)}')

        matched_df = pd.read_csv(df[0])

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
    joined_matched_df = pd.concat(matched_dfs, ignore_index=True)

    train_csv = TrainSplit(regions, ratio)
    val_csv = ValidateSplit(regions, ratio)
    matched_df = TileMatch(regions)

    joined_train_df.to_csv(train_csv.archive_path())
    joined_val_df.to_csv(val_csv.archive_path())
    joined_matched_df.to_csv(matched_df.archive_path())
    print(f'Saving to {train_csv.archive_path()}, {val_csv.archive_path()}, and {matched_df.archive_path()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', '-r', type=str, required=True, nargs='+',
                        help='Name of the regions to create the train / validate split file for. Tile match files ' \
                             'for each of the regions will be combined before splitting.'
                        )
    parser.add_argument('--training_ratio', type=int, required=False, default=70, help='Percentage of data to use as'
                                                                                       ' training data. Default is 70')

    args = parser.parse_args()
