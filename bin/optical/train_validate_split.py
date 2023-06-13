import argparse
import multiprocessing as mp
import os
import random
from typing import List

import numpy as np
import pandas as pd
from file_types import TrainSplit, ValidateSplit, TileMatch
from definitions import MULTIVARIATE_TILE_DIR, OPTICAL_TILE_DIR

CORES = mp.cpu_count() - 1


def create_dset_csv(regions: List[str], ratio: int) -> None:
    assert 100 > ratio > 0

    dfs = []
    for region in regions:
        df = TileMatch.find_files()

    if out_dir is None:
        out_dir = os.path.dirname(tile_match_csv_path)

    matched_df = pd.read_csv(tile_match_csv_path)

    # Create training sets and validation sets separate the training and validation into separate files
    b_ix = matched_df.index[matched_df['is_bridge']].tolist()
    nb_ix = matched_df.index[~matched_df['is_bridge']].tolist()
    b_train_ix = random.sample(b_ix, int(round(ratio / 100 * len(b_ix))))
    nb_train_ix = random.sample(nb_ix, int(round(ratio / 100 * len(nb_ix))))
    b_val_ix = np.setdiff1d(b_ix, b_train_ix)
    nb_val_ix = np.setdiff1d(nb_ix, nb_train_ix)

    train_csv = TrainSplit(ratio)
    val_csv = ValidateSplit(ratio)
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
    train_df.to_csv(os.path.join(out_dir, train_csv.name))
    val_df.to_csv(os.path.join(out_dir, val_csv.name))
    print(f'Saving to {train_csv} and {val_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', '-r', type=str, required=True, nargs='+',
                        help='Name of the regions to create the train / validate split file for. Tile match files ' \
                             'for each of the regions will be combined before splitting.'
                        )
    parser.add_argument('--training_ratio', type=int, required=False, default=70, help='Percentage of data to use as'
                                                                                       ' training data. Default is 70')

    args = parser.parse_args()
