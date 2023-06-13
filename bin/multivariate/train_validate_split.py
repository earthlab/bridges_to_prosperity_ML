import argparse
import multiprocessing as mp
import os
import random
from typing import List

import numpy as np
import pandas as pd
from file_types import TrainSplit, ValidateSplit, TileMatch, FileType
from definitions import MULTIVARIATE_TILE_DIR, OPTICAL_TILE_DIR

CORES = mp.cpu_count() - 1


def create_train_validate_csv(file_type: FileType, regions: List[str], ratio: int) -> None:
    assert 100 > ratio > 0

    tile_dir = MULTIVARIATE_TILE_DIR if file_type == FileType.MULTIVARIATE_COMPOSITE else OPTICAL_TILE_DIR

    dfs = []
    for region in regions:
        df = TileMatch.find_files(os.path.join(tile_dir, region))
        if not df:
            raise LookupError(f'Could not find tile match file for region {region} in {os.path.join(tile_dir, region)}')
        dfs.append(pd.read_csv(df[0]))

    all_region_df = pd.concat(dfs, ignore_index=True)

    # Create training sets and validation sets separate the training and validation into separate files
    b_ix = all_region_df.index[all_region_df['is_bridge']].tolist()
    nb_ix = all_region_df.index[~all_region_df['is_bridge']].tolist()
    b_train_ix = random.sample(b_ix, int(round(ratio / 100 * len(b_ix))))
    nb_train_ix = random.sample(nb_ix, int(round(ratio / 100 * len(nb_ix))))
    b_val_ix = np.setdiff1d(b_ix, b_train_ix)
    nb_val_ix = np.setdiff1d(nb_ix, nb_train_ix)

    train_csv = TrainSplit(ratio)
    val_csv = ValidateSplit(ratio)
    train_df = pd.concat(
        [
            all_region_df.iloc[b_train_ix],
            all_region_df.iloc[nb_train_ix]
        ],
        ignore_index=True
    )
    val_df = pd.concat(
        [
            all_region_df.iloc[b_val_ix],
            all_region_df.iloc[nb_val_ix]
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
