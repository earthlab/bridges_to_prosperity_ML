import os
import random

import numpy as np
import pandas as pd


def subset_data(match_file:str='data/tiles/tiles_with_truth.csv'):
    df = pd.read_csv(match_file)
    out_dir = 'data/tiles/'
    # Make a training set where the # of bridge tiles == # of no bridge tiles
    b_ix = df.index[df['is_bridge']].tolist()
    nb_ix = df.index[False == df['is_bridge']].tolist()
    b_train_ix = random.sample(b_ix, int(round(0.7*len(b_ix))))
    nb_train_ix = random.sample(nb_ix, len(b_train_ix))
    b_val_ix = np.setdiff1d(b_ix, b_train_ix)
    nb_val_ix = np.setdiff1d(nb_ix, nb_train_ix)
    nb_val_ix = random.sample(nb_val_ix.tolist(), len(b_val_ix))
    print(f'b_train_ix: {len(b_train_ix)}')
    print(f'nb_train_ix: {len(nb_train_ix)}')
    print(f'b_val_ix: {len(b_val_ix)}')
    print(f'nb_val_ix: {len(nb_val_ix)}')

    train_csv = os.path.join(out_dir, 'train_df_lite.csv')
    val_csv = os.path.join(out_dir, 'val_df_lite.csv')
    train_df = pd.concat(
        [
            df.iloc[b_train_ix],
            df.iloc[nb_train_ix]
        ], 
        ignore_index=True
    )
    val_df = pd.concat(
        [
            df.iloc[b_val_ix],
            df.iloc[nb_val_ix]
        ], 
        ignore_index=True
    )
    train_df.to_csv(train_csv) 
    val_df.to_csv(val_csv) 
    print(f'Saving to {train_csv} and {val_csv}')

if __name__ == '__main__':
    main()