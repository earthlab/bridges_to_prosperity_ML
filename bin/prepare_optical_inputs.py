from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os 
import pandas as pd
import numpy as np
import multiprocessing as mp
import random 
import boto3

from bin.composites_to_tiles import create_tiles
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
COMPOSITE_DIR = os.path.join(BASE_DIR, 'data', 'tmp_composites')
TILE_DIR = os.path.join(BASE_DIR, 'data', 'tmp_tiles')
TRUTH_DIR = os.path.join(BASE_DIR, 'data', 'ground_truth')
CORES = mp.cpu_count() - 1

def this_download(ks):
    key, sz, n = ks
    s3 = boto3.resource('s3')
    b = s3.Bucket('b2p.njr')
    dst = os.path.join(COMPOSITE_DIR, key)
    if os.path.isfile(dst):
        return None
    dstRoot = os.path.split(dst)[0]
    os.makedirs(dstRoot, exist_ok=True)
    # print(key)
    with tqdm(total=sz, unit='B', unit_scale=True, desc=key, leave=False, position=n) as pbar:
        b.download_file(key, dst, Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    return None

def main():
    ## load composites from s3
    folders = [
        'composites/Rwanda/all',
        'composites/Uganda/Ibanda',
        'composites/Uganda/Kabarole',
        'composites/Uganda/Kasese'
    ]
    s3 = boto3.resource('s3')
    b = s3.Bucket('b2p.njr')
    
    composites = [] 
    n = 1
    for p in folders:
        for obj in b.objects.filter(Prefix=p):
            composites.append((obj.key, obj.size, n%CORES+1))
            n += 1  
    process_map(this_download, composites, max_workers=CORES)
    ## call composites to tiles
    print('Making tiles...')

    matched_df = create_tiles(
        COMPOSITE_DIR,
        TILE_DIR,
        TRUTH_DIR,
        CORES
    )
    ## create training sets and validation sets
    # Seperate the training and validation into seperate files
    b_ix = matched_df.index[matched_df['is_bridge']].tolist()
    nb_ix = matched_df.index[False == matched_df['is_bridge']].tolist()
    b_train_ix = random.sample(b_ix, int(round(0.7*len(b_ix))))
    nb_train_ix = random.sample(nb_ix, int(round(0.7*len(nb_ix))))
    b_val_ix = np.setdiff1d(b_ix, b_train_ix)
    nb_val_ix = np.setdiff1d(nb_ix, nb_train_ix)
    print(f'b_train_ix: {len(b_train_ix)}')
    print(f'nb_train_ix: {len(nb_train_ix)}')
    print(f'b_val_ix: {len(b_val_ix)}')
    print(f'nb_val_ix: {len(nb_val_ix)}')

    train_csv = os.path.join(TILE_DIR, 'train_df.csv')
    val_csv = os.path.join(TILE_DIR, 'val_df.csv')
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
    train_df.to_csv(train_csv) 
    val_df.to_csv(val_csv) 
    print(f'Saving to {train_csv} and {val_csv}')

    b_ix = matched_df.index[matched_df['is_bridge']].tolist()
    nb_ix = matched_df.index[False == matched_df['is_bridge']].tolist()
    b_train_ix = random.sample(b_ix, int(round(0.7*len(b_ix))))
    nb_train_ix = random.sample(nb_ix, int(round(0.7*len(b_ix))))
    b_val_ix = np.setdiff1d(b_ix, b_train_ix)
    nb_val_ix = np.setdiff1d(nb_ix, nb_train_ix)
    nb_val_ix = random.sample(nb_val_ix.tolist(), len(b_val_ix))
    print(f'b_train_ix: {len(b_train_ix)}')
    print(f'nb_train_ix: {len(nb_train_ix)}')
    print(f'b_val_ix: {len(b_val_ix)}')
    print(f'nb_val_ix: {len(nb_val_ix)}')

    train_csv = os.path.join(TILE_DIR, 'train_df_lite.csv')
    val_csv = os.path.join(TILE_DIR, 'val_df_lite.csv')
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
    train_df.to_csv(train_csv) 
    val_df.to_csv(val_csv) 
    print(f'Saving to {train_csv} and {val_csv}')
if __name__ == "__main__":
    main()