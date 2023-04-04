from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os 
import pandas as pd
import numpy as np
import multiprocessing as mp
import random 
import boto3
from src.utilities.coords import get_bridge_locations
from bin.composites_to_tiles import create_tiles

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
COMPOSITE_DIR = os.path.join(BASE_DIR, 'data', 'composites')
TILE_DIR = os.path.join(BASE_DIR, 'data', 'tiles')
TRUTH_DIR = os.path.join(BASE_DIR, 'data', 'ground_truth')
CORES = mp.cpu_count() - 1


def this_download(ksn):
    for key, sz, n in ksn: 
        s3 = boto3.resource('s3')
        b = s3.Bucket('b2p.njr')
        dst = os.path.join(COMPOSITE_DIR, key.split('composites/')[1])
        if os.path.isfile(dst):
            return None
        dstRoot = os.path.split(dst)[0]
        os.makedirs(dstRoot, exist_ok=True)
        # print(key)
        with tqdm(total=sz, unit='B', unit_scale=True, desc=key, leave=False, position=n) as pbar:
            b.download_file(key, dst, Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    return None


def creat_dset_csv(matched_df, ratio):
    assert ratio<1 and ratio > 0
    ## create training sets and validation sets
    # Seperate the training and validation into seperate files
    b_ix = matched_df.index[matched_df['is_bridge']].tolist()
    nb_ix = matched_df.index[False == matched_df['is_bridge']].tolist()
    b_train_ix = random.sample(b_ix, int(round(ratio*len(b_ix))))
    nb_train_ix = random.sample(nb_ix, int(round(ratio*len(nb_ix))))
    b_val_ix = np.setdiff1d(b_ix, b_train_ix)
    nb_val_ix = np.setdiff1d(nb_ix, nb_train_ix)

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
    
    ks = [] 
    for p in folders:
        for obj in b.objects.filter(Prefix=p):
            ks.append((obj.key, obj.size))
    inputs = []
    for i, ks_i in enumerate(np.array_split(ks, CORES)):
        inputs.append([])
        for ks_ij in ks_i:
            inputs[i].append((ks_ij[0], ks_ij[1], i+1))
    process_map(
        this_download, 
        inputs, 
        max_workers=CORES
    )

    ## call composites to tiles
    print('Making tiles...')
    bridge_locations = get_bridge_locations(TRUTH_DIR)  
    composites = glob(os.path.join(COMPOSITE_DIR, "**/*multiband.tiff"), recursive=True)
    inputs = [
        (
            cs,
            TILE_DIR,
            bridge_locations, 
            n+1
        )
    for n, cs in enumerate(np.array_split(composites, CORES))]
    matched_df = process_map(
        create_tiles, 
        inputs, 
        max_workers=CORES
    )
    matched_df = pd.concat(matched_df, ignore_index=True)
    
    print("Creating data set")
    creat_dset_csv(matched_df, 0.7)


if __name__ == "__main__":
    main()