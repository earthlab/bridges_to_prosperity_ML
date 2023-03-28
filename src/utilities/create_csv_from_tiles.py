import os
import pandas as pd


def create_csv(indirs, outpath):
    dfs = []
    for region_dir in indirs:
        for zone in os.listdir(region_dir):
            for file in os.listdir(os.path.join(region_dir, zone)):
                if '_geoloc.csv' in file:
                    df = pd.read_csv(os.path.join(region_dir, zone, file))
                    dfs.append(df)

    df = pd.concat(dfs)
    #df['index'] = [i for i in range(len(df))]
    df.to_csv(outpath)
