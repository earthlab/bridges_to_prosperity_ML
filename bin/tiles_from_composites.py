"""
Creates tiff tile files for the specified composites. By default, this will crawl the data/composites directory and make
tiles for all composites and write them to data/tiles. The input composite directory, output tile directory, and ground
truth directory paths can be overriden so process is only completed for specified region.
"""
import pandas
import yaml
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from definitions import REGION_FILE_PATH, COMPOSITE_DIR, TILE_DIR
from src.utilities.coords import get_bridge_locations
from src.utilities.imaging import composite_to_tiles
from file_types import OpticalComposite, TileMatch, FileType, MultiVariateComposite, File


def create_tiles(args):
    composite_files, bridge_locs, pos = args
    df = []
    for composite_path in tqdm(composite_files, position=0, leave=True):
        composite_file = File.create(composite_path)
        if isinstance(composite_file, OpticalComposite):
            bands = composite_file.bands
        elif isinstance(composite_file, MultiVariateComposite):
            bands = ['multivariate']
        else:
            continue
        df_i = composite_to_tiles(composite_path, bands, bridge_locs, pos)
        df.append(df_i)
    df = pd.concat(df, ignore_index=True)
    return df


def tiles_from_composites(no_truth: bool, cores: int, region: str):
    bridge_locations = None if no_truth else get_bridge_locations()

    with open(REGION_FILE_PATH, 'r') as f:
        region_info = yaml.safe_load(f)
        districts = list(region_info[region]['districts'].keys())

    dfs = []
    for district in districts:
        composite_dir = os.path.join(COMPOSITE_DIR, region, district)
        composites = MultiVariateComposite.find_files(composite_dir, recursive=True)

        if cores == 1:
            matched_df = create_tiles((composites, bridge_locations, 1))
        else:
            inputs = [
                (
                    cs,
                    bridge_locations,
                    n
                )
                for n, cs in enumerate(np.array_split(composites, cores))]
            matched_df = process_map(
                create_tiles,
                inputs,
                max_workers=cores
            )

        tile_match_file = TileMatch()
        tile_match_path = tile_match_file.archive_path(region, district)

        unique_bridge_locations = filter_non_unique_bridge_locations(matched_df)
        dfs.append(unique_bridge_locations)

        unique_bridge_locations.to_csv(tile_match_path)

    regional_matched_df = pd.concat(dfs, ignore_index=True)
    unique_bridge_locations = filter_non_unique_bridge_locations(regional_matched_df)
    regional_tile_match_file = TileMatch()
    unique_bridge_locations.to_csv(regional_tile_match_file.archive_path(region))


def filter_non_unique_bridge_locations(matched_df: pandas.DataFrame) -> pandas.DataFrame:
    rows_to_delete = []
    bridge_dup = []
    for i, t in enumerate(matched_df['is_bridge']):
        if t:
            if matched_df['bridge_loc'][i] in bridge_dup:
                rows_to_delete.append(i)
            else:
                bridge_dup.append(matched_df['bridge_loc'][i])

    return matched_df.drop(rows_to_delete)
