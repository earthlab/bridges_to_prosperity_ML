"""
Runs inference over a set of regions given a certain input model. Inference results are output to both csv and
shapefile formats.
If there is truth data for the set of regions, the truth data flag can be set and the target for each prediction will
be added to the results files.
"""
import argparse
import os
from typing import List

import pandas as pd

from src.ml.inference import inference_torch
from src.utilities.config_reader import CONFIG
from src.utilities.files import results_csv_to_shapefile, filter_non_unique_bridge_locations
from file_types import InferenceResultsCSV, TrainedModel, MultiRegionTileMatch, SingleRegionTileMatch, InferenceResultsShapefile


def run_inference(model_file_path: str, regions: List[str], truth_data: bool,
                  batch_size: int = CONFIG.TORCH.INFERENCE.BATCH_SIZE,
                  num_workers: int = CONFIG.TORCH.INFERENCE.NUM_WORKERS, print_frequency: int = 100) -> None:
    """
    Runs inference over a set of regions given a certain input model. Inference results are output to both csv and
    shapefile formats.
    Args:
        model_file_path (str): Path to the pytorch model file that will be used to run inference over the specified
        input regions
        regions (list): Region(s) to run inference over. Must be in regions_info.yaml file
        truth_data (bool): If set truth data will be searched for the input regions and a target row will be added to
        the output csv file
        batch_size (int): Batch size for inference 
        num_workers(int): Number of workers to use for running inference
        print_requency (int): Frequency of progress updates printed to the console
    """
    model_file = TrainedModel.create(model_file_path)
    if model_file is None:
        raise ValueError(f'{os.path.basename(model_file_path)} is not a valid model file')

    if len(regions) > 1:
        all_region_tile_match = MultiRegionTileMatch(regions=regions, tile_size=model_file.tile_size)
        if not all_region_tile_match.exists:
            all_region_tile_match.create_archive_dir()
            dfs = []
            for region in regions:
                region_tile_match = SingleRegionTileMatch(region=region, tile_size=model_file.tile_size)
                if not region_tile_match.exists:
                    raise LookupError(f'Could not find tile match csv for region {region}')
                dfs.append(pd.read_csv(region_tile_match.archive_path))
            dfs = filter_non_unique_bridge_locations(pd.concat(dfs, ignore_index=True))
            dfs.to_csv(all_region_tile_match.archive_path)
    else:
        all_region_tile_match = SingleRegionTileMatch(region=regions[0], tile_size=model_file.tile_size)

    results_csv = InferenceResultsCSV(regions=regions, architecture=model_file.architecture, layers=model_file.layers,
                                      epoch=model_file.epoch, ratio=model_file.ratio, tile_size=model_file.tile_size,
                                      best=model_file.best)
    results_csv.create_archive_dir()
    if results_csv.exists:
        raise FileExistsError(f'File at {results_csv.archive_path} already exists')

    # Don't want to get to end of inference and then not be able to write to output file
    with open(results_csv.archive_path, 'w+') as f:
        if not f.writable():
            raise ValueError(f'Cannot write to output file {results_csv.archive_path}')

    inference_torch(
        model_file=model_file,
        tile_match_file=all_region_tile_match,
        results_file=results_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        print_frequency=print_frequency,
        truth_data=truth_data
    )

    results_shapefile = InferenceResultsShapefile(regions=regions, architecture=model_file.architecture,
                                                  layers=model_file.layers, epoch=model_file.epoch,
                                                  ratio=model_file.ratio, tile_size=model_file.tile_size,
                                                  best=model_file.best)
    results_shapefile.create_archive_dir()
    results_csv_to_shapefile(results_csv.archive_path, results_shapefile.archive_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file_path', required=True, type=str,
                        help='Path to the model used to run inference')
    parser.add_argument('--inference_regions', required=True, type=str, nargs='+',
                        help='Region(s) to run inference over')
    parser.add_argument('--truth_data', action='store_true',
                        help='If set then truth data will be read in and target column will be included in results csv')
    parser.add_argument('--batch_size', required=False, type=int, default=CONFIG.TORCH.INFERENCE.BATCH_SIZE,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', '-w', required=False, type=int, default=CONFIG.TORCH.INFERENCE.NUM_WORKERS,
                        help='Number of workers for inference')
    parser.add_argument('--print_frequency', '-p', required=False, type=int, default=100)
    args = parser.parse_args()

    run_inference(model_file_path=args.model_file_path, regions=args.inference_regions,
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  print_frequency=args.print_frequency, truth_data=args.truth_data)
