import argparse
import os
from pathlib import Path

from src.ml.inference import inference_torch
from src.utilities.config_reader import CONFIG
from src.utilities.files import results_csv_to_shapefile
from file_types import InferenceResultsCSV, TrainedModel, TileMatch, InferenceResultsShapefile


def run_inference(model_file_path: str, tile_csv_path: str, truth_data: bool,
                  batch_size: int = CONFIG.TORCH.INFERENCE.BATCH_SIZE,
                  num_workers: int = CONFIG.TORCH.INFERENCE.NUM_WORKERS, print_frequency: int = 100):
    tile_csv_file = TileMatch.create(tile_csv_path)
    if tile_csv_file is None:
        raise ValueError(f'{os.path.basename(tile_csv_path)} is not a valid tile csv file')

    regions = tile_csv_file.regions

    model_file = TrainedModel.create(model_file_path)
    if model_file is None:
        raise ValueError(f'{os.path.basename(model_file_path)} is not a valid model file')

    results_csv = InferenceResultsCSV(architecture=model_file.architecture, layers=model_file.layers,
                                      epoch=model_file.epoch, ratio=model_file.ratio, best=model_file.best)
    results_csv_archive_path = results_csv.archive_path(regions)
    if os.path.exists(results_csv_archive_path):
        raise FileExistsError(f'File at {results_csv_archive_path} already exists')

    # Don't want to get to end of inference and then not be able to write to output file
    Path(os.path.dirname(results_csv_archive_path)).mkdir(parents=True, exist_ok=True)
    with open(results_csv_archive_path, 'w+') as f:
        if not f.writable():
            raise ValueError(f'Cannot write to output file {results_csv_archive_path}')

    inference_torch(
        model_file_path=model_file_path,
        tile_csv_path=tile_csv_path,
        results_csv_path=results_csv_archive_path,
        batch_size=batch_size,
        num_workers=num_workers,
        print_frequency=print_frequency,
        truth_data=truth_data
    )

    results_shapefile = InferenceResultsShapefile(architecture=model_file.architecture, layers=model_file.layers,
                                                  epoch=model_file.epoch, ratio=model_file.ratio, best=model_file.best)
    results_csv_to_shapefile(results_csv_archive_path, results_shapefile.archive_path(regions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file_path', required=True, type=str,
                        help='Path to the model used to run inference')
    parser.add_argument('--tile_csv_path', required=True, type=str,
                        help='Path to csv file describing all tiles to run inference on.')
    parser.add_argument('--truth_data', action='store_true',
                        help='If set then truth data will be read in and target column will be included in output csv')
    parser.add_argument('--batch_size', required=False, type=int, default=CONFIG.TORCH.INFERENCE.BATCH_SIZE,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', '-w', required=False, type=int, default=CONFIG.TORCH.INFERENCE.NUM_WORKERS,
                        help='Number of workers for inference')
    parser.add_argument('--print_frequency', '-p', required=False, type=int, default=100)
    args = parser.parse_args()

    run_inference(model_file_path=args.model_file_path, tile_csv_path=args.tile_csv_path,
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  print_frequency=args.print_frequency, truth_data=args.truth_data)
