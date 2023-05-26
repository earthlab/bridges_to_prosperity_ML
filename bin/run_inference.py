import argparse
import os
from pathlib import Path

from src.ml.inference import inference_torch
from src.utilities.config_reader import CONFIG


def inference_optical(model_file_path: str, tile_csv_path: str, results_csv_path: str, truth_data: bool,
                      batch_size: int = CONFIG.TORCH.INFERENCE.BATCH_SIZE,
                      num_workers: int = CONFIG.TORCH.INFERENCE.NUM_WORKERS, print_frequency: int = 100):
    if os.path.exists(results_csv_path):
        raise FileExistsError(f'File at {results_csv_path} already exists. Please move / remove the file or input a '
                              f'different output file path')

    # Don't want to get to end of inference and then not be able to write to output file
    Path(os.path.dirname(results_csv_path)).mkdir(parents=True, exist_ok=True)
    with open(results_csv_path, 'w+') as f:
        if not f.writable():
            raise ValueError(f'Cannot write to output file {results_csv_path}')

    inference_torch(
        model_file_path=model_file_path,
        tile_csv_path=tile_csv_path,
        results_csv_path=results_csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        print_frequency=print_frequency,
        truth_data=truth_data
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file_path', required=True, type=str,
                        help='Path to the model used to run inference')
    parser.add_argument('--tile_csv_path', required=True, type=str,
                        help='Path to csv file describing all tiles to run inference on.')
    parser.add_argument('--results_csv_path', required=True, type=str,
                        help='Path where the inference results csv will be written')
    parser.add_argument('--truth_data',  action='store_true',
                        help='If set then truth data will be read in and target column will be included in output csv')
    parser.add_argument('--batch_size', required=False, type=int, default=CONFIG.TORCH.INFERENCE.BATCH_SIZE,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', '-w', required=False, type=int, default=CONFIG.TORCH.INFERENCE.NUM_WORKERS,
                        help='Number of workers for inference')
    parser.add_argument('--print_frequency', '-p', required=False, type=int, default=100)
    args = parser.parse_args()

    inference_optical(model_file_path=args.model_file_path, tile_csv_path=args.tile_csv_path,
                      results_csv_path=args.results_csv_path, batch_size=args.batch_size, num_workers=args.num_workers,
                      print_frequency=args.print_frequency, truth_data=args.truth_data)
