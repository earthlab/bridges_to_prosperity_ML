from pathlib import Path

import os
import argparse

from src.ml.inference import inference_torch


def main(tile_csv_path: str, model_path: str, results_path: str):
    if os.path.exists(results_path):
        raise FileExistsError(f'File at {results_path} already exists. Please move / remove the file or input a '
                              f'different output file path')

    # Don't want to get to end of inference and then not be able to write to output file
    Path(os.path.dirname(results_path)).mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w+') as f:
        if not f.writable():
            raise ValueError(f'Cannot write to output file {results_path}')

    inference_torch(
        model_file=model_path,
        res_csv=results_path,
        tile_csv=tile_csv_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_csv', required=True, type=str, help='Path to csv file describing all tiles to run '
                                                                    'inference on.')
    parser.add_argument('--model', required=True, type=str, help='Path to the model used to run inference')
    parser.add_argument('--results_csv', required=True, type=str,
                        help='Path where the inference results csv will be written')
    args = parser.parse_args()
    main(tile_csv_path=args.tile_csv, model_path=args.model, results_path=args.results_csv)
