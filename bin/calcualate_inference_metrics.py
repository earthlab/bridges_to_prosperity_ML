"""
Calculautes receiver operator characteristic curve for inference results. Also outputs the optimal confidence threshold based on the G-Mean.
"""
import argparse
from src.ml.metrics import Metrics


def calculate_inference_metrics(inference_results_path: str, validation_set_path: str, out_path: str):
    """
    Calculates and outputs the receiver operator characteristic curve for an inference results csv file. Also
    calculates the optimal confidence threshold based on the G-Mean.
    Args:
        inference_results_path (str): Path to the inference results csv file
        validation_set_path (str): Path to the set of validation data used to train the inference model
        out_path (str): Path to save the output ROC curve file to
    """
    metrics = Metrics(inference_results_path, validation_set_path)
    metrics.plot_roc(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_results', type=str, required=True, help='Path to inference results csv file')
    parser.add_argument('--validation_set', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save the ROC plot to')

    args = parser.parse_args()

    calculate_inference_metrics(args.inference_results, args.validation_set, args.out_path)
