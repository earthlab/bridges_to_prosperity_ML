import argparse
from src.ml.metrics import Metrics


def calculate_inference_metrics(inference_results_path: str, validation_set_path: str, out_path: str):
    metrics = Metrics(inference_results_path, validation_set_path)
    metrics.plot_roc(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_results', type=str, required=True, help='Path to inference results csv file')
    parser.add_argument('--validation_set', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save the ROC plot to')
