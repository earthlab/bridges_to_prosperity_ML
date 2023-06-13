"""
Creates tiff tile files for the specified composites. By default, this will crawl the data/composites directory and make
tiles for all composites and write them to data/tiles. The input composite directory, output tile directory, and ground
truth directory paths can be overriden so process is only completed for specified region.
"""
import argparse
import multiprocessing as mp
import os

from definitions import B2P_DIR
from file_types import FileType
from src.base.tiles_from_composites import tiles_from_composites


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--region',
        '-r',
        type=str,
        required=True,
        help='Name of the composite region (Ex Uganda)'
    )
    parser.add_argument(
        '--no_truth',
        '-nt',
        action='store_true',
        help='If set then no truth data will be used to create output dataframe'
    )
    parser.add_argument(
        '--truth_dir',
        '-t',
        type=str,
        required=False,
        default=os.path.join(B2P_DIR, "data", "ground_truth"),
        help='Path to directory where csv bridge locations'
    )
    parser.add_argument(
        '--cores',
        '-c',
        type=int,
        default=mp.cpu_count() - 1,
        required=False,
        help='Number of cores to use in parallel for tiling'
    )

    args = parser.parse_args()

    tiles_from_composites(file_type=FileType.MULTIVARIATE_COMPOSITE, no_truth=args.no_truth, cores=args.cores,
                          region=args.region)
