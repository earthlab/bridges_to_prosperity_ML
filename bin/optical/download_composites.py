import argparse
import multiprocessing as mp

from src.utilities.config_reader import CONFIG
from src.base.download_composites import download_composites

# TODO: Add mgrs parameter
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', '-r', type=str, required=False, help='Name of the composite region (Ex Uganda)')
    parser.add_argument('--districts', '-d', type=str, nargs='+', required=False,
                        help='Name of the composite district (Ex. Fafan Ibanda)')
    parser.add_argument('--s3_bucket_name', '-b', required=False, default=CONFIG.AWS.BUCKET, type=str,
                        help='Name of s3 bucket to search for composites in. Default is from project config, which is'
                             f' currently set to {CONFIG.AWS.BUCKET}')
    parser.add_argument('--cores', required=False, type=int, default=mp.cpu_count() - 1,
                        help='Number of cores to use when making tiles in parallel. Default is cpu_count - 1')
    parser.add_argument('--bands', required=False, type=str, default=None, nargs='+',
                        help='The bands to download composites for. The default is None which will select all band '
                             'combinations')
    args = parser.parse_args()

    download_composites(region=args.region, districts=args.districts, s3_bucket_name=args.s3_bucket_name,
                        cores=args.cores, bands=args.bands)
