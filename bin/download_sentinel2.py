# This file will call the Sentinel2 API to download data for a specified region and then clean that data by making
# composites
import os
from argparse import ArgumentParser
from src.api.sentinel2 import SinergiseSentinelAPI

def download_sentinel2(outdir, bounds, start_date, end_date, buffer):
    api = SinergiseSentinelAPI()
    api.download(bounds, buffer, outdir, start_date, end_date)

    if not outdir:
        raise FileNotFoundError('No files returned from the query parameters')


if __name__ == '__main__':
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)
            ), 
            '..'
        )
    )
    parser = ArgumentParser()
    parser.add_argument(
        '--outdir', 
        '-o', 
        type=str, 
        required=False, 
        default=os.path.join(base_dir, 'data', 'sentinel2'),
        help='Where to write the sentinel2 files and composites')
    parser.add_argument(
        '--bbox', 
        '-b', 
        type=float, 
        nargs=4, 
        required=True, 
        help='Bounding box defined in lat / lon in the order min_lon min_lat max_lon max_lat')
    parser.add_argument(
        '--start_date', 
        '-sd', 
        type=str, 
        required=True, 
        help='Start date for Sentinel2 file query in YYYY-MM-DD'
    )
    parser.add_argument(
        '--end_date', 
        '-ed', 
        type=str, 
        required=True, 
        help='End date for Sentinel2 file query in YYYY-MM-DD'
    )
    parser.add_argument(
        '--buffer', 
        required=False, 
        default=100,
        type=float, 
        help='Buffer for bounding box query in meters'
    )
    parser.add_argument(
        '--slices', 
        required=False, 
        type=int, 
        default=1,
        help='Number of slices to split the tiles up into when making the composite. Default is 1'
    )
    parser.add_argument(
        '--region', 
        required=True, 
        type=str, 
        help='Region that the data pertains to (ie Uganda)'
    )
    parser.add_argument(
        '--district', 
        required=False,
        default='all', 
        type=str, 
        help='District is the sub-region (ex. Uganda->Ibanda)'
    )
    args = parser.parse_args()
    out_dir =  os.path.join(args.outdir, args.region, args.district)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    download_sentinel2(
        out_dir, 
        args.bbox, 
        args.start_date, 
        args.end_date, 
        args.buffer
    )
