# This file will call the Sentinel2 API to download data for a specified region and then clean that data by making
# composites
import os
from argparse import ArgumentParser
from api.sentinel2 import SinergiseSentinelAPI
from composites.sentinel2 import create_composite


def download_and_make_composites(outdir, bounds, start_date, end_date, region_name, slices, buffer):
    api = SinergiseSentinelAPI()
    api.download(bounds, buffer, outdir, start_date, end_date)

    if not outdir:
        raise FileNotFoundError('No files returned from the query parameters')

    for coordinate_dir in outdir:
        in_out_dir = os.path.join(outdir, coordinate_dir)
        create_composite(in_out_dir, in_out_dir, region_name, slices)

    print(f'Wrote sentinel2 files and composites to {outdir}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--outdir', '-o', type=str, required=True, help='Where to write the sentinel2 files and'
                                                                        ' composites')
    parser.add_argument('--bbox', '-b', type=float, nargs=4, required=True,
                        help='Bounding box defined in lat / lon in the order min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_date', '-sd', type=str, required=True, help='Start date for Sentinel2 file query in '
                                                                             'YYYY-MM-DD')
    parser.add_argument('--end_date', '-ed', type=str, required=True, help='End date for Sentinel2 file query in '
                                                                           'YYYY-MM-DD')
    parser.add_argument('--region', '-r', type=str, required=True, help='Name of the region being queried for')
    parser.add_argument('--buffer', required=False, type=float, help='Buffer for bounding box query in meters')
    parser.add_argument('--slices', required=False, type=int, default=1,
                        help='Number of slices to split the tiles up into when making the composite. Default is 1')
    args = parser.parse_args()
    download_and_make_composites(args.outdir, args.bbox, args.start_date, args.end_date, args.region_name, args.slices,
                                 args.buffer)
