import getpass
import botocore
import multiprocessing as mp
import os
import pathlib
import re
import subprocess as sp
from argparse import Namespace
from datetime import datetime, timedelta
from typing import List, Tuple

import boto3
import geojson
import tqdm
from shapely.geometry import Polygon
from src.utilities.config_reader import CONFIG

from file_types import Sentinel2Tile, Sentinel2Cloud

PATH = os.path.dirname(__file__)


def _no_iam_auth_bucket(bucket_name: str):
    try:
        s3_resource = boto3.resource(
            's3',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            aws_session_token=os.environ['AWS_SESSION_TOKEN'] if 'AWS_SESSION_TOKEN' in os.environ else ''
        )
        s3_resource.meta.client.head_bucket(Bucket=bucket_name)
        return s3_resource.Bucket(bucket_name)
    except botocore.exceptions.ClientError:
        raise ValueError(f'Invalid AWS credentials or bucket {bucket_name} does not exist.')


def _no_iam_auth_client(bucket_name: str):
    try:
        s3_resource = boto3.resource(
            's3',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            aws_session_token=os.environ['AWS_SESSION_TOKEN'] if 'AWS_SESSION_TOKEN' in os.environ else ''
        )
        s3_resource.meta.client.head_bucket(Bucket=bucket_name)
        return boto3.client(
            's3',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            aws_session_token=os.environ['AWS_SESSION_TOKEN'] if 'AWS_SESSION_TOKEN' in os.environ else ''
        )
    except botocore.exceptions.ClientError:
        raise ValueError(f'Invalid AWS credentials or bucket {bucket_name} does not exist.')


def initialize_s3_bucket(bucket_name: str = CONFIG.AWS.BUCKET):
    # First try to authenticate with properly configured IAM profile
    try:
        s3 = boto3.resource('s3')
        s3.meta.client.head_bucket(Bucket=bucket_name)
        return boto3.resource('s3').Bucket(bucket_name)

    # If that doesn't work try to get credentials from ~/.aws/credentials file and start a new session
    except botocore.exceptions.NoCredentialsError:
        try:
            return _no_iam_auth_bucket(bucket_name)
        except KeyError:
            APIAuth.parse_aws_credentials()
            try:
                return _no_iam_auth_bucket(bucket_name)
            except KeyError:
                raise KeyError(
                    'Could not find AWS credentials in environment variables. Please either access s3 from an '
                    'EC2 instance with S3 IAM permissions or use the aws CLI to authenticate and set the '
                    'following environment variables:\n AWS_ACCESS_KEY_ID\n AWS_SECRET_ACCESS_KEY')


def initialize_s3_client(bucket_name: str = CONFIG.AWS.BUCKET):
    # First try to authenticate with properly configured IAM profile
    try:
        s3 = boto3.resource('s3')
        s3.meta.client.head_bucket(Bucket=bucket_name)
        return boto3.client('s3'), True

    # If that doesn't work try to get credentials from ~/.aws/credentials file and start a new session
    except botocore.exceptions.NoCredentialsError:
        try:
            return _no_iam_auth_client(bucket_name)
        except KeyError:
            APIAuth.parse_aws_credentials()
            try:
                return _no_iam_auth_client(bucket_name)
            except KeyError:
                raise KeyError(
                    'Could not find AWS credentials in environment variables. Please either access s3 from an '
                    'EC2 instance with S3 IAM permissions or use the aws CLI to authenticate and set the '
                    'following environment variables:\n AWS_ACCESS_KEY_ID\n AWS_SECRET_ACCESS_KEY')


def _download_task(namespace: Namespace) -> None:
    """
    Downloads a single file from the indicated s3 bucket. This function is intended to be spawned in parallel from the
    parent process.
    Args:
        namespace (Namespace): Contains the bucket name, s3 file name, and destination required for s3 download request.
        Each value in the namespace must be a pickle-izable type (i.e. str).
    """
    s3 = initialize_s3_client(namespace.bucket_name)
    os.makedirs(namespace.outdir, exist_ok=True)
    dst = os.path.join(namespace.outdir, namespace.available_file.replace('/', '_'))
    if os.path.isfile(dst):
        return
    s3.download_file(namespace.bucket_name, namespace.available_file,
                     dst,
                     ExtraArgs={'RequestPayer': 'requester'}
                     )


class APIAuth:
    """
    Contains methods for creating temporary auth credentials through saml2aws. Currently supports all Earth Lab AWS
    accounts. saml2aws authentication is not necessary when making requests from an EC2 instance with the properly
    configured IAM profile.
    Ex.,
        auth = APIAuth()
        s3 = auth.login(aws_account='b2p')

    """
    SAML2AWS_INSTALL_ERROR_STR = "Error configuring saml2aws. Is it installed? Check for installation by running" \
                                 " 'saml2aws version'\n If not installed follow instructions at " \
                                 "https://curc.readthedocs.io/en/iaasce-954_grouper/cloud/aws/getting-started/" \
                                 "aws-cli-saml2aws.html"
    MAX_SESSION_DURATION = 7200

    ACCOUNT_TO_ARN = {
        'b2p': 'arn:aws:iam::120656651053:role/Shibboleth-Customer-Admin',
        'career': 'arn:aws:iam::031843903858:role/Shibboleth-Customer-Admin',
        'cnh': 'arn:aws:iam::756129716358:role/Shibboleth-Customer-Admin',
        'deloitte': 'arn:aws:iam::819721361220:role/Shibboleth-Customer-Admin',
        'grandchallenge': 'arn:aws:iam::325229007483:role/Shibboleth-Customer-Admin',
        'jsfp': 'arn:aws:iam::276588439095:role/Shibboleth-Customer-Admin',
        'macrosystems': 'arn:aws:iam::973137535535:role/Shibboleth-Customer-Admin',
        'nccasc': 'arn:aws:iam::454742582098:role/Shibboleth-Customer-Admin',
        'opp': 'arn:aws:iam::080475411655:role/Shibboleth-Customer-Admin'
    }

    def __init__(self, identikey: str, no_auth: bool = False) -> None:
        """
        Queries the user for identikey password and configures saml2aws for CU AWS authentication
        Args:
            identikey (str): CU identikey used to log into AWS accounts
        """
        if not no_auth:
            self._identikey = identikey
            self._password = getpass.getpass('identikey password:')
            self._configure_saml2aws()

    def login(self, aws_account: str, ttl: int = MAX_SESSION_DURATION) -> boto3.client:
        """
        Creates a temporary credential to the specified AWS account and returns a boto3 s3 client object that can be
        used to make requests.
        Args:
            aws_account (str): Which AWS account to log in to. Must be a key in the ACCOUNT_TO_ARN dictionary
            ttl (int): How long the credentials will be valid for in seconds. Max is 7200 (2 hours)
        Returns:
            (boto3.client): AWS client objects that can be used to make requests to the account
        """
        self._configure_session_ttl(ttl)
        try:
            if aws_account not in self.ACCOUNT_TO_ARN.keys():
                raise ValueError(f'{aws_account} is not a valid account string. Please input one of the following:\n '
                                 f'{self.ACCOUNT_TO_ARN.keys()}')

            sp.call([
                'saml2aws', 'login', f'--username={self._identikey}', f'--password={self._password}',
                f'--role={self.ACCOUNT_TO_ARN[aws_account]}', '--skip-prompt', '--force'
            ])

            self.parse_aws_credentials()

            return boto3.client(
                's3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                aws_session_token=os.environ['AWS_SESSION_TOKEN']
            )

        except FileNotFoundError:
            print(self.SAML2AWS_INSTALL_ERROR_STR)
            return

    @staticmethod
    def parse_aws_credentials():
        with open(os.path.join(pathlib.Path().home(), '.aws', 'credentials'), 'r') as f:
            start = False

            f1 = False
            f2 = False
            f3 = False
            for line in f.readlines():
                if line == '[saml]\n':
                    start = True
                if not start:
                    continue

                if line.startswith('aws_access_key_id'):
                    os.environ['AWS_ACCESS_KEY_ID'] = str(line.split('= ')[1].strip('\n'))
                    f1 = True
                elif line.startswith('aws_secret_access_key'):
                    os.environ['AWS_SECRET_ACCESS_KEY'] = str(line.split('= ')[1].strip('\n'))
                    f2 = True
                elif line.startswith('aws_security_token'):
                    os.environ['AWS_SESSION_TOKEN'] = str(line.split('= ')[1].strip('\n'))
                    f3 = True

                if f1 and f2 and f3:
                    break

    def _configure_session_ttl(self, ttl: int) -> None:
        """
        Sets how long the login credentials will be valid for in the ~/.saml2aws configuration file.
        Args:
             ttl (int): How long the login credentials will be valid for in seconds. Max is 7200 (2 hours)
        """
        if ttl > self.MAX_SESSION_DURATION:
            print(f'Requested TTL exceeds max of {self.MAX_SESSION_DURATION}. Using max duration.')
            ttl = self.MAX_SESSION_DURATION

        saml2aws_path = os.path.join(pathlib.Path.home(), '.saml2aws')
        if not os.path.exists(saml2aws_path):
            print(f'Could not find .saml2aws file at {saml2aws_path}. Is saml2aws installed? If not installed follow'
                  f' instructions at "\
                  "https://curc.readthedocs.io/en/iaasce-954_grouper/cloud/aws/getting-started/aws-cli-saml2aws.html"')
            return

        with open(saml2aws_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('aws_session_duration'):
                lines[i] = f'aws_session_duration    = {ttl}\n'

        with open(saml2aws_path, 'w') as f:
            f.writelines(lines)

        print(f'Set ttl to {ttl} seconds')

    def _configure_saml2aws(self):
        """
        Sets the saml2aws configuration for authenticating with CU AWS accounts
        """
        try:
            sp.call([
                'saml2aws', 'configure', '--idp-provider=ShibbolethECP', '--mfa=push',
                '--url=https://fedauth.colorado.edu/idp/profile/SAML2/SOAP/ECP',
                f'--username={self._identikey}', f'--password={self._password}', '--skip-prompt'
            ])
        except FileNotFoundError:
            print(self.SAML2AWS_INSTALL_ERROR_STR)


class SinergiseSentinelAPI:
    """
    Contains methods for downloading data from the Sinergise Sentinel2 LC1 data hosted on AWS. On EC2 instances with
    the proper IAM configuration, authentication is handled automatically. From any other platform, users must provide
    credentials.
    Ex.,
        api = SinergiseSentinelAPI()
        api.download([27.876, -45.678, 28.012, -45.165], 1000, '/example/outdir', '2021-01-01', '2021-02-01')
    """
    CLOUD_PATTERN = r'tiles/(?P<utm_code>\d+)/(?P<latitude_band>\S+)/(?P<square>\S+)/(?P<year>\d{4})/(?P<month>\d+)/(?P<day>\d+)/0/qi/MSK_CLOUDS_B00.gml.*'

    def __init__(self, identikey: str = None) -> None:
        """
        Creates a boto3 client object for making requests. If using a CU AWS account the user's identikey can be input
        to create temporary credentials.
            Args:
                identikey (str): If not None, will be used to create temporary credentials for the CU AWS B2P account.
        """
        self._bucket_name = 'sentinel-s2-l1c'

        # When the API is not used from an EC2 instance with the proper IAM profile configured credentials need to be
        # created
        if identikey is not None:
            auth = APIAuth(identikey=identikey)
            self._s3 = auth.login(aws_account='b2p')
        else:
            self._s3 = initialize_s3_client()

    def download(self, bounds: List[float], buffer: float, outdir: str, start_date: str, end_date: str,
                 bands=None) -> None:
        """
        Downloads a list of .jp2 files from the Sinergise Sentinel2 LC1 bucket given a bounding box defined in lat/long,
         a buffer in meters, and a start and end date . Only Bands 2-4 are requested.
         Args:
             bounds (list): Bounding box defined in lat / lon [min_lon, min_lat, max_lon, max_lat]
             buffer (float): Amount by which to extend the bounding box by, in meters
             outdir (str): Directory where requested files will be written to
             start_date (str): Beginning of requested data creation date YYYY-MM-DD
             end_date (str): End of requested data creation date YYYY-MM-DD
             bands (list): The bands to download for each file. Default is ['B02', 'B03', 'B04', 'B08'] for R, G, B, and
              near wave IR, respectively
        """
        # Convert the buffer from meters to degrees lat/long at the equator
        if bands is None:
            bands = ['B02', 'B03', 'B04', 'B08']
        buffer /= 111000

        # Adjust the bounding box to include the buffer (subtract from min lat/long values, add to max lat/long values)
        bounds[0] -= buffer
        bounds[1] -= buffer
        bounds[2] += buffer
        bounds[3] += buffer

        os.makedirs(outdir, exist_ok=True)

        available_files = self._find_available_files(bounds, start_date, end_date, bands)
        total_data = 0
        for file in available_files:
            total_data += file[1]
        total_data /= 1E9

        args = []
        for file_info in available_files:
            file_path = file_info[0]
            if '/preview/' in file_path:
                continue
            print(file_path)
            file = Sentinel2Tile.create(file_path)
            print(file)
            if file is None:
                file = Sentinel2Cloud.create(file_path)
                print(file)
                if file is None:
                    continue

            file_dir = os.path.join(outdir, file.mgrs_grid, file.date_str)

            # Skip if file is already local
            if os.path.exists(os.path.join(file_dir, os.path.basename(file_path))):
                continue

            args.append(Namespace(available_file=file_path, bucket_name=self._bucket_name, outdir=file_dir))

        # proceed = input(f'Found {len(args)} files for download. Total size of files is'
        #                 f' {round(total_data, 2)}GB and estimated cost will be ${round(0.09 * total_data, 2)}'
        #                 f'. Proceed (y/n)?')
        print(f'Found {len(args)} files for download. Total size of files is'
              f' {round(total_data, 2)}GB and estimated cost will be ${round(0.09 * total_data, 2)}'
              )
        # if proceed == 'y':
        with mp.Pool(mp.cpu_count() - 1) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(_download_task, args), total=len(args)):
                pass

    def _find_available_files(self, bounds: List[float], start_date: str, end_date: str,
                              bands: List[str]) -> List[Tuple[str, str]]:
        """
        Given a bounding box and start / end date, finds which files are available on the bucket and meet the search
        criteria
        Args:
            bounds (list): Lower left and top right corner of bounding box defined in lat / lon [min_lon, min_lat,
            max_lon, max_lat]
            start_date (str): Beginning of requested data creation date YYYY-MM-DD
            end_date (str): End of requested data creation date YYYY-MM-DD
        """
        ref_date = self._str_to_datetime(start_date)
        date_paths = []
        while ref_date <= self._str_to_datetime(end_date):
            tt = ref_date.timetuple()
            date_paths.append(f'/{tt.tm_year}/{tt.tm_mon}/{tt.tm_mday}/')
            ref_date = ref_date + timedelta(days=1)

        info = []
        mgrs_grids = self._find_overlapping_mgrs(bounds)

        for grid_string in mgrs_grids:
            utm_code = grid_string[:2]
            latitude_band = grid_string[2]
            square = grid_string[3:5]
            grid = f'tiles/{utm_code}/{latitude_band}/{square}'
            response = self._s3.list_objects_v2(
                Bucket=self._bucket_name,
                Prefix=grid + '/',
                MaxKeys=300,
                RequestPayer='requester'
            )
            if 'Contents' not in list(response.keys()):
                continue

            for date in date_paths:
                response = self._s3.list_objects_v2(
                    Bucket=self._bucket_name,
                    Prefix=grid + date + '0/',  # '0/' is for the sequence, which in most cases will be 0
                    MaxKeys=100,
                    RequestPayer='requester'
                )
                # TODO: Maybe use file type class creation here
                if 'Contents' in list(response.keys()):
                    info += [
                        (v['Key'], v['Size']) for v in response['Contents'] if
                        any([band + '.jp2' in v['Key'] for band in bands]) or 'MSK_CLOUDS_B00.gml' in v['Key']
                    ]

        return info

    @staticmethod
    def _find_overlapping_mgrs(bounds: List[float]) -> List[str]:
        """
        Files in the Sinergise Sentinel2 S3 bucket are organized by which military grid they overlap. Thus, the
        military grids that overlap the input bounding box defined in lat / lon must be found. A lookup table that
        includes each grid name and its geometry is used to find the overlapping grids.
        """
        print('Finding overlapping tiles... ')
        input_bounds = Polygon([(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]),
                                (bounds[0], bounds[3]), (bounds[0], bounds[1])])
        with open(os.path.join(PATH, 'mgrs_lookup.geojson'), 'r') as f:
            ft = geojson.load(f)
            return [i['properties']['mgs'] for i in ft[1:] if
                    input_bounds.intersects(Polygon(i['geometry']['coordinates'][0]))]

    @staticmethod
    def _str_to_datetime(date: str):
        return datetime.strptime(date, '%Y-%m-%d')
