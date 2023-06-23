from src.utilities.config_reader import CONFIG

import os
from typing import List, Type, Tuple
import pathlib

from tqdm import tqdm
import botocore
import boto3



def upload_files(files: List[Type], s3_bucket_name: str) -> None:
    """
    Uploads all the multivariate composite files found to s3 storage. Region, district, and military grid can be specified to narrow the search
    Args:
        s3_bucket_name (str): Name of the AWS S3 bucket name to upload the files to. Defualts to bucket in project configuration
        region (str): Name of the region to upload the composites for. If none specified, all regions are uploaded
        district (str): Name of the district to upload. If none specified, all districts are uploaded
        mgrs (list): List of military grid coordinates to upload. If none then all mgrs are uploaded
    """
    s3 = initialize_s3_bucket(s3_bucket_name)
    for file_object in tqdm(files, leave=True, position=0):
        if not hasattr(file_object, 's3_archive_path'):
            continue
        file_size = os.stat(file_object.archive_path).st_size
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_object.archive_path, leave=False, position=1) as pbar:
            s3.upload_file(
                Filename=file_object.archive_path,
                Key=file_object.s3_archive_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )


def s3_download_task(location_request_info: Tuple[str, int, int, str]) -> None:
    """
    Downloads file from s3 storage given s3 path and local destination path.
    Args:
        location_request_info (tuple):
            location_path (str): Path to composite object in s3 storage relative to the s3 bucket name
            composite_size (int): Size of the composite object in bytes
            destination (str): Local path where the downloaded composite object will be written to
            position (int): Position in the download queue
            bucket_name (str): Name of the s3 bucket that the composite object is in
    """
    bucket = initialize_s3_bucket(bucket_name)
    for file_path, destination, position, bucket_name in location_request_info:
        if os.path.exists(destination):
            return 
        obj = bucket.objects.filter(Prefix=file_path)[0]
        with tqdm(total=int(obj.size), unit='B', unit_scale=True, desc=file_path, leave=False,
                  position=int(position)) as pbar:
            bucket.download_file(Key=file_path, Filename=destination,
                                 Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    return


def list_s3_files(s3_dir: str, s3_bucket_name: str) -> List[str]:
    """ 
    """
    s3 = initialize_s3_bucket(s3_bucket_name)
    files = []
    for obj in s3.objects.filter(Prefix=s3_dir):
        files.append(obj.key)
    
    return files


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
    except (botocore.exceptions.NoCredentialsError, botocore.exceptions.ClientError):
        try:
            return _no_iam_auth_bucket(bucket_name)
        except KeyError:
            parse_aws_credentials()
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
        return boto3.client('s3')

    # If that doesn't work try to get credentials from ~/.aws/credentials file and start a new session
    except (botocore.exceptions.NoCredentialsError, botocore.exceptions.ClientError):
        try:
            return _no_iam_auth_client(bucket_name)
        except KeyError:
            parse_aws_credentials()
            try:
                return _no_iam_auth_client(bucket_name)
            except KeyError:
                raise KeyError(
                    'Could not find AWS credentials in environment variables. Please either access s3 from an '
                    'EC2 instance with S3 IAM permissions or use the aws CLI to authenticate and set the '
                    'following environment variables:\n AWS_ACCESS_KEY_ID\n AWS_SECRET_ACCESS_KEY')
            

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
