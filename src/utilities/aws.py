import os
import shutil

import boto3
import botocore.exceptions
import tqdm

from src.api.sentinel2 import APIAuth
from src.utilities.config_reader import CONFIG
from definitions import COMPOSITE_DIR
from file_types import OpticalComposite


def upload_to_s3(session, filename: str, key: str, bucket: str = None):
    """
    Uploads a file to S3 storage
    """
    file_size = os.stat(filename).st_size
    with tqdm.tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            desc=filename,
            leave=False,
            position=0
    ) as pbar:
        session.upload_file(
            Filename=filename,
            Bucket=CONFIG.AWS.BUCKET if bucket is None else bucket,
            Key=key,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )


def _no_iam_auth(bucket_name: str):
    s3 = boto3.resource(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        aws_session_token=os.environ['AWS_SESSION_TOKEN'] if 'AWS_SESSION_TOKEN' in os.environ else ''
    )
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError:
        raise ValueError(f'Invalid AWS credentials or bucket {bucket_name} does not exist.')
    return s3


def initialize_s3(bucket_name: str = CONFIG.AWS.BUCKET):

    # First try to authenticate with properly configured IAM profile
    try:
        s3 = boto3.resource('s3')
        s3.meta.client.head_bucket(Bucket=bucket_name)
        return boto3.client('s3'), True

    # If that doesn't work try to get credentials from ~/.aws/credentials file and start a new session
    except botocore.exceptions.NoCredentialsError:
        try:
            return _no_iam_auth(bucket_name), False
        except KeyError:
            APIAuth.parse_aws_credentials()
            try:
                return _no_iam_auth(bucket_name), False
            except KeyError:
                raise KeyError(
                    'Could not find AWS credentials in environment variables. Please either access s3 from an '
                    'EC2 instance with S3 IAM permissions or use the aws CLI to authenticate and set the '
                    'following environment variables:\n AWS_ACCESS_KEY_ID\n AWS_SECRET_ACCESS_KEY')


def rename_and_reupload(region: str, districts):
    files_to_upload = []
    for district in districts:
        dir = os.path.join(COMPOSITE_DIR, region, district)
        for file in os.listdir(dir):
            if 'multiband' in file:
                mgrs = file[:5]
                new_name = OpticalComposite(region, district, mgrs, ['B02', 'B03', 'B04'])
                aws_path = os.path.join(COMPOSITE_DIR, region, district, new_name.name)

                print(new_name.name)
                files_to_upload.append((os.path.join(dir, file), new_name.archive_path, aws_path))

    s3, client = initialize_s3('b2p.njr')
    if not client:
        s3_bucket = s3.Bucket('b2p.njr')
    else:
        s3_bucket = s3

    for filetuple in tqdm.tqdm(files_to_upload, leave=True, position=0):
        file_size = os.stat(filetuple[0]).st_size
        key = filetuple[2]
        with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=filetuple[0], leave=False, position=1) as pbar:
            s3_bucket.upload_file(
                Filename=filetuple[0],
                Bucket='b2p.njr',
                Key=key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
            shutil.move(filetuple[0], filetuple[1])
