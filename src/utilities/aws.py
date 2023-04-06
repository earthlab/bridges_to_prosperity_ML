import os

import botocore.exceptions
import tqdm
import boto3

from src.utilities.config_reader import CONFIG
from src.api.sentinel2 import APIAuth


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
        return s3

    # If that doesn't work try to get credentials from ~/.aws/credentials file and start a new session
    except botocore.exceptions.NoCredentialsError:
        try:
            return _no_iam_auth(bucket_name)
        except KeyError:
            APIAuth.parse_aws_credentials()
            try:
                return _no_iam_auth(bucket_name)
            except KeyError:
                raise KeyError(
                    'Could not find AWS credentials in environment variables. Please either access s3 from an '
                    'EC2 instance with S3 IAM permissions or use the aws CLI to authenticate and set the '
                    'following environment variables:\n AWS_ACCESS_KEY_ID\n AWS_SECRET_ACCESS_KEY')
