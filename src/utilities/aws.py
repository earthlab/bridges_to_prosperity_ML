import os
import shutil

import tqdm

from src.utilities.config_reader import CONFIG
from definitions import COMPOSITE_DIR
from file_types import OpticalComposite
from src.api.sentinel2 import APIAuth
import boto3


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
            Key=key,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )


def refresh_token():
    s = APIAuth('', no_auth=True)
    s.parse_aws_credentials()

    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        aws_session_token=os.environ['AWS_SESSION_TOKEN'],
        region_name='us-west2'
    )

    # Check the current expiration time of the session token
    print(session.get_credentials().expiry_time)

    # Refresh the session token
    session.get_credentials().refresh()

    # Check the new expiration time of the session token
    print(session.get_credentials().expiry_time)
