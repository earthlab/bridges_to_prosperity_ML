import os
import shutil

import tqdm

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
            Key=key,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )
