import os
import tqdm

from config_reader import CONFIG


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
