"""
Uploads all the composite files found in the specified input directory to s3 storage
"""
import os
from typing import List

from tqdm import tqdm

from src.api.sentinel2 import initialize_s3_bucket
from file_types import File


def upload_composites(comp_files: List[str], s3_bucket_name: str):
    s3 = initialize_s3_bucket(s3_bucket_name)

    for filename in tqdm(comp_files, leave=True, position=0):
        file = File.create(filename)

        if file is None:
            continue

        file_size = os.stat(filename).st_size
        key = file.s3_archive_path()
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, leave=False, position=1) as pbar:
            s3.upload_file(
                Filename=filename,
                Key=key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
