import glob
import os
from glob import glob

import boto3
from tqdm import tqdm


def sync_s3():
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)
            ), 
            '..'
        )
    )
    comp_files = glob(os.path.join(base_dir, "data/composites/**/*_multiband.tiff" ), recursive=True)
    session = boto3.Session()
    s3 = session.client('s3')
    bucket = 'b2p.njr'
    for filename in tqdm(comp_files, leave=True, position=0):
        filesize = os.stat(filename).st_size
        key = filename.split('data/')[1]
        with tqdm(total=filesize, unit='B', unit_scale=True, desc=filename, leave=False, position=1) as pbar:
            s3.upload_file(
                Filename=filename,
                Bucket=bucket,
                Key=key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
if __name__ == "__main__":
    sync_s3()
    

