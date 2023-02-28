import yaml
import os 
import boto3
import tqdm
from src.api.sentinel2 import SinergiseSentinelAPI
from src.utilities import imaging
import numpy as np
import subprocess

# define global consts
BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ), 
        '..'
    )
)
BANDS = ['B02', 'B03', 'B04']
DTYPE = np.float32
BUFFER = 100
SLICES = 12
N_CORES = 1
SESSION = boto3.Session()
S3 = SESSION.client('s3')
BUCKET = 'b2p.njr'
## define helpers
def upload_to_s3(filename, key):
    filesize = os.stat(filename).st_size
    with tqdm.tqdm(
        total=filesize, 
        unit='B', 
        unit_scale=True, 
        desc=filename, 
        leave=False, 
        position=0
    ) as pbar:
        S3.upload_file(
            Filename=filename,
            Bucket=BUCKET,
            Key=key,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )

## Start script
region_info_file = 'data/region_info.yaml'
with open(region_info_file, 'r') as file:
    region_info = yaml.safe_load(file)
api = SinergiseSentinelAPI()
print('======================================================')
print('Staring Scipt')
for region, info in region_info.items():
    dates = info['dates']
    if region != "Cote d'Ivoire": 
        continue
    for district, district_info in info['districts'].items():
        # if district_info['spec'] != 'train':
        #     continue
        bbox = district_info['bbox']
        print(f'{region}/{district}\n' 
            f'\tdates : {dates}\n'
            f'\tbbox : {bbox}') 
        # define src and dst dirs
        s2_dir = os.path.join(BASE_DIR, 'data', 'sentinel2', region, district)
        composite_dir = os.path.join(BASE_DIR, 'data', 'composites', region, district)
        os.makedirs(s2_dir, exist_ok=True)
        os.makedirs(composite_dir, exist_ok=True)
        # download data from s2 api
        print('--------------------------------')
        print('downloading sentinel2')
        for date in dates:
            if district == 'Kasese':
                continue
            start_date, end_date = date
            api.download(bbox, BUFFER, s2_dir, start_date, end_date)
        # create composite
        print('--------------------------------')
        print('creating composites')
        for coord in os.listdir(s2_dir):
            print(f'Creating composite for {region}/{district}/{coord}...')
            multiband_file_path = imaging.createComposite(
                s2_dir, 
                composite_dir, 
                coord, 
                BANDS, 
                DTYPE,
                SLICES,
                N_CORES>1
            )
            upload_to_s3(
                multiband_file_path, 
                os.path.join('composites', region, district, os.path.basename(multiband_file_path))
            )
        # Compress sentinel2
        print('-----------------------------')
        print('Compressing raw s2')
        tar_file =  os.path.join(BASE_DIR, 'data', 'sentinel2', f's2_{region}_{district}.tar.gz')
        tar_cmd = f'tar -czvf {tar_file} --remove-files {s2_dir}'
        process = subprocess.Popen(tar_cmd.split(), shell=False)
        process.communicate()
        # Upload raw s2 to s3
        upload_to_s3(
            tar_file, 
            os.path.join('sentinel2_raw', os.path.basename(tar_file))
        )
        # remove raw data from this vm
        os.remove(tar_file)
        print('============================================\n\n')