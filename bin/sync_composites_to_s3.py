import boto3
import glob
import os
import tqdm
def sync_s3():
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)
            ), 
            '..'
        )
    )
    comp_files = glob.glob(os.path.join(base_dir, "**/*multiband_cld_NAN_median_corrected*.tiff"),recursive=True)
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('b2p.njr')
    pbar = tqdm.tqdm(comp_files)
    for cF in pbar:
        pbar.set_description(os.path.basename(cF))
        pbar.refresh()
        r = cF.split('sentinel2/')[1].split('/')[0]
        oF = os.path.join('composites', r, os.path.basename(cF))
        with open(cF, 'rb') as data:
            bucket.put_object(Body=data, Key=oF)
if __name__ == "__main__":
    sync_s3()
    

