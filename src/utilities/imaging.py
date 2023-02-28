from typing import Union
import numpy as np

import os
import shutil
from glob import glob 
import rasterio
from rasterio import features
from rasterio.windows import Window
import geopandas as gpd
import copy
from tqdm.notebook import tqdm 
import warnings
BANDS_TO_IX = {
    'B02': 3, # Blue
    'B03': 2, # Green
    'B04': 1 # Red
}
MAX_PIXEL_VAL = 4000
# the max_pixel_val was set to 2500? this probably should match with the max allowed value for clouds or it shouldnt be included...
def scale(
    x:Union[float, np.array], 
    max_pixel_val:float = MAX_PIXEL_VAL
) -> Union[float, np.array]:
    return x / max_pixel_val * 255

def getImgFromFile(img_path, g_ncols, dtype, row_bound=None):
    img = rasterio.open(img_path, driver='JP2OpenJPEG')
    ncols, nrows = img.meta['width'], img.meta['height']
    assert g_ncols == ncols, f'Imgs have different size ncols: {ncols} neq {g_ncols}'
    if row_bound is None:
        pixels = img.read(1).astype(np.float32)
    else:
        pixels = img.read(
            1,
            window=Window.from_slices(
                        slice(row_bound[0], row_bound[1]),
                        slice(0, ncols)
                    )
        ).astype(dtype)
    return pixels

def getCloudMaskFromFile(cloud_path, crs, transform, shape, row_bound=None):
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    # filter out RuntimeWarnings, due to geopandas/fiona read file spam
    # https://stackoverflow.com/questions/64995369/geopandas-warning-on-read-file
    try: 
        cloud_file = gpd.read_file(cloud_path)
        cloud_file.crs = (str(crs))
        # convert the cloud mask data to a raster that has the same shape and transformation as the
        # img raster data
        cloud_img = features.rasterize(
            (
                (g['geometry'], 1) for v, g in cloud_file.iterrows()
            ),
            out_shape=shape,
            transform=transform, 
            all_touched=True
        )
        if row_bound is None:
            return np.where(cloud_img == 0, 1, 0)
        return np.where(cloud_img[row_bound[0]:row_bound[1],:] == 0, 1, 0)
    except: 
        return None

def nanClouds(pixels, cloud_channels, max_pixel_val:float=MAX_PIXEL_VAL):
    cp = pixels * cloud_channels
    mask = np.where(np.logical_or(cp == 0, cp > max_pixel_val ))
    cp[mask] = np.nan
    return cp

def createComposite(s2_dir:str, composite_dir:str, coord:str, bands:list, dtype:type, num_slices:int=1, pbar:bool=True):
    assert os.path.isdir(s2_dir)
    os.makedirs(composite_dir, exist_ok=True)
    if num_slices > 1:
        os.makedirs(os.path.join(composite_dir, coord), exist_ok=True)
    multiband_file_path = os.path.join(composite_dir, f'{coord}_multiband.tiff')
    if os.path.isfile(multiband_file_path): 
        return multiband_file_path
    # Loop through each band, getting a median estimate for each
    crs = None
    transform = None
    for band in tqdm(bands, desc=f'Processing {coord}', leave=True, position=1, total=len(bands), disable=pbar):
        band_files = glob(os.path.join(s2_dir, coord, f'**/*{band}*.jp2'))
        assert len(band_files) > 1, f'{os.path.join(s2_dir, coord)}'
        with rasterio.open(band_files[0], 'r', driver='JP2OpenJPEG') as rf:
            g_nrows, g_ncols = rf.meta['width'], rf.meta['height']
            crs = rf.crs
            transform = rf.transform
        # handel slicing if necessary, slicing along rows only
        if num_slices > 1:
            slice_width = g_nrows / num_slices
            slice_end_pts = [int(i) for i in np.arange(0, g_nrows + slice_width, slice_width)]
            slice_bounds = [ (slice_end_pts[i], slice_end_pts[i+1]-1)  for i in range(num_slices-1)]
            slice_bounds.append((slice_end_pts[-2], slice_end_pts[-1]))
        else: 
            slice_bounds = [None]
        joined_file_path = os.path.join(composite_dir, coord, f'{band}.tiff')
        if os.path.isfile(joined_file_path): 
            continue
        ## Median across time, slicing if necessary
        for k, row_bound in tqdm(enumerate(slice_bounds), desc=f'band={band}', total=num_slices, position=2, disable=pbar):
            if num_slices > 1:
                slice_file_path = os.path.join(composite_dir, coord, f'{band}_slice|{row_bound[0]}|{row_bound[1]}|.tiff')
            else:
                slice_file_path = joined_file_path
            if os.path.isfile(slice_file_path): 
                continue
            cloud_correct_imgs = []
            for img_path in tqdm(band_files, desc=f'slice {k+1}',leave=False, position=3, disable=pbar):
                # Get data from files
                pixels = getImgFromFile(img_path, g_ncols, dtype, row_bound)
                date_dir = os.path.dirname(img_path)
                cloud_files = glob(os.path.join(date_dir, "*.gml"))
                assert len(cloud_files) == 1, f'Cloud path does not exist for {date_dir}'

                cloud_channels = getCloudMaskFromFile(cloud_files[0], crs, transform, (g_nrows, g_ncols), row_bound)
                if cloud_channels is None: continue
                # add to list to do median filte later
                cloud_correct_imgs.append(nanClouds(pixels, cloud_channels) ) 
                del pixels

            corrected_stack = np.vstack([img.ravel() for img in cloud_correct_imgs])
            median_corrected = np.nanmedian(corrected_stack, axis=0, overwrite_input=True)
            median_corrected = median_corrected.reshape(cloud_correct_imgs[0].shape)

            with rasterio.open(slice_file_path, 'w', driver='Gtiff', width=g_ncols, height=g_nrows, count=1, crs=crs,
                                       transform=transform, dtype=dtype) as wf:
                wf.write(median_corrected.astype(dtype), 1)
            # release mem
            median_corrected = []
            del median_corrected
            corrected_stack = []
            del corrected_stack
        ## Combine slices 
        if num_slices > 1:
            with rasterio.open(joined_file_path, 'w', driver='GTiff', width=g_ncols, height=g_nrows, count=1, crs=crs,
                                   transform=transform, dtype=dtype) as wf:
                for slice_file_path in glob(os.path.join(composite_dir, coord, f'{band}_slice*.tiff')):
                    prts = slice_file_path.split('|')
                    left = int(prts[1])
                    right = int(prts[2])
                    with rasterio.open(slice_file_path, 'r', driver='GTiff') as rf: 
                        wf.write(
                            rf.read(1), 
                            window=Window.from_slices(
                                slice(left, right), 
                                slice(0, g_ncols)
                            ),
                            indexes=1
                        )
                    os.remove(slice_file_path)
    ## Combine Bands
    n_bands = len(bands)
    with rasterio.open(
        multiband_file_path, 
        'w',
        driver='GTiff', 
        width=g_ncols, 
        height=g_nrows, 
        count=n_bands, 
        crs=crs, 
        transform=transform, 
        dtype=dtype
    ) as wf: 
        for band in tqdm(bands, total=n_bands, desc='Combining bands...', leave=False, position=1, disable=pbar):
            j = BANDS_TO_IX[bands]
            with rasterio.open(os.path.join(composite_dir, coord, f'{band}.tiff'), 'r', driver='GTiff') as rf:
                wf.write(rf.read(1), indexes=j)
    shutil.rmtree(os.path.join(composite_dir, coord))
    return multiband_file_path
    
def flipRGB(infile, outfile, dtype = np.float32): 
    crs, transform, g_ncols, g_nrows, = None, None, None, None
    b02, b03, b04 = None, None, None
    with rasterio.open(
        infile, 
        'r',
        driver='GTiff', 
    ) as rf: 
        g_nrows, g_ncols = rf.meta['width'], rf.meta['height']
        crs = rf.crs
        transform = rf.transform
        b02 = rf.read(1)
        b03 = rf.read(2)
        b04 = rf.read(3)
    with rasterio.open(
        outfile, 
        'w',
        driver='GTiff', 
        width=g_ncols, 
        height=g_nrows, 
        count=3, 
        crs=crs, 
        transform=transform, 
        dtype=dtype
    ) as wf: 
        wf.write(b04, indexes=1)
        wf.write(b03, indexes=2)
        wf.write(b02, indexes=3)
    return None