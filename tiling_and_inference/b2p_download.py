# this is the list of necessary import statements
# note that in addition to these statements, you will need to run !pip install boto3, !pip install sentinelsat, !pip install raster2xyz, and !pip install fastai==1.0.61 in order to run the following code
import geopandas as gpd
from shapely.geometry import Polygon
from collections import OrderedDict
from sentinelsat import SentinelAPI
import zipfile
import os
import numpy as np
import pandas as pd
import shapely
import rasterio
import gdal
import shutil
import boto3
import sentinelsat
import rioxarray
from rasterio import features
from rasterio.windows import Window
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack
from fastai.vision import *
from glob import glob
from raster2xyz.raster2xyz import Raster2xyz
import time

def download_sent(bounds, buffer, mynewmap, dire, login, password, startdate, enddate):
    '''
    Download Sentinel2 data using the Sentinel API

    Arguments:
    bounds: bounding box for the desired region stored as a list in the form [min lon, min lat, max lon, max lat]
    buffer: the buffer to put around the bounding box in meters
    mynewmap: the shp file containing the Sentinel master key file
    dire: the path to your home directory
    login: a string containing your Copernicus Open Access Hub username
    password: a string containing your Copernicus Open Access Hub password
    start_dt: a string containing the start date for which you want Sentinel2 data in YYYYMMDD format
    end_dt: a string containing the end date for which you want Sentinel2 data in YYYYMMDD format

    Returns:
    Nothing, but downloads the requested Sentinel2 data as .zip files to a sentinel_folder off of your home directory
    '''
    # read in the Sentinel2 master shp file to a geopandas data frame
    sent = gpd.read_file(mynewmap)
    
    # convert the buffer from meters to degrees lat/long at the equator
    buffer /= 111000
    
    # adjust the bounding box to include the buffer (subtract from min lat/long values, add to max lat/long values)
    bounds[0] -= buffer
    bounds[1] -= buffer
    bounds[2] += buffer
    bounds[3] += buffer
    
    # create a Polygon of the bounding box from bottom left to top left to top right to bottom right
    bound= Polygon(((bounds[0],bounds[1]),(bounds[0],bounds[3]),(bounds[2],bounds[3]),(bounds[2],bounds[1])))
    tile_extr=[]
    # iterate over the rows in the Sentinel master geopandas data frame
    for i,y in sent.iterrows():
        # if the geometry of the current row in the Sentinel master key is in the bounding box, add the row
        # of the Sentinel master key to the list of tiles to download from Sentinel
        if bound.contains(y.geometry):
            tile_extr.append(y)

    # clear the memory from the Sentinel master key file
    del sent

    # change to the current (home) directory
    os.chdir(dire)

    # make a folder to store the downloaded Sentinel data
    aoi_name = 'folder'
    aoi_folder = f'sentinel_{aoi_name}'

  
    if not os.path.exists(aoi_folder):
        os.makedirs(aoi_folder)

    # change into the folder that will hold the downloaded Sentinel data
    os.chdir(os.getcwd()+'/'+aoi_folder)

    # log into the Sentinel API
    api = SentinelAPI(str(login), str(password))

    # set up the arguments for the Sentinel query - ask for Sentinel2 data over the desired date range
    query_kwargs = {
        'platformname': 'Sentinel-2',
        'producttype': 'S2MSI1C',
        'date': (str(startdate),str(enddate))} 

    # this will store the dictionary of Sentinel2 products to download
    products = OrderedDict()
    
    # iterate over the tiles for which there is Sentinel2 data in the bounding box
    for tile in tile_extr: 
        kw = query_kwargs.copy()
        # add the tile name to the arguments for the query
        kw['tileid'] = tile.Name
        # use the query arugments to query the Sentinel API and store the query results in the pp variable
        pp = api.query(**kw)
        # add the products from the Sentinel query to the ordered dictionary of products we want to download
        products.update(pp)

    # download all the products from the Sentinel query over the desired date range and bounding box at once
    api.download_all(products)

    # clear up memory
    del tile_extr, products, query_kwargs, kw, pp

    return

def unzip(dire):
    '''
    This function unzips the zip files from Sentinel2 that were previously downloaded

    Argument:
    dire: the directory containing the downloaded zip files from Sentinel2

    Returns:
    Nothing, but unzips and extracts the contents of each downloaded zip file from Sentinel2 in dire
    '''

    # change to the current directory (in this case the sentinel_folder directory containing downloaded Sentinel2 data)
    os.chdir(dire)
    
    # list the files in the current directory
    zipped = os.listdir()

    # iterate over the files in the current directory
    for i in zipped:
        # if the current file is a zip file, try to unzip the zip file and extract its contents to the current directory; otherwise, skip the zip file
        if '.zip' in i:
            try:
                zip_ref= zipfile.ZipFile(os.getcwd()+'/'+str(i),'r') 
                zip_ref.extractall(os.getcwd())
                zip_ref.close()
            
            except:
                continue
    
    # clear up memory
    del zipped, i, zip_ref

    return

def list_files(dire):
    '''
    This function finds the paths to each of the image data files that were downloaded from Sentinel2 and the paths to their corresponding cloud mask data files

    Arguments:
    dire: the directionary in which the downloaded and unzipped Sentinel2 data is stored

    Returns:
    a tuple containing two values, finalpth and cloud_pth 
        finalpth: a list of paths to the image data for each downloaded Sentinel2 file
        cloud_pth: a list of paths to the cloud mask file for each downloaded Sentinel2 file
    '''

    # list the files in the current directory
    names = os.listdir(dire)

    # change into the current directory
    os.chdir(dire)  

    namepaths=[]
    # iterate over the files in the current directory
    for i in names:
        # if we are looking at a folder of extracted Sentinel2 data, find the path to its GRANULE folder and
        # append that path to the list of namepaths
        if '.zip' not in i:
            if 'ipynb' not in i:
                namepaths.append(os.getcwd()+'/'+str(i)+'/GRANULE/')

    finalpth=[]
    # iterate over all the paths to the extracted Sentinel2 folder GRANULE subfolders and add the path to the
    # IMG_DATA subfolder within the GRANULE folder to the list of finalpaths
    for i in namepaths:
        temp = os.listdir(i)[0]
        finalpth.append(i+os.listdir(i)[0]+'/IMG_DATA/')

    cloud_pth = []
    # iterate over the paths to the extracted Sentinel2 folder GRANULE subfolders and find the path to the cloud
    # mask file and append its path to the list of cloud paths
    for i in namepaths:
        temp = os.listdir(i)[0]
        cloud_pth.append(i+os.listdir(i)[0]+'/QI_DATA/'+'MSK_CLOUDS_B00.gml')

    # clean up memory
    del names, namepaths, temp
  
    # return the list of paths to the IMG_DATA for all Sentinel2 zip files and the list of paths to the
    # cloud mask data for all Sentinel2 zip files
    return(finalpth , cloud_pth)

def dict_names(list_ofnames):    
    band_med={} #each band has a unique name that contains the date and the index of the tile , save them in a dictionary
  
    # iterate over all of the paths to the IMG_DATA folders from the extracted Sentinel2 zip files
    for imagepath in list_ofnames:
        # iterate over all of the files in the current IMG_DATA folder
        for y in os.listdir(imagepath):
      
          # if we are looking at band 2, open the current IMG_DATA file with a JPEG driver and add that 
          # open file to the dictionary of image files with the file name as its key
          if 'B02' in y:
            band2 = rasterio.open(imagepath+str(y), driver='JP2OpenJPEG')
            band_med[y]=band2
            #band2.close()
            #del band2
        
          # if we are looking at band 3, open the current IMG_DATA file with a JPEG driver and add that open
          # file to the dictionary of image files with the file name as its key
          if 'B03' in y:
            band3 = rasterio.open(imagepath+str(y), driver='JP2OpenJPEG') #green
            band_med[y] = band3
            #band3.close()
            #del band3
      
          # if we are looking at band 4, open the current IMG_DATA file with a JPEG driver and add that open
          # file to the dictionary of image files with the file name as its key
          if 'B04' in y:
            band4 = rasterio.open(imagepath+str(y), driver='JP2OpenJPEG') #red
            band_med[y] = band4
            #band4.close()
            #del band4

    # find the list of Sentinel2 index values, which is a 3-letter abbreviation, for all the  downloaded Sentinel2 files
    index = [i.split('_')[0][3:6] for i in list(band_med.keys())]
    # find the list of unique Sentinel2 3-letter index abbreviations
    index = list(np.unique(index))

    # return the dictionary of file name and open image file key:value pairs for the Sentinel2 data that was 
    # downloaded and the list of unique 3-letter Sentinel2 index abbreviations
    return(band_med, index)

def get_median_slices(indexes, dictionary_of_bands, clouds_pth, dire, band, slices):
    # change to the current directory (here the home directory)
    os.chdir(dire)

    # create a bands_folder that will hold the processed Sentinel2 satellite imagery
    aoi_name = 'folder' 
    aoi_folder = f'bands_{aoi_name}'

    # make the folder if it does not already exist
    if not os.path.exists(aoi_folder):
        os.makedirs(aoi_folder)

    # change into the folder that will hold the processed Sentinel2 satellite imagery
    os.chdir(os.getcwd()+'/'+aoi_folder) 
   
    # here we are defining slices of the Sentinel2 image data so that we don't have to process the image files
    # in their entirety
    # Note: This is completely optional
    towindow = slices
    mediocre=[]
    aditi = 0
    mediocre.append(int(aditi))
    for i in range(towindow):
        aditi+= 10980/towindow
        mediocre.append(int(aditi))
    
    # store a list of strings that will give us progress updates about which slice we are working on and how
    # many slices we have in total
    jojo = [str(i)+'of'+str(towindow-1) for i in range(len(mediocre)-1)]

    windowed = mediocre 
    nam = jojo

    # iterate over the 3-letter Sentinel2 location abbreviations
    for z in indexes:
        # iterate over all of the slices of data that we are working on
        for gg in range(len(windowed)-1):
            # store the current 3-letter Sentinel2 abbreviation
            first_ind= z
            # initialize variables that will be used to hold and process data later
            med_band = {}
            transforms = {} 
            b02,b03,b04 = [],[],[]
            cnt_clouds=0 

            # iterate over all of the image data paths and files in the dictionary
            for y,d in dictionary_of_bands.items():
                # if the current image data file has the current 3-letter Sentinel2 abbreviation and we are looking at band2 data,
                # then get the image data from the current slice, get the path to the corresponding cloud data, and filter out cloud pixels from the current slice
                '''BAND 2'''
                if str(first_ind) in y and 'B02' in y and band == 'B02':
                    # extract the image data for the current slice
                    med_band[str(first_ind)+'_'+str(y)]=d.read(1,window=Window.from_slices(slice(windowed[gg],windowed[gg+1]), slice(0, 10980))) # (row start, row stop, col start, col stop)
                    # find the path to the corresponding cloud data mask file
                    cnt_clouds = [j for j in clouds_pth if str(d.name.split('/')[6]) in str(j.split('/')[6])]
                    # set cloud pixels to np.nan by filtering out pixels whose brightness value is too high (> 4000) or whose brightness value is 0 (all black)
                    med_band[str(first_ind)+'_'+str(y)] = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan,np.where(med_band[str(first_ind)+'_'+str(y)]>4000,np.nan,med_band[str(first_ind)+'_'+str(y)]))
                    
                    try:
                        # if the cloud mask data file is large enough, we will process
                        if os.path.getsize(cnt_clouds[0]) > 500:
                            # read the cloud mask data file
                            cld = gpd.read_file(cnt_clouds[0])
                            # find the coordinate reference system for the cloud mask file
                            cld.crs = ({'init':str(d.crs)})

                            # convert the cloud mask data to a raster that has the same shape and transformation as the image raster data
                            image = features.rasterize(
                                        ((g['geometry'], 1) for v,g in cld.iterrows()),
                                          out_shape=d.read(1).shape,
                                          transform=d.transform,all_touched=True)
              
                            # find the indices in the cloud mask raster data where the red channel is 0, the green channel is 1, and the blue channel is 0
                            imagee = np.where(image==0,1,0)
                        
                        # if the cloud mask data file is too small, create an empty numpy array that has the same shape as the shape of the current image data file
                        if os.path.getsize(cnt_clouds[0])<500:
                            # create an empty numpy array that has the same shape as the shape of the current image data file
                            imagee=np.empty_like(med_band[str(first_ind)+'_'+str(y)].shape)
                            # find the indices in the empty numpy array where the red channel is 0, the green channel is 1, and the blue channel is 1
                            imagee=np.where(imagee==0,1,1)
                            continue
                    except:
                        # if we can't open the cloud mask, create an empty numpy array
                        imagee=np.empty_like(d.read(1).shape)
                        # find the indices where the red channel is 0, the green channel is 1, and the blue channel is 1
                        imagee=np.where(imagee==0,1,1)
                        continue

                    # select the image data from the current slice and index
                    med_band[str(first_ind)+'_'+str(y)] = med_band[str(first_ind)+'_'+str(y)] * imagee[windowed[gg]:windowed[gg+1],0:10980]
                    # filter out cloud pixels in the current slice by finding pixels that have a brightness of 0 or whose brightness value is too high (> 4000) and set them to np.nan
                    med_band[str(first_ind)+'_'+str(y)] = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan,np.where(med_band[str(first_ind)+'_'+str(y)]>4000,np.nan,med_band[str(first_ind)+'_'+str(y)]))

                    # find the number of indices that are equal to zero in the current slice
                    num_zeros = med_band[str(first_ind)+'_'+str(y)][med_band[str(first_ind)+'_'+str(y)]==0]
          
                    # if there are indices in the current slice that are equal to zero, set those indices to np.nan
                    if len(num_zeros) != 0 :
                        get_this=0
                        # set the indices that are equal to 0 in the current slice to be np.nan
                        get_this = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan ,med_band[str(first_ind)+'_'+str(y)])
                        # replace the zero indices with np.nan in the current slice and update the dictionary value accordingly
                        med_band[str(first_ind)+'_'+str(y)] = get_this

                '''BAND 3'''
                # if the current image data file has the current 3-letter Sentinel2 abbreviation and we are looking at band3 data,
                # then get the image data from the current slice, get the path to the corresponding cloud data, and filter out cloud pixels from the current slice
                if str(first_ind) in y and 'B03' in y and band == 'B03':
                    # extract the image data for the current slice
                    med_band[str(first_ind)+'_'+str(y)]=d.read(1,window=Window.from_slices(slice(windowed[gg],windowed[gg+1]), slice(0, 10980))) # (row start, row stop, col start, col stop)
                    # find the path to the corresponding cloud data mask file
                    cnt_clouds = [j for j in clouds_pth if str(d.name.split('/')[6]) in str(j.split('/')[6])]
                    # set cloud pixels to np.nan by filtering out pixels whose brightness value is too high (> 4000) or whose brightness value is 0 (all black)
                    med_band[str(first_ind)+'_'+str(y)] = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan,np.where(med_band[str(first_ind)+'_'+str(y)]>4000,np.nan,med_band[str(first_ind)+'_'+str(y)]))
                    
                    try:
                        # if the cloud mask data file is large enough, we will process
                        if os.path.getsize(cnt_clouds[0]) > 500:
                            cld = gpd.read_file(cnt_clouds[0])
                            cld.crs = ({'init':str(d.crs)})
                            image = features.rasterize(
                                      ((g['geometry'], 1) for v,g in cld.iterrows()),
                                      out_shape=d.read(1).shape,
                                      transform=d.transform,all_touched=True)
              
                            imagee = np.where(image==0,1,0)
                        
                        if os.path.getsize(cnt_clouds[0])<500:
                            imagee=np.empty_like(med_band[str(first_ind)+'_'+str(y)].shape)
                            imagee=np.where(imagee==0,1,1)
                            continue
                    except:
                        imagee=np.empty_like(d.read(1).shape)
                        imagee=np.where(imagee==0,1,1)
                        continue
          
                    med_band[str(first_ind)+'_'+str(y)] = med_band[str(first_ind)+'_'+str(y)] * imagee[windowed[gg]:windowed[gg+1],0:10980]
                    med_band[str(first_ind)+'_'+str(y)] = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan,np.where(med_band[str(first_ind)+'_'+str(y)]>4000,np.nan,med_band[str(first_ind)+'_'+str(y)]))
        
                    num_zeros = med_band[str(first_ind)+'_'+str(y)][med_band[str(first_ind)+'_'+str(y)]==0]
          
                    if len(num_zeros) != 0 :
                        get_this=0
                        get_this = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan ,med_band[str(first_ind)+'_'+str(y)])
                        med_band[str(first_ind)+'_'+str(y)] = get_this


                '''BAND 4'''
                # if the current image data file has the current 3-letter Sentinel2 abbreviation and we are looking at band4 data,
                # then get the image data from the current slice, get the path to the corresponding cloud data, and filter out cloud pixels from the current slice
                if str(first_ind) in y and 'B04' in y and band == 'B04':
                    # extract the image data for the current slice
                    med_band[str(first_ind)+'_'+str(y)]=d.read(1,window=Window.from_slices(slice(windowed[gg],windowed[gg+1]), slice(0, 10980))) # (row start, row stop, col start, col stop)
                    # find the path to the corresponding cloud data mask file
                    cnt_clouds = [j for j in clouds_pth if str(d.name.split('/')[6]) in str(j.split('/')[6])]
                    # set cloud pixels to np.nan by filtering out pixels whose brightness value is too high (> 4000) or whose brightness value is 0 (all black)
                    med_band[str(first_ind)+'_'+str(y)] = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan,np.where(med_band[str(first_ind)+'_'+str(y)]>4000,np.nan,med_band[str(first_ind)+'_'+str(y)]))
            
                    try:
                        # if the cloud mask data file is large enough, we will process
                        if os.path.getsize(cnt_clouds[0]) > 500:
                            cld = gpd.read_file(cnt_clouds[0])
                            cld.crs = ({'init':str(d.crs)})

                            image = features.rasterize(
                                      ((g['geometry'], 1) for v,g in cld.iterrows()),
                                      out_shape=d.read(1).shape,
                                      transform=d.transform,all_touched=True)
              
                            imagee = np.where(image==0,1,0)
                
                        if os.path.getsize(cnt_clouds[0])<500:
                            imagee=np.empty_like(med_band[str(first_ind)+'_'+str(y)].shape)
                            imagee=np.where(imagee==0,1,1)
                            continue
                    except:
                        imagee=np.empty_like(d.read(1).shape)
                        imagee=np.where(imagee==0,1,1)
                        continue
        
                    med_band[str(first_ind)+'_'+str(y)] = med_band[str(first_ind)+'_'+str(y)] * imagee[windowed[gg]:windowed[gg+1],0:10980]
                    med_band[str(first_ind)+'_'+str(y)] = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan,np.where(med_band[str(first_ind)+'_'+str(y)]>4000,np.nan,med_band[str(first_ind)+'_'+str(y)]))

                    num_zeros = med_band[str(first_ind)+'_'+str(y)][med_band[str(first_ind)+'_'+str(y)]==0]
          
                    if len(num_zeros) != 0 :
                        get_this=0
                        get_this = np.where(med_band[str(first_ind)+'_'+str(y)]==0,np.nan ,med_band[str(first_ind)+'_'+str(y)])
                        med_band[str(first_ind)+'_'+str(y)] = get_this

                if str(first_ind) in y:
                    transforms[first_ind] = d.transform

                if str(first_ind) in y:
                    crs_correct = d.crs
                    reshap = list(med_band.values())[0].shape

                '''BAND 2'''
                '''Each of the bands after created, we delete the variables to save ram so that the instance does not crash '''
                if band == 'B02':

                    Y_2 = np.vstack((e.ravel() for x,e in med_band.items() if 'B02' in x ))

                    del med_band

                    Z2 = np.nanmedian(Y_2,axis = 0, overwrite_input= True)

                    del Y_2

                    Z2 = np.uint16(Z2.reshape(reshap))

                    trueColor = rasterio.open(os.getcwd()+'/'+str(first_ind)+str(nam[gg])+'_cld_msk_wThresh_above4000_'+str(band)+\
                                  'multi_newER'+'.tiff','w',driver='Gtiff',
                                width=reshap[1], height=reshap[0],
                                count=1,
                                crs=crs_correct,
                                transform=list(transforms.values())[0],
                                dtype=Z2.dtype
                                )

                    trueColor.write(Z2,1) #green
                    trueColor.close()
                    del Z2 

                '''BAND 3'''
                if band == 'B03':

                    Y_3 = np.vstack([e.ravel() for x,e in med_band.items() if 'B03' in x])

                    del med_band

                    Z3 = np.nanmedian(Y_3,axis = 0,overwrite_input=True)
                    del Y_3

                    Z3 = np.uint16(Z3.reshape(reshap))
                    trueColor = rasterio.open(os.getcwd()+'/'+str(first_ind)+str(nam[gg])+'_cld_msk_wThresh_above4000_'+str(band)+\
                                  'multi_newER'+'.tiff','w',driver='Gtiff',
                                width=reshap[1], height=reshap[0],
                                count=1,
                                crs=crs_correct,
                                transform=list(transforms.values())[0],
                                dtype=Z3.dtype
                                )

                    trueColor.write(Z3,1) #green
                    trueColor.close()

                    del Z3


                '''BAND 4'''
                if band == 'B04':

                    Y_4 = np.vstack((e.ravel() for x,e in med_band.items() if 'B04' in x ))

                    del med_band

                    Z4 = np.nanmedian(Y_4,axis = 0, overwrite_input= True)

                    del Y_4

                    Z4 = np.uint16(Z4.reshape(reshap))

                    trueColor = rasterio.open(os.getcwd()+'/'+str(first_ind)+str(nam[gg])+'_cld_msk_wThresh_above4000_'+str(band)+\
                                  'multi_newER'+'.tiff','w',driver='Gtiff',
                                width=reshap[1], height= reshap[0],
                                count=1,
                                crs=crs_correct,
                                transform=list(transforms.values())[0],
                                dtype=Z4.dtype
                                )

                    trueColor.write(Z4,1) #green
                    trueColor.close()
                    del Z4

    return

def write_out(indexes, bands, slices, name, dire):
    towindow = slices
    mediocre=[]
    aditi = 0
    mediocre.append(int(aditi))
    for i in range(towindow):
        aditi+= 10980/towindow
        mediocre.append(int(aditi))
    
    jojo = [str(i)+'of'+str(towindow-1) for i in range(len(mediocre)-1)]

    tsotso = os.listdir(str(dire))# catch all the slices of all bands in order to write them as an image

    windowed = mediocre 
    nam = jojo
    
    for j in indexes:
        first_ind = j
    
        final_truecolor= []# used to catch each of the bands in order to write in disk the overall Truecolo picture

        for k in bands:
            overbnd = []
            for g in tsotso:
                if first_ind in g and k in g and 'of' in g:
                    file = rasterio.open(str(g))
                    overbnd.append(file)
                
                #del file, g
      
            with rasterio.open(
                os.getcwd()+'/'+'Multiband_median_corrected'+str(first_ind)+str(k)+'.tiff', 'w',
                driver='GTiff', width=10980, height=10980, count=1, crs = overbnd[0].crs,transform= overbnd[0].transform,
                dtype=overbnd[0].dtypes[0]) as dst:
                    for v in range(len(windowed)-1):
                        dst.write(overbnd[v].read(1),window = Window.from_slices(slice(windowed[v],windowed[v+1]),slice(0,10980)),indexes=1)
     
            file2 = rasterio.open('Multiband_median_corrected'+str(first_ind)+str(k)+'.tiff')

            final_truecolor.append(file2)

            file2.close()

            #del k, overbnd, file2

        TruE = rasterio.open(os.getcwd()+'/'+name+'_multiband_cld_NAN_median_corrected'+str(first_ind)+'.tiff','w',driver='Gtiff',
                        width=final_truecolor[0].width, height=final_truecolor[0].height,
                        count=3,
                        crs=final_truecolor[0].crs,
                        transform=final_truecolor[0].transform,
                        dtype=final_truecolor[0].dtypes[0]
                        )

        TruE.write(final_truecolor[0].read(1),3) #green
        TruE.write(final_truecolor[1].read(1),2)
        TruE.write(final_truecolor[2].read(1),1)

        TruE.close()

        ds = gdal.Open(os.getcwd()+'/'+name+'_multiband_cld_NAN_median_corrected'+str(first_ind)+'.tiff')
        r = ds.GetRasterBand(3).ReadAsArray()
        g = ds.GetRasterBand(2).ReadAsArray()
        b = ds.GetRasterBand(1).ReadAsArray()

        ds = None
        del ds
    
        r = scale(r)
        g = scale(g)
        b = scale(b)

        r = r.astype('uint8')
        g = g.astype('uint8')
        b = b.astype('uint8')
    
        dss=rasterio.open(str(input_rstr))
        dss.transform
    
        output_scaled = os.getcwd()+'/'+'multiband_scaled_corrected'+str(input_rstr.split('.')[0][-3:])+'.tiff'

        TruE = rasterio.open(str(output_scaled),'w',driver='Gtiff',
                        width=dss.width, height=dss.height,
                        count=3,
                        crs=dss.crs,
                        transform=dss.transform,
                        dtype='uint8'
                        )

        TruE.write(r,3)
        TruE.write(g,2)
        TruE.write(b,1)

        TruE.close()

        # upload the scaled, color corrected tiff to the S3 bucket
        store = [os.getcwd()+'/'+'multiband_scaled_corrected'+str(input_rstr.split('.')[0][-3:])+'.tiff']
        
        os.chdir(os.getcwd())
        
        for obj in store: 
            s3.Bucket(bucket_name).upload_file(obj, name + '_' + obj)
    
        del r, g, b, dss, TruE
    
    return

def download(bound, name, buffer, sent, dire_home, reg_account, password, start_dt, end_dt, dire_sent, dir_store, bucket_name, s3):
    # download the requested Sentinel2 image data as .zip files 
    download_sent(bound, buffer, sent, dire_home, str(reg_account) , str(password) , str(start_dt), str(end_dt))
    # unzip the folder of Sentinel2 .zip files
    unzip(dire_sent)
    
    ls_names, ls_clouds = list_files(dire_sent)
    #dict_med, indx = dict_names(ls_names)
    dict_med , indx = dict_names(ls_names)
    get_median_slices(indx, dict_med, ls_clouds, dire_home, 'B02', 2)

    dict_med , indx = dict_names(ls_names)
    get_median_slices(indx, dict_med, ls_clouds, dire_home, 'B03', 2)

    dict_med , indx = dict_names(ls_names)
    get_median_slices(indx, dict_med, ls_clouds, dire_home, 'B04', 2)
    
    write_out(indx, ['B02','B03','B04'] , 2, name, dir_store)
    
    store = [i for i in os.listdir(dir_store) if i.endswith('tiff') and  i and 'NAN' in i]

    for obj in store: 
        s3.Bucket(bucket_name).upload_file(obj, obj)
        
    return

def main():
    # define the paths to each of the folders that we will create
    # note: you might need to add a /AWS_name where AWS_name is 
    dire_home = '/home'
    dire_sent = dire_home + '/sentinel_folder'
    dir_store = dire_home + '/bands_folder'

    os.chdir(dire_home)

    # AOI = gpd.read_file('zambia.shp')
    sent = dire_home + '/kml_sentinelmy_new_sentinel.shp'

    # account information for Copernicus Open Access Hub, register at https://scihub.copernicus.eu/userguide/SelfRegistration
    reg_account = 'meco6443'
    password = ''

    bucket = 'b2p.meco' # replace with your bucket name
    s3 = boto3.resource('s3')

    # start and ending dates for Sentinel2 data in YYYYMMDD format
    start_dt = '20190901'
    end_dt = '20191114'

    # buffer in meters around the bounding box of the region
    buffer = 500

    # bounding boxes for the regions we need inference for by Wednesday
    chipata_bounds = [32.1144, -14.1086, 32.7708, -13.2593]

    kafue_bounds = [27.8998, -16.5625, 29.2126, -14.8798]

    chibombo_bounds = [27.1101, -15.7011, 28.4230, -14.0114]

    chirundu_bounds = [28.3633, -16.5796, 29.0197, -15.7400]

    # np array containing all of the region bounds for which we need inference
    zambia_bounds = np.array([chipata_bounds, kafue_bounds, chibombo_bounds, chirundu_bounds])

    # names of the districts for which we want inference
    names = ['chipata', 'kafue', 'chibombo', 'chirundu']

    # iterate over the bounding boxes and region names, download and process the corresponding Sentinel2 data, upload the processed 
    # files to the S3 bucket, and delete all the downloaded, pre-processed data
    for bound, name in zip(zambia_bounds, names):
        os.chdir(dire_home)
    
        download(bound, name, buffer, sent, dire_home, reg_account, password, start_dt, end_dt, dire_sent, dir_store, bucket, s3)
    
        shutil.rmtree(dire_sent)

    return

if __name__ == '__main__':
    main()