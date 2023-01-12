# bridges_to_prosperity_ML


## download_and_composite.py

`python bin/download_and_composite.py --outdir <abs_path_to_your_outdir> --bbox <min_lon,min_lat,max_lon,max_lat> --start_date <date> --end_date <date> --region <region> --buffer 500 --slices 10`
## b2p_train_optical

This script trains all the RESNET models on 70% of the data from Uganda and Rwanda, exports the saved state of each RESNET model to a pkl file, and then evaluates all the RESNET models on the remaining 30% of the data from Uganda and Rwanda. 
It then prints out a report summarizing how each of the RESNET models performed on the held-out validation data (the remaining 30% of the data). This requires that someone decides which model they want to use for inference by-hand, but it allows for greater flexibility and provides all the possible models so that the models can be swapped out if the need arises. This function takes two arguments: a dictionary of the form {'Region Name': region_csv} where Region Name is a string like 'Uganda' or 'Rwanda' and region_csv is a cleaned version of the csv files provided by Kyle and a dictionary of the form {'Region Name': path_to_tiff_tiles} where Region Name is a string like 'Uganda' or 'Rwanda' and path_to_tiff_files is a path from the current working directory to the directory containing the tiff tiles for the current region.  This code assumes that we have already run the api_download.py script on the bounding boxes for the regions in Uganda and Rwanda to obtain all the S2 data for these regions, that we have then created a composite for the S2 data in these regions, and that we then called the make_tile_tiffs.py script to create 300m x 300m tiff tiles across all the S2 tiles in these regions. The path to the directory containing the 300m x 300m tiff tiles for each region should be placed in the path_to_tiff_tiles place in the 
aforementioned dictionary. 
