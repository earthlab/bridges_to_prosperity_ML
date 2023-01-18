
# Getting started 
open a screen, install docker, set permissions, start docker instance with docker run (dont forget to mount repo)

[Instructions](https://docs.google.com/document/d/1SiW0amSxxCDKeDycJTu-SBKYeerSx2lM9a7lR_F3PYs/edit#)
get into aws instance
``` bash 
ssh -i niru8088.cer ec2-user@ec2-18-236-56-225.us-west-2.compute.amazonaws.com 
```

open screen and start docker container
``` bash 
screen -S njr
sudo yum update 
sudo amazon-linux-extras install docker
sudo service docker start
sudo chmod 666 /var/run/docker.sock
docker system prune
docker run --name hella --hostname localhost -v $PWD/bridges_to_prosperity_ML:/b2p -it -p 8888:8888 earthl
ab/earth-analytics-python-env
```
exit with cntl+a,cntl+d

exec into the docker container
``` bash 
docker exec -it hella /bin/bash
cd /b2p
source env.sh
conda install --file requirements.txt
ipython profile create
echo "c.InteractiveShellApp.extensions = ['autoreload']" >> ~/.ipython/profile_default/ipython_config.py 
echo "c.InteractiveShellApp.exec_lines = ['%autoreload 2']"  >> ~/.ipython/profile_default/ipython_config.py 
```

# scripts 


## download_sentinel2.py

This script downloads the files from sentinel2 for a specified bounding box and date range. I have been told to use images from 2019 because more recent data is not always good/avaiable.

### Dry Season 
It is important to minimize the ammount of clouds in the data set, so check this (website)[https://www.worlddata.info/africa/ivory-coast/climate.php] for initial estimate of good dates per region. 
`python bin/download_and_composite.py --bbox <min_lon,min_lat,max_lon,max_lat> --start_date <date> --end_date <date> --region <region> --buffer 500 --slices 10`
## b2p_train_optical

This script trains all the RESNET models on 70% of the data from Uganda and Rwanda, exports the saved state of each RESNET model to a pkl file, and then evaluates all the RESNET models on the remaining 30% of the data from Uganda and Rwanda. 
It then prints out a report summarizing how each of the RESNET models performed on the held-out validation data (the remaining 30% of the data). This requires that someone decides which model they want to use for inference by-hand, but it allows for greater flexibility and provides all the possible models so that the models can be swapped out if the need arises. This function takes two arguments: a dictionary of the form {'Region Name': region_csv} where Region Name is a string like 'Uganda' or 'Rwanda' and region_csv is a cleaned version of the csv files provided by Kyle and a dictionary of the form {'Region Name': path_to_tiff_tiles} where Region Name is a string like 'Uganda' or 'Rwanda' and path_to_tiff_files is a path from the current working directory to the directory containing the tiff tiles for the current region.  This code assumes that we have already run the api_download.py script on the bounding boxes for the regions in Uganda and Rwanda to obtain all the S2 data for these regions, that we have then created a composite for the S2 data in these regions, and that we then called the make_tile_tiffs.py script to create 300m x 300m tiff tiles across all the S2 tiles in these regions. The path to the directory containing the 300m x 300m tiff tiles for each region should be placed in the path_to_tiff_tiles place in the 
aforementioned dictionary. 
