
# Getting started 
open a screen, install docker, set permissions, start docker instance with docker run (dont forget to mount repo)

[Instructions](https://docs.google.com/document/d/1SiW0amSxxCDKeDycJTu-SBKYeerSx2lM9a7lR_F3PYs/edit#)
get into aws instance
``` bash 
ssh -i niru8088.cer ec2-user@ec2-18-236-56-225.us-west-2.compute.amazonaws.com 
```

open screen and start docker container
<!-- ``` bash 
screen -S dockerRun
sudo yum update 
sudo amazon-linux-extras install docker
sudo service docker start
sudo chmod 666 /var/run/docker.sock
docker system prune
'
docker run --name hella --hostname localhost -v $PWD/bridges_to_prosperity_ML:/b2p -it -p 8888:8888 earthlab/earth-analytics-python-env
``` -->
exit with cntl+a,cntl+d

exec into the docker container
``` bash 
screen -S dockerExec
docker exec -it hella /bin/bash
cd /b2p
source env.sh
```

# scripts 


## download_sentinel2.py

This script downloads the files from sentinel2 for a specified bounding box and date range. I have been told to use images from 2019 because more recent data is not always good/avaiable.

```bash 
python bin/download_sentinel2.py --bbox <min_lon,min_lat,max_lon,max_lat> --start_date <date> --end_date <date> --region <region> --buffer 500 --slices 1
```

### Dry Season 
FIRST TRY: the dates of --start_date 2019-09-01 --end_date 2019-11-05 for all training data (Rwanda and Uganda)
### Preliminary Data analysis
[https://cloudless.eox.at/] out of the box solution
Motivations It is important to minimize the ammount of clouds in the data set.
From visual inspection of [https://www.worlddata.info/africa/ivory-coast/climate.php] we get an inspection of . 
for deep dive [https://www.visualcrossing.com/weather/weather-data-services#] (nic has an account)
- Rwanda : all of Jun and July into the early part of Aug
- Uganda : Dry season appears to be mid Jun, all of July and the early part of Aug (Aug looks better than Rwanda)
- Ivory Coast : All of Dec - Jan
- Ethiopia: all of Dec - Feb
- Zambia: all of May-Sep

## sentinel2_to_composite 

c

## b2p_train_optical

This script trains all the RESNET models on 70% of the data from Uganda and Rwanda, exports the saved state of each RESNET model to a pkl file, and then evaluates all the RESNET models on the remaining 30% of the data from Uganda and Rwanda. 
It then prints out a report summarizing how each of the RESNET models performed on the held-out validation data (the remaining 30% of the data). This requires that someone decides which model they want to use for inference by-hand, but it allows for greater flexibility and provides all the possible models so that the models can be swapped out if the need arises. This function takes two arguments: a dictionary of the form {'Region Name': region_csv} where Region Name is a string like 'Uganda' or 'Rwanda' and region_csv is a cleaned version of the csv files provided by Kyle and a dictionary of the form {'Region Name': path_to_tiff_tiles} where Region Name is a string like 'Uganda' or 'Rwanda' and path_to_tiff_files is a path from the current working directory to the directory containing the tiff tiles for the current region.  This code assumes that we have already run the api_download.py script on the bounding boxes for the regions in Uganda and Rwanda to obtain all the S2 data for these regions, that we have then created a composite for the S2 data in these regions, and that we then called the make_tile_tiffs.py script to create 300m x 300m tiff tiles across all the S2 tiles in these regions. The path to the directory containing the 300m x 300m tiff tiles for each region should be placed in the path_to_tiff_tiles place in the 
aforementioned dictionary. 

## Jupyter notebook
In a docker 
`docker exec -it hella bash`
`jupyter notebook list `  -> gives a URL 
Take URL open in browser back on personal machine
EX: http://0.0.0.0:8888/?token=38e9d27171d4c02e1841b90ab5f003848ffa0dedb8b7172a