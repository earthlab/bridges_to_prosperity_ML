# 
# Getting started 

## With Docker on AWS EC2 Instance (recommended)

### EC2 Security Group Configuration
Along with provisioning the code runtime environment, the container also 
starts a Jupyterlab server on port 8888. If you would like to use the Jupyter
server to interact with the code, make sure that the EC2 instance's security group has
an inbound rule set allowing traffic on port 8888.


### EC2 IAM Profile Configuration
In order to transfer data to and from the EC2 instance and S3 storage, the 
EC2 instance must be initialized with an IAM profile which has the S3FullAccessPermissions role.
This role can then be attached to the instance in the IAM Profile Settings section of the Advanced 
Settings when starting an instance.


### Installing Docker and Screen on EC2 instance with AWS Linux 2023 AMI
``` bash 
sudo yum update 
sudo yum install docker
sudo yum install screen
```

### Running Docker
``` bash 
sudo service docker start
sudo chmod 666 /var/run/docker.sock
```

### Open screen and start Docker Container
Make sure to replace <server_password> with a password of your choice
``` bash 
screen -S docker
sudo docker run -p 8888:8888 -e JUPYTER_TOKEN=<server_password> earthlabcu/b2p
```
To put screen in background
``` bash 
exit with cntl+a+d
```

### Using the container application

#### Connect to Jupyter server
Use your browser to go to <ec2_instance_public_ipv4_address>:8888  
You should be prompted to input the server token. Input the JUPYTER_TOKEN that was 
used to start the container  
Once on the server, the terminal application can be used to run the programs in bin. See the 
Running the Code section below for further instructions.  
Jupyter notebooks can also be used to import functions into and run code.


#### Exec into container to use command line 
If you prefer to use only the command line to interact with the application, use the
following command from within the EC2 instance.
``` bash 
sudo docker exec -it b2p /bin/bash
```
This will start a session in the docker container's bash shell

## Without Docker 

If Conda is not already installed, install with:
``` bash 
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
```

This will download a file called Anaconda3-2023.03-1-Linux-x86_64.sh in the directory that you ran the wget command in.
Update the file permissions with:
``` bash 
wget -v +x Anaconda*.sh
```

Run the installer file by running the command:
``` bash 
./Anaconda3-2023.03-1-Linux-x86_64.sh
```
You will have to press enter and type yes to get through the licensing / prompts

Add conda to your path and initialize base environment by running:
``` bash 
source ~/.bashrc
```
You will now be in the base conda environment

Install git and clone the GitHub repo
Install git with:
``` bash 
sudo yum install git
```
Clone the repo with:
``` bash 
git clone https://github.com/earthlab/bridges_to_prosperity_ML
``` 

cd into repo
``` bash 
cd bridges_to_prosperity_ML
```

Create the new environment with:
``` bash 
conda env create -n b2p --file environment.yml
```

Activate the conda env with:
``` bash 
conda activate b2p 
```

If not using an EC2 instance with a properly configured IAM profile (recommended)
you must install the AWS CLI and authenticate with the AWS account whose S3 bucket 
will be configured to the project

# Overview

## Running the code

This application is designed to be interacted with through its command line executables. These programs are in the bin directory and can be run with
``` bash 
python <path_to_.py_script>
```

To see a description of each program's input parameters, run the program with the --help flag like so
``` bash 
python <path_to_.py_script> --help
```
A description of each of these programs is included further down in this README

## Configuring S3

Several of the programs will attempt to interact with the project's s3 bucket. The user will be initially prompted for the name of the s3 bucket to use for the project. 
To change the name of the project's s3 bucket, update its value in the config.yaml file

## region_info.yaml file

Each region of interest and its districts are kept track of in the data/region_info.yaml file. Each entry has a range of dates and a bounding box with format [min_lon, min_lat, max_lon, max_lat]. There is also a specification for ground truth data. 
If there is ground truth data for a district, then the ground_truth key should be True, and False if not. For regions for which there is ground truth, a target column will be added to the tile match csv and inference results file. 

Each new region and its districts for which the user would like to create composites, tiles, train new models on etc. must be added to this file first.

## Composite data APIs

In general we get data from three sources:  
    - Sentinel2: The most time consuming data source to use, but shows considerable promise. Both the visual light bands and near infrared both come from this source.  
    - SRTM: This is a NASA database that provides elevation data. Slope is computed via finite difference methods from here  
    - OSM: there is a convenient python api that is leveraged to obtain this data source  

There code for each of these data sources are scripts located in the src/api directory named sentinel2.py, lp_daac.py, and osm.py respectively. The specifics of these apis are abstracted away through the programs in bin, although the APIs may be used as stand alone modules for downloading data if the user wishes.
Documentation has been added to each of these modules and is sufficient for learning how to use them.

## Ground truth data

Ground truth files are stored as csv in the data/ground_truth directory. When ground truth is looked for, the most recent of the csv files is returned. Thus, any new ground truth data should be appended to the latest existing csv file. If the user wishes, they can create a new 
ground truth csv file with the same naming convention of ground_truth_{mm}.{dd}.{yyyy}.csv for the most recent date.

# File types

There are several different types of files serving various purposes throughout the application. Each file type has a defined naming structure and archival location within the data directory. Below is a short description of each file type.
The code defining the structure of each file type can be found in the file_types.py script.

## Composites
Composites are the main data source used for training and running inference over a region. They are stored as tif files with 8 bands. The bands, in order, are red, green, blue, near IR, open street map water, open street map admin boundaries, elevation, and slope. 
Each composite covers a 1 deg lat / lon area as defined in the Sentinel-2 UTM Tiling Grid system. In this system each grid location is specified with a 2 digit number and 3 letters, like 35MGR. When creating composites, data for each UTM tile that overlaps the bounding box specified in the region_info.yaml is found. Each file's name will include the name of the UTM grid that it covers. The spatial resolution of the composites is 10m. More information on the UTM Tiling Grid system can be found at https://eatlas.org.au/data/uuid/f7468d15-12be-4e3f-a246-b2882a324f59 .
Each composite is stored at data/composites/{region}/{district}

## Tiles
Tiles are tif files that are a part of a larger composite file. The size of each of these tiles can be specified from the command line when running the tiles_from_composites.py program. The default size of the tiles is 300x300m, in which case a single composite 
produces 133956 tiles. The tiles are used as input to the training and inference programs. The tiles are stored at data/tiles/{region}/{district}/{utm_tile}

## Tile Match CSV
Every time a set of tiles is made from a composite, an accompanying tile match csv file is created. This tile match file includes a row for each created tile and a column for the path to the tile's pytorch tensor file and its bounding box. If there is truth data for the region, an is_bridge column will be
included in the tile match file. Single region tile match files are stored in data/tiles/{region}/{district}/{utm_tile} for the tiles in a specific utm tile. The tile match file for an entire district is located at data/tiles/{region}/{district} and is the concatnation of all the utm tile match files for that district. The tile match file for an entire region is stored at data/tiles/{region} and is the concatenation of all the district tile match files for that region. Multi region tile match files are found in the data/multi_region_tile_match directory.

## Trained Models
Each epoch of training will produce a tar file which can be loaded in by PyTorch to perform inference. Each tar file will be named such that the regions, architecture, class ratio, layers, and tile size used to train the model will be included in it. For example,
the tar file Rwanda_Uganda_resnet18_r2.0_ts300_nir_osm-water_elevation_epoch20_best.tar represents a model trained on tiles from Rwanda and Uganda, using resnet18 architecture, with a no bridge / bridge class ratio of 2, IR, elevation, and osm-water layers, on tiles of size 300x300m. Further, this
model was output after the 20th epoch of training, and was thus far the best model to be output due to its total accuracy score, indicated by the _best at the end of the file name. Thus, for a single round of training the tar file with highest epoch and _best in the filename has the highest overall accuracy score.
These model files are stored locally at data/trained_models/{region(s)}/{architecture}/{class_ratio}

## Inference Results
Running inference will produce both a csv and shapefile with the results of the run. The shapefile is actually a set of files, one of which is of the .shp extension. This set of files exist together in a directory. When the upload_s3.py program is used to upload the inference results, this directory
is compressed and then the resulting tar file is uploaded. These files are stored locally at data/inference_results/{region(s)}/{architecture}/{class_ratio}

# Programs

Creating new composites, tiles, models, and running inference can be accomplished with the programs executable programs in the bin directory. Each of these programs is run with the command  
``` bash 
python <path_to_.py_script>
```

And a description of each program's input parameters can be found by running the program with the --help flag like so
``` bash 
python <path_to_.py_script> --help
```

Each program has several optional flags for tuning the run to a specific location, for example. A more thorough description of each program and its input parameters can be found below

## download_s3.py

Used to download composites, trained models, or inference results from the s3 bucket configured to the project. 
One of these file types must be specified when calling this program.

Example usage for downloading existing files in s3 of a certain type:

### composites:
``` bash 
python bin/download_s3.py composites
``` 
Will download all existing composites. To further refine your download, specify any combination of region, district, or utm tile:

``` bash 
python bin/download_s3.py composites --region Uganda
```
Will download all composites for Uganda only

``` bash 
python bin/download_s3.py composites --region Uganda --district Kasese
```
Will download all composites for Uganda in the Kasese district only

``` bash 
python bin/download_s3.py composites --mgrs 35NRA
```
Will download all composites that belong to the 35NRA UTM tile, regardless of region or district

### models:
``` bash 
python bin/download_s3.py models
```

Will download all available trained PyTorch models

### inference_results:
``` bash 
python bin/download_s3.py inference_results
```

Will download all available sets of inference result csv and shapefiles  

To refine the download criteria for both model and inference results files, specify any combination of 
regions, architecture, layers, epoch, ratio, tile size, or 'best' model

``` bash 
python bin/download_s3.py models --regions Rwanda Uganda
```

Will download all models trained on districts from Rwanda and Uganda only

``` bash 
python bin/download_s3.py inference_results --regions Rwanda Uganda --architecture resnet50 --layers elevation nir 
osm-water --epoch 27 --ratio 2.0 --tile_size 250 --best
```

Will download inference results for runs over Rwanda and Uganda that used resnet50 models trained on elevation, near IR, 
and OSM water layers. Further, only epoch 27, which must also be marked as best, for a bridge / no bridge ratio of 2 and
tile size of 250 m will be downloaded

The name of the s3 bucket as well as the number of cores can be changed from their project configuration values by 
specifying the following flags 

``` bash 
python bin/download_s3.py --s3_bucket_name backup_bucket --cores 10 composites 
```

Will download all composites from the s3 bucket names backup bucket and will use 10 cores to download the files in
parallel

## calculate_inference_metrics.py
Used for calculating receiver operator characteristic curve for a set of inference results over an area which was used
to train a model. By combining the inference results with the training validation set, various performance metrics can
be calculated, such as the optimal confidence threshold based on the G-Mean. 

There are 3 arguments-- inference_results, validation_set, and out_path, which must be specified:

``` bash 
python bin/calculate_inference_metrics.py --inference_results data/inference_results/Rwanda_Uganda/resnet18/r2.0/Rwanda_Uganda_resnet18_r2.0_ts300_elevation_nir_osm-water_epoch25_best.csv --validation_set data/tiles/train_validate_split/validate_Rwanda_Uganda_2.0_ts300.csv --out_path data/Rwanga_Uganda_300_roc.png
```

## create_composites.py
Creates the multivariate composites for input regions, districts, and mgrs tiles. Multivariate composites are comprised
of Sentinel2 red, green, blue, infrared, osm water, osm admin boundaries, elevation, and slope data respectively for a
single region. Any existing optical only composites will be searched for in s3 storage. Any remaining bands are created
as files and then merged together to create the final multivariate composite.

Composite creation can be done for a set of regions, districts, or UTM tiles. A slices parameter can be set in order to 
split up the file creation in memory. Creating cloud corrected optical layers can take a lot of memory, and 
specifying a greater amount of slices will lower the amount used at one time.

``` bash 
python bin/create_composites.py 
```

This will create composites for all regions, districts, and mgrs tiles in the region_info.yaml file with the default 
amount of cores and slices

``` bash 
python bin/create_composites.py --regions Uganda --districts Kasese Kibaale
```

This will create composites for the Kasese and Kibaale districts in Uganda 

``` bash 
python bin/create_composites.py --mgrs 35MRV --slices 20 --cores 5 
```

Creates a composites for UTM tile 35MRV found in any of the input regions. 20 slices will be used to create the cloud 
cleaned optical composites and 5 cores will be used to create the composites in parallel

## run_inference.py
A set of regions can be specified to run inference over, and a set of results files (csv and shapefile) will be output
for each one. If there is ground truth data for a region, a target column indicating whether a bride exists at the 
location will be output to the files. The batch size and number of workers can be set as input parameters.

Note that tiles of the same size as the model must already exist for all the specified inference regions


``` bash 
python bin/run_inference.py --model_file_path data/trained_models/Rwanda_Uganda/resnet18/r2.0/Rwanda_Uganda_resnet18_r2.0_ts300_elevation_nir_osm-water_epoch25_best.tar --inference_regions "Cote d'Ivoire" --batch_size 80 --num_workers 10
```

Will use the resnet18 model trained over Rwanda and Uganda with a tile size of 300m to run inference over each district 
in Cote d'Ivoire with a batch size of 80 and 10 workers

## tiles_from_composites.py 

Creates tiff tile files from the specified region's composites. By default, this will crawl the data/composites
directory and make tiles for all composites and write them to the appropriate location in data/tiles. A tile match csv 
file will also be written for each UTM tile. The UTM tile match files will be concatenated for each region, and
ultimately a tile match file for the region will be output. 

``` bash 
python bin/tiles_from_composites.py
```

Will create tiles for each existing composites in data/tiles with the default size of 300m,
utilizing the amount of available cores - 1.

``` bash 
python bin/tiles_from_composites.py --region Zambia --cores 3 --tile_size 200
```

Will create tiles with a tile size of 200m for each existing composite in Zambia. 3 cores will be utilized for creating
the tiles in parallel 

## train_models.py 

Performs model training in order to infer bridge locations and outputs the models to data/trained_models. The input
regions, architecture, no bridge / bridge ratio, and layers used to train the model can be specified. 

``` bash 
python bin/train_models.py --regions Uganda Rwanda --layers elevation nir osm-water
```

Will train several models using the tiles for Uganda and Rwanda. A model will be trained for each of the default bridge
/ no bridge ratios (0.5, 1.0, 2.0, 5.0) and architectures ('resnet18', 'resnet34', 'resnet50'). The default tile size of
300m will be used with a train / validate split of 70 / 30 respectively. The elevation, near infrared, and OSM water
features will be used as the input tensors for training.

``` bash 
python bin/train_models.py --regions Uganda Rwanda --layers elevation nir osm-water --class_ratios 2 --tile_size 250 --training_ratio 60 --architectures resnet18
```

Will train several models using the tiles for Uganda and Rwanda. A model will be trained for 
a bridge / no bridge ratio of 2.0 and resnet18 architecture. A tile size of 250m will be used with a train / validate
split of 60 / 40 respectively. The elevation, near infrared, and OSM water features will be used as the input tensors
for training.

## train_validate_split.py

Creates a train / validation csv set from a single tile match file for a given ratio of training to validation data. 
The tile match file for each region and tile size must already exist.

``` bash 
python bin/train_validate_split.py --regions "Cote d'Ivoire" Rwanda Uganda
```

Will create a train / validate csv set for the three input regions. The training ratio will be the default (70%) and the
tile size searched for will be the default (300m).

``` bash 
python bin/train_validate_split.py --regions "Cote d'Ivoire" Rwanda Uganda --training_ratio 50 --tile_size 100
```

Will create a train / validate csv set for the three input regions. The training ratio will be 50% and the
tile size searched for will be 100m.


## upload_s3.py

Uploads different file types to s3 storage including composite, trained model, and inference result files. Several 
parameters can be specified in order to make the uploaded file set more specific. The s3 bucket will default to the 
bucket configured to the project. If the s3_bucket_name argument is specified from the command line, this value will
override the default.

### composites 

``` bash 
python bin/upload_s3.py composites
``` 
Will upload all existing composites. To further refine your upload, specify any combination of region, district,
or utm tile:

``` bash 
python bin/upload_s3.py composites --region Uganda
```
Will upload all composites for Uganda only

``` bash 
python bin/upload_s3.py composites --region Uganda --district Kasese
```
Will upload all composites for Uganda in the Kasese district only

``` bash 
python bin/upload_s3.py composites --mgrs 35NRA
```
Will upload all composites that belong to the 35NRA UTM tile, regardless of region or district

### models:
``` bash 
python bin/upload_s3.py models
```

Will upload all available trained PyTorch models

### inference_results:
``` bash 
python bin/upload_s3.py inference_results
```

Will upload all available sets of inference result csv and shapefiles  

To refine the upload criteria for both model and inference results files, specify any combination of 
regions, architecture, layers, epoch, ratio, tile size, or 'best' model

