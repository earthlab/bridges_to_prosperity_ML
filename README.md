# 
# Getting started 
## without docker 

clone code 
`git clone https://github.com/earthlab/bridges_to_prosperity_ML`

install anaconda and move to 3.8
<https://phoenixnap.com/kb/how-to-install-anaconda-centos-7>
`cd bridges_to_prosperity_ML`
`conda install python=3.8`
`conda create --name b2p --file=enviornment.yml python=3.8 ` 
`source env.sh`
It is worth notig that this method is not advised. Especially when using a machine with gpu's there was considerable effort to make sure that the machine had the proper drivers and so on. But this may be necessary as the code matures.

## with docker

* Erik plz update so that this is specific, I believe that the entry point is should take care of sourcing the env.sh and so on
* Erik plz set up the docker image so that they also start a jupyter server so that they can use the jupyter notebook on their local machine 

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

# Suggested Workflow 

Singing a Jupyter notebook is sometimes preferential to running scripts in the terminal. Computations such as creating input data, tiling, or training and to a lesser extent inference. It is suggested that you create a screen and run the provided scripts. This will allow for code to run for a long time without fear of the ssh tunnel being closed and stopping your progress. For smaller tasks such as plotting it may be nice to work in a Jupyter notebook. Examples include downloading/uploading files from S3 or creating plots/metrics. An example Jupyter notebook `sandbox.ipynb` is left with some example code.

## Using screens
Upon entering the virtual machine. Start a screen that will run your docker container. Then open a separate screen and exec into the docker instance to run scripts.

## Jupyter notebook
The docker container should have started a Jupyter server for you so that you can use the Jupyter notebook on your local device. This means you can run Python code on the remote machine while editing and seeing results through local web browsers. There are alternative solutions through IDE's like VSCode that allow you to edit code and IPython notebook while on the remote. Depending on the user one may be more preferential than another. 

In a docker 
`docker exec -it hella bash`
May need to make b2p visible
`ln -s /b2p /work/b2p`
`jupyter notebook list `  -> gives a URL 
Take URL open in browser back on personal machine
EX: http://0.0.0.0:8888/?token=38e9d27171d4c02e1841b90ab5f003848ffa0dedb8b7172a

# Description of Code Base

There are three main functionalities to this repo: 
- Generating Input Data
- Training New Models
- Generating Metrics
- Running Inference with Existing models

## Obtaining Input Data

There are several ways to generate input data. Depending on which model you are using only some may be relevant. It is worth checking whether composites exist on s3 before recreating input data because this can be quite time-consuming. One can look at what exists by going to s3 in a web browser and then using the provided s3 API to download the specific folders (See the IPython notebook for examples).

In general, we get data from three separate sources:
    - Sentinel2: The most time-consuming data source to use, but shows considerable promise. The visible (red, green, and blue) and the near-infrared bands come from this source.
    - SRTM: This is a NASA database that provides elevation data. The slope layer is computed via finite difference methods from here
    - OSM: there is a convenient Python API that is leveraged to obtain this data source

The code that does the heavy lifting for these three sources is located in `src/api`, but convenient wrapper scripts are available in `/bin`. In particular, the scripts that will presumably be of the most interest are `create_data_cubes.py` and `multi_variate_to_tiles.py`. Although these scripts will take a long time, they are parallelized to work as fast as possible. Another benefit to using these scripts is that they have been spot checked to create data cubes and then tiles that are of good quality

`create_data_cubes.py` does use functions from other scripts which can be called standalone if desired to break the process into multiple steps. This involves downloading data from sentinel2 in the specified date range (corresponding to an estimated dry season), creating an optical/NIR composite by removing clouds, and taking a median. Then the script should bring in the other data sources concatenating across data channels. Start to finish for a country the size of Rwanda this process can take up to two days assuming no other parts have been completed.

When new areas are added or new training data becomes available. It is suggested to leverage the `regions_info.yml`. Here you can specify the bounding box for the new region, the date range for the dry season, and whether truth data is available. From this .yml then you can specify just the region's name and the scripts to do the rest automatically. 

`multi_variate_to_tiles.py` is going to split the composites into smaller files (tiles) that the machine learning algorithms can ingest quickly. Tiles are saved to both .tif and .pt files. By default, the .tif files are deleted to save disc space, but this can be toggled by setting the parameter `remove_tiff=False` in the function `composite_to_tiles`. pt files are saved so that Pytorch has as low latency as possible when reading files. The tiling process is necessary to decrease that latency. Also, a corresponding .csv file will be saved giving the metadata for each pt file. This is important because, without the .csv file, it is arduous to back out what the bounding box is for any particular tile. If truth data is available, then tiles can be identified as being bridge suitable or not in the .csv file. This allows for training and validation if desired.

## Training 

Training requires one setup step and then a run. Given a set of tiles where truth data is available in the .csv, one can run `prepare_inputs.py` to split the data set into a training set and validation set. From here one then can run `train_models.py`. This script allows for architectures, ratios, and data channels to be specified. Ratios specify # non-bridge-suitable/ # bridge-suitable tiles in any given training/validation set. Multiple ratios can be given to produce multiple models. The data channels specify which of the multivariate data channels are desired to be used in the models to be trained. Further studies are necessary to see which combination of parameters is optimal for B2P, but the initial results are in the final slide deck and report. 

## Metrics 

Metrics code lives primarily in two places `src/ml/metrics.py` and `src/utilities/plotting.py`. Both provide interfaces to create relevant plots and figures. `src/ml/metrics.py` defines a Metrics class that allows one to look at the results from an inference run where truth data is available. You can then get a confusion matrix and plot a roc curve. `src/utilities/plotting.py` allows one to look at the results from a training run either from the point of view of a line plot or a box plot. Example code can be found in the provided IPython notebook. 

## Inference

Given a set of times (and a corresponding .csv) and a pre-trained model either from a local training run or from a model downloaded from s3 one can run inference over an area of interest. This can be done for regions where truth data is available and for regions where it is not. If truth data is available, running inference can allow us to see accuracy metrics over the entire data set rather than the subsampled ratio seen in the training phase. 

When truth data is not available running inference could be used to predict where bridge suitable sites may exist. It is worth looking back at the metrics computed early during training and validation to temper expectations. Also, if the region where inference is being done varies greatly in topography, ecology, or visual the results may vary greatly from the training/validation. Please keep this in mind.

The script to run is `run_inference.py`, specify the model, the .csv that corresponds to the tiles of interest, and output file, and other parameters such as if the truth is available. 
