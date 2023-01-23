
#! /bin/bash
if [ -z "$STY" ]; then exec screen -dm -S dockerRun /bin/bash "$0"; fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
sudo yum update 
sudo amazon-linux-extras install docker
sudo service docker start
sudo chmod 666 /var/run/docker.sock
docker system prune -f
docker run --name hella --hostname localhost -v $SCRIPT_DIR:/b2p -it -p 8888:8888 earthlab/earth-analytics-python-env