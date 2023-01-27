FROM python:3

WORKDIR /usr/src/app

RUN git init && git clone https://github.com/earthlab/bridges_to_prosperity_ML.git
WORKDIR /usr/src/app/bridges_to_prosperity_ML
RUN git checkout docker
RUN pip install ipython

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev

RUN pip install -r requirements.txt
ENTRYPOINT ["/bin/bash", "env.sh"]
