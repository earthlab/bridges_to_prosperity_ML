FROM python:3.10

WORKDIR /usr/src/app

RUN git init && git clone https://github.com/earthlab/bridges_to_prosperity_ML.git
WORKDIR /usr/src/app/bridges_to_prosperity_ML
RUN git checkout docker
RUN pip install ipython

# RUN apt-get update && apt-get install -y \
#     gdal-bin \
#     libgdal-dev

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essentials  && \
    apt-get install -y wget &&
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda activate
RUN conda install -c conda-forge gdal
RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash", "env.sh"]
