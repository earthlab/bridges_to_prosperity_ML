FROM jupyter/minimal-notebook

WORKDIR /usr/src/app

RUN git init && git clone https://github.com/earthlab/bridges_to_prosperity_ML.git
WORKDIR /usr/src/app/bridges_to_prosperity_ML
RUN git checkout docker
RUN pip install ipython

# Install base utilities
RUN conda install -c conda-forge mamba && \
    mamba install -c conda-forge -y gdal=3.4.1 && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash", "env.sh"]
