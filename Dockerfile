# Use a base image with Conda pre-installed
FROM continuumio/miniconda3

# Copy the directory containing environment.yml into the Docker image
COPY ./ /app

# Change the working directory
WORKDIR /app

# Install Conda and create a new environment from environment.yml
RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yml

# Activate the created environment
RUN echo "conda activate b2p" >> ~/.bashrc
ENV PATH /opt/conda/envs/b2p/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH
# Install JupyterLab
RUN conda install -n b2p jupyterlab

# Expose the port for JupyterLab
EXPOSE 8888

# Start JupyterLab
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
