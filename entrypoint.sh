#!/bin/bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=$JUPYTER_TOKEN --allow-root
