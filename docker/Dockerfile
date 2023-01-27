FROM python:3

WORKDIR /usr/src/app

RUN git init && git clone https://github.com/earthlab/bridges_to_prosperity_ML.git
ENTRYPOINT ["/bin/bash", "env.sh"]
