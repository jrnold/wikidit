FROM jupyter/datascience-notebook:latest

USER root
RUN apt-get update && \
    apt-get install -y \
        p7zip \
        hunspell

USER jovyan
RUN conda install -y \
        flake8 \
        flask \
        docopt \
        numpy \
        pycodestyle \
        tqdm \
        tabulate && \
    pip install \
        articlequality \
        revscoring \
        mwapi
