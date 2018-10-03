FROM jupyter/datascience-notebook:latest

USER root
RUN apt-get update && \
    apt-get install -y \
        p7zip \
        hunspell

USER jovyan
# RUN conda update --all
RUN conda install -c conda-forge -y \
        flask \
        gunicorn \
        lxml \
        plotly \
        scikit-learn>=0.20 \
        tqdm \
        xgboost

RUN pip install \
    cssselect \
    mwapi \
    mwparserfromhell \
    mwxml \
    sklearn-pandas \
    yarl

WORKDIR /home/jovyan/work
