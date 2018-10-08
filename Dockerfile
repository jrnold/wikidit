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
        joblib \
        lxml \
        plotly \
        scikit-learn>=0.20 \
        spacy=2.0 \
        smart_open \
        tqdm \
        xgboost

RUN pip install \
    cssselect \
    importlib_resources \
    mwapi \
    mwparserfromhell \
    mwxml \
    sklearn-pandas \
    yarl \
    git+https://github.com/jrnold/sklearn-ordinal.git

RUN python -m spacy download en
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_lg

WORKDIR /home/jovyan/work
