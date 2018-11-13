FROM jupyter/datascience-notebook:latest

USER root
RUN apt-get update && \
    apt-get install -y \
        p7zip \
        hunspell

USER jovyan

COPY environment.yml .

# RUN conda update --all
RUN conda env update -n base -f "environment.yml" && \
    conda clean -tipsy && \
    conda list -n base

RUN python -m spacy download en
RUN python -m spacy download en_core_web_md

WORKDIR /home/jovyan/work
