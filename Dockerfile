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

# Download stopwords
RUN python -c "import nltk;nltk.download('stopwords')"

# Add Spacy and models
conda install -c conda-forge spacy && \
    python -m spacy download en && \
    python -m spacy download en_core_web_md && \
    python -m spacy download en_core_web_lg && \
    python -m spacy download en_vectors_web_lg

WORKDIR /home/jovyan/work
