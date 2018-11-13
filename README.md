# wikidit source code

Source code for [wikidit](http://wikidit.jrnold.me).

## Install and Run

This requires the [Anaconda](https://www.anaconda.com/) distribution with Python 3.6, since a conda environment is used to manage dependencies.

Create and activate a conda environment with the dependencies for this project.
```console
$ conda env create --force -f environment.yml
$ conda activate wikidit
```

The app can be locally for development with the Flask development web-server using:
```
$ python app.py
```
It can be run in production with a WSGI app like gunicorn,
```
$ gunicorn --bind 0.0.0.0:8000 app
```

## Training the Model

Download texts for revisions in the training sample from the Wikipedia API.
```console
$ python -m download_enwiki_wp10_revisions.py \
    rawdata/enwiki.labeling_revisions.nettrom_30k.json \
    enwiki.labeling_revisions.w_text.nettrom_30k.ndjson.gz
```

Add features to the training data.
```console
$ python -m wikidit.scripts.add_features \
    enwiki.labeling_revisions.nettrom_30k.json \
    enwiki-labeling_revisions-w_features
```

The predictive model used in the app is defined in the notebook `notebooks/quality_predictions.ipynb`. This will update the pickled model at
`wikidit/xgboost-sequential.pkl`.
```console
$ jupyter nbconvert --execute --to notebook --inplace notebooks/quality_predictions.ipynb
```

## Description

The file [enwiki.labeling_revisions.nettrom_30k.json](https://github.com/wikimedia/articlequality/blob/master/datasets/enwiki.labeling_revisions.nettrom_30k.json)
 is a sample of 30,000+ revisions, equally balanced between
the Stub, Start, C, B, and A categories. This is used for training Mediawiki's prediction model in the
[articlequality](https://github.com/wikimedia/articlequality) package.
