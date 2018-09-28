"""Add features to the WP10 revision data"""
import gzip
import json
import pandas as pd
from joblib import Parallel, delayed
import re

import mwparserfromhell as mwparser
from mwparserfromhell.nodes import Tag, HTMLEntity, Text

from multiprocessing import cpu_count
from joblib import Parallel, delayed
import itertools

from ..mw import featurize


def load_and_featurize(line):
    """Parse JSON and add features for a revision"""
    revision = json.loads(line)
    revision.update(featurize(revision['wikitext']))
    return revision


def run(input_file, output_file, workers=1, n=None):
    """Load revision text and metadata, add features, and save to csv file."""
    with gzip.open(input_file, "rt") as f:
        pool = Parallel(n_jobs=workers, verbose=True)
        lines = itertools.islice(iter(f), n)
        revisions = pool(delayed(load_and_featurize)(line) for line in lines)
    df = pd.DataFrame.from_records(revisions)
    df = df.set_index('revid')
    df.to_csv(output_file, index=True, compression="gzip")

def main():
    output_file = "../data/enwiki.labeling_revisions.w_features.nettrom_30k.csv.gz"
    input_file = "../data/enwiki.labeling_revisions.w_text.nettrom_30k.ndjson.gz"
    workers = cpu_count() - 1
    n = None
    run(input_file, output_file, workers=workers, n=n)

if __name__ == "__main__":
    main()
