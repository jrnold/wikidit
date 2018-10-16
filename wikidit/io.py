import json
import gzip
import os.path
import os

import pandas as pd
from joblib import Parallel, delayed

from .preprocessing import WP10_DTYPE


def load_ndjson(file):
    """Load lines from new-line delimited JSON file"""
    for line in file:
        yield json.loads(line)


def dump_ndjson(data, file):
    """Load lines from new-line delimited JSON file."""
    for x in data:
        file.write(json.dumps(line) + "\n")


def read_labeled_one(filename):
    """Read a labeled revision file"""
    out = []
    with gzip.open(filename, "rt") as f:
        for line in f:
            row = json.loads(line)
            del row['wikitext']
            del row['text']
            out.append(row)
    return pd.DataFrame.from_records(out)


def read_labeled(dirname, n_jobs=6):
    """Read all labeled revisions files and concatenate into a single data frame"""
    filenames = [os.path.join(dirname, f) for f in os.listdir(dirname)]    
    revisions = pd.concat(Parallel(n_jobs=6)(delayed(read_labeled_one)(f) 
                                             for f in filenames))
    revisions['wp10'] = pd.Series(revisions['wp10'], dtype=WP10_DTYPE)
    return revisions