from typing import Generator

import pandas as pd

def read_ndjson(file):
    """Load lines from new-line delimited JSON file"""
    for line in file:
        yield json.loads(line)


def write_ndjson(data, file) -> None:
    """Load lines from new-line delimited JSON file."""
    for x in data:
        file.write(json.dumps(line) + "\n")


WP10_LABELS = ("Stub", "Start", "C", "B", "A", "GA", "FA")
"""Wikipeda WP10 Quality labels"""
 

def load_wp10(input_file):
    """Load wp10 data with features from a csv file"""
    revisions = pd.read_csv(input_file, index_col='revid')
    revisions['wp10'] = pd.Categorical(revisions['wp10'],
                                    categories=WP10_LABELS, ordered=True)
    return revisions
