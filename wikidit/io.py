import json

import pandas as pd


def load_ndjson(file):
    """Load lines from new-line delimited JSON file"""
    for line in file:
        yield json.loads(line)


def dump_ndjson(data, file):
    """Load lines from new-line delimited JSON file."""
    for x in data:
        file.write(json.dumps(line) + "\n")

