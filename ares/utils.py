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

