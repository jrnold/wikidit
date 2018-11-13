"""Add features to the training/test data."""
import gzip
import argparse
import json
import logging
import os.path

from joblib import Parallel, delayed

from ..io import load_ndjson
from ..preprocessing import Featurizer, WP10_LABELS

logger = logging.getLogger(__name__)


def read_labeling_revisions(filename):
    groups = {k: [] for k in WP10_LABELS}
    with gzip.open(filename, "rt") as f:
        for row in load_ndjson(f):
            groups[row["wp10"]].append(row)
    return groups


def process(output_dir, wp10, data):
    featurizer = Featurizer()
    filename = os.path.join(output_dir, f"{wp10}.ndjson.gz")
    with gzip.open(filename, "wt") as f:
        for x in data:
            newx = featurizer.featurize(x, content="wikitext")
            f.write(json.dumps(newx) + "\n")


def run(input_file, output_dir, n_jobs=1):
    if os.path.exists(output_dir):
        logging.warning(f"{output_dir} already exists")
    else:
        logging.info(f"Creating {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    data = read_labeling_revisions(input_file)
    exc = Parallel(n_jobs=n_jobs)
    exc(delayed(process)(output_dir, *x) for x in data.items())


def main():
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("-j", "--n-jobs", type=int, default=1)
    args = parser.parse_args()
    run(args.input, args.output, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
