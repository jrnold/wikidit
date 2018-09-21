"""Add features to revisions in the test/train split."""
import gzip
import json
from multiprocessing import Pool, cpu_count
import argparse
import itertools

import articlequality.feature_lists.enwiki
from revscoring.datasources import revision_oriented
from revscoring.dependencies import solve


def add_features(observation):
    """Add features to a revision dictionary."""
    dependents = articlequality.feature_lists.enwiki.wp10
    cache = {revision_oriented.revision.text: observation['wikitext']}
    vals = list(solve(dependents, observation['wikitext'], cache=cache))
    vals = {str(d): x for d, x in zip(dependents, vals)}
    observation.update(vals)
    return observation


def main():
    """Read revisions with text from json dump and features."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=cpu_count() - 1,
                        help="Number of processes to use")
    parser.add_argument("--chunksize", type=int, default=1000,
                        help="Chunksize when parallel processing revisions.")
    parser.add_argument("-n", default=None, type=int,
                        help="Number of revisions to process.")
    args = parser.parse_args()
    with gzip.open('data/labelled.enwiki.w_text.ndjson.gz', 'rt') as f:
        data = [json.loads(line) for line in itertools.islice(f, args.n)]
    with Pool(args.workers) as pool:
        featurized = pool.map(add_features, data, chunksize=args.chunksize)
    with gzip.open("data/labelled.enwiki.w_features.ndjson.gz", "wt") as f:
        for line in featurized:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
