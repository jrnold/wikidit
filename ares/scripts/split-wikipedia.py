"""Script to extract plain text files from a Wikipedia dump.

- output_dir
    - text
        - revid.txt
    - parsed
        - revid.pk
    - tmp
        - revisions/batch.csv
        - templates/batch.csv
        - wikilinks/batch.csv

"""
import logging
import os
import os.path
from itertools import islice
import argparse
from typing import Optional
import dill
from typing import List, Tuple
import csv

import mwxml
import mwparserfromhell as mwparser
from toolz.itertoolz import partition_all
from joblib import Parallel, delayed
from smart_open import smart_open

from ..utils import (iter_revisions, revision_to_dict, template_counts,
                     wikilinks_counts)

import json

LOGGER = logging.getLogger(__name__)


FIELDNAMES = (
    'rev_id',
    'timestamp',
    'user_id',
    'user_text',
    'page_id',
    'page_title',
    'page_namespace',
    'page_redirect',
    'minor',
    'comment',
    'sha1',
    'parent_id',
    'deleted_text',
    'deleted_comment',
    'deleted_user'
)


def process_wikilink(revid: int, title: str, count: int):
    """Create dict for wikilinks."""
    return {'revid': revid, 'title': title, 'count': count}


def process_template(revid: int, name: str, count: int):
    """Create dict for templates."""
    return {'revid': revid, 'name': name, 'count': count}


def process_revision(revision: mwxml.Revision, output: dict) -> dict:
    """Parse a single wikpedia revision."""
    if revision.page.redirect:
        return
    rev_id = revision.id
    LOGGER.debug(f"Parsing revision {rev_id}")
    revdict = revision_to_dict(revision)
    text = revdict['text']
    del revdict['text']
    parsed = mwparser.parse(text)
    # dump files
    filename_text = os.path.join(output['texts'], f"{rev_id}.txt")
    LOGGER.debug(f"Writing to {filename_text}")
    with open(filename_text, "w") as f:
        f.write(text)
    # dump files
    filename_parsed = os.path.join(output['parsed'], f"{rev_id}.pkl")
    LOGGER.debug(f"Writing to {filename_parsed}")
    with open(filename_parsed, "wb") as f:
        dill.dump(parsed, f)
    return revdict


def create_dirs(output_dir: str) -> dict:
    """Create the output directories needed for this script"""
    if not os.path.exists(output_dir):
        logging.info(f"Creating {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    out = {'root': output_dir}
    subdirs = ('texts', 'parsed')
    for d in subdirs:
        dirname = os.path.join(output_dir, d)
        os.makedirs(dirname, exist_ok=True)
        out[d] = dirname
    return out


def run(input_file: str, output_dir: str, workers: Optional[int]=None,
        verbose: bool=False, max_docs: Optional[int]=None) -> None:
    workers = workers or max(os.cpu_count() - 1, 1)
    output = create_dirs(output_dir)
    with smart_open(input_file, "r") as f:
        dump = mwxml.Dump.from_file(f)
        corpus = islice(iter_revisions(dump), max_docs)
        with Parallel(n_jobs=workers, backend='threading') as parallel:
            revisions = parallel(delayed(process_revision)(rev, output)
                                 for rev in corpus)
    with open(os.path.join(output_dir, "metadata.csv"), 'w') as f:
        writer = csv.DictWriter(f, FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(r for r in revisions if r is not None)


def main() -> None:
    """Extract plain text files from a wikipedia dump."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to wikipedia dump", nargs=1,
                        type=str)
    parser.add_argument(
        "output_dir", help="Output directory for cleaned files", type=str,
        nargs=1)
    parser.add_argument(
        "--workers",
        "-w",
        help="Number of workers to use",
        type=int,
        default=max(os.cpu_count() - 1, 1))
    parser.add_argument(
        "--max-docs",
        "-D",
        help="Maximum number of documents to process",
        type=int)
    parser.add_argument(
        "--verbose", "-v", help="More verbose logging", action="store_true")
    args = parser.parse_args()
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel)
    run(args.input_file[0],
        args.output_dir[0],
        workers=args.workers,
        max_docs=args.max_docs)


if __name__ == '__main__':
    main()
