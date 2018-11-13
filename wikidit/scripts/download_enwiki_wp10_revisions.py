# coding: utf-8
"""Download metadata and texts for WP10 Quality sample."""
import gzip
import json
import os.path
import argparse

import mwapi

from ..utils import split_seq
from ..mw import Session


def run(input_file, output_file, chunksize=50):
    # Possible rvprop
    # - ids: Get the revid and, from 1.16 onward, the parentid. 1.11+
    # - roles: List content slot roles that exist in the revision. 1.32+
    # - flags: Whether the revision was a minor edit. 1.11+
    # - timestamp: The date and time the revision was made, in ISO 8601 combined date and time format.
    # - user: The user who made the revision, and if applicable, the flags: userhidden if revision deleted and/or anon if unregistered.
    # - userid: User id of revision creator, as well as userhidden and anon flags. 1.17+
    # - size: The size of the revision text in bytes. 1.11+
    # - sha1: SHA-1 (base 16) of the revision. 1.19+
    # - contentmodel: Content model id of the revision. 1.21+
    # - comment: The edit comment.
    # - parsedcomment: The edit/log comment in HTML format with wikilinks and section references expanded into hyperlinks 1.16+
    # - content: The revision content. If set, the maximum limit will be 10 times as low. (Note: If you want HTML rather than wikitext, use action=parse instead.)
    # - tags: Any tags for this revision, such as those added by AbuseFilter. 1.16+
    rvprop = "content|comment|sha1|size|userid|user|timestamp|flags|ids"

    session = Session()

    with open(input_file, "r") as f:
        revisions = {x["rev_id"]: x["wp10"] for x in [json.loads(line) for line in f]}

    if os.path.exists(output_file):
        raise FileExistsError(f"{output_file} exists")

    with gzip.open(output_file, "wt") as f:
        for i, rev_id in enumerate(split_seq(revisions, chunksize)):
            print(f"downloading chunk {i}")
            revids = "|".join(str(x) for x in rev_id)
            r = session.get(
                action="query",
                revids=revids,
                prop="revisions",
                rvprop=rvprop,
                rvslots="main",
            )
            pages = r["query"]["pages"]
            for page in pages.values():
                for revision in page["revisions"]:
                    # A few of these revisions have had their content removed
                    try:
                        revision["wikitext"] = revision["slots"]["main"]["*"]
                    except KeyError:
                        print(revision)
                        continue
                    del revision["slots"]
                    for k in ("pageid", "ns", "title"):
                        revision[k] = page[k]
                    f.write(json.dumps(revision) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
