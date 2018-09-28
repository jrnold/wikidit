import itertools
import re
from collections import Counter
from typing import Generator, Optional

from mwxml import Dump, Revision
import mwparserfromhell as mwparser
from mwparserfromhell.wikicode import Wikicode, Wikilink, Template


def iter_revisions(dump: Dump, max_pages: Optional[int]=None) -> Generator[Revision, None, None]:
    """Iterate over revisions of Wikipedia dump."""
    for page in itertools.islice(dump.pages, max_pages):
        for rev in page:
            yield rev


def revision_to_dict(rev: Revision) -> dict:
    """Convert a revision object to a flattened dict object"""
    rev = rev.to_json()
    rev['rev_id'] = rev['id']
    del rev['id']
    for k in ("id", "title", "namespace", 'redirect'):
        if k != "restrictions":
            try:
                rev[f"page_{k}"] = rev['page'][k]
            except KeyError:
                rev[f"page_{k}"] = None
        else:
            try:
                rev[f"page_restrictions"] = " ".join(str(x)
                    for x in rev['page'][k])
            except KeyError:
                rev[f"page_restrictions"] = None
    del rev["page"]
    for k, v in rev["deleted"].items():
        rev[f"deleted_{k}"] = v
    del rev["deleted"]
    for k in ("id", "text"):
        try:
            rev[f"user_{k}"] = rev["user"][k]
        except KeyError:
            rev[f"user_{k}"] = None
    del rev["user"]
    rev['is_revision'] = 'page_redirect' in rev
    if 'comment' not in rev:
        rev['comment'] = ''
    return rev


def template_counts(wikicode: Wikicode) -> Counter:
    """Count unique templates in a wikicode object"""
    return Counter(str(x.name).strip()
                   for x in wikicode.ifilter_templates())


def wikilinks_counts(wikicode: Wikicode) -> Counter:
    """Count unique wikilinks in a Wikicode object."""
    return Counter(str(x.title).strip()
                   for x in wikicode.ifilter_wikilinks())


def num_headings(text: str, level: int) -> int:
    """Count number of headings in wikicode document with ``level``"""
    return len([x for x in text.filter_headings() if x.level == level])


def clean_template_name(x: Template):
    """Return the cleaned and standardized template name"""
    return str(x.name).strip().lower().replace(" ", "_").replace("-", "_")


def match_template(x, pattern: str) -> bool:
    """Does the object match ``pattern``"""
    return bool(re.match(pattern, clean_template_name(x), re.I))


def wikilink_title_matches(pattern: str, link: str) -> bool:
    """Does wikilink title match ``pattern``"""
    return bool(re.match(pattern, str(link.title), re.I))
