import itertools
import re
from collections import Counter
from typing import Generator, Optional

from mwxml import Dump, Revision
import mwparserfromhell as mwparser
import mwapi
from mwparserfromhell.wikicode import Wikicode, Wikilink, Template

from sklearn_ordinal import OrdinalClassifier


class Session(mwapi.Session):
    
    _HOSTNAME = "https://en.wikipedia.org"
    _USER_AGENT = "wikidit <jeffrey.arnold@gmail.com>"
    
    def __init__(self):
        super().__init__(self._HOSTNAME, user_agent=self._USER_AGENT)


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


def get_page(session, title):
    if title is None or title == '':
        return None
    params = {'action': 'query', 'titles': title, 'prop': "revisions",
              'redirects': True,
              'rvprop': 'ids|content|timestamp', "rvslots": "main"}
    r = session.get(**params)    
    page = list(r['query']['pages'].values())[0]
    # There is no such page!
    if 'missing' in page:
        return None
    rev = page['revisions'][0]
    rev['content'] = rev['slots']['main']['*']
    del rev['slots']
    rev['title'] = page['title']
    rev['pageid'] = page['pageid']
    return rev



def normalize_title(title, session=None):
    if session is None:
        session = Session()
    result = session.get(action="query", titles=title, redirects=True)
    page = list(result['query']['pages'].values())[0]
    if 'missing' in page:
        raise ValueError(f"Title {title} does not exist")
    else:
        return page['title']


def get_talk_page(title, session=None):
    if session is None:
        session = Session()
    norm_title = normalize_title(title, session=session)
    result = session.get(action='query', titles=f"Talk:{norm_title}",
                         prop='revisions', rvprop='content', 
                         rvslots='main')
    return list(result['query']['pages'].values())[0]


def get_content(page):
    return page['revisions'][0]['slots']['main']['*']


def clean_wp_class(x):
    replacements = {"disambig": "dab", 
                    "current": "cur",
                    "a": "ga",
                    "bplus": "b",
                    "none": None}

    # See https://en.wikipedia.org/wiki/MediaWiki:Gadget-metadata.js
    x = str(x).strip().lower()
    if x in replacements:
        x = replacements[x]
    return x


def clean_wp_importance(x):
    x = str(x).strip().lower()
    if x == "none":
        return None
    return x


def parse_project(tmpl):
    class_ = [x.value for x in tmpl.params if x.name == "class"]
    class_ = None if not len(class_) else class_[0]
    importance = [x.value for x in tmpl.params if x.name == "importance"]
    importance = None if not len(importance) else importance[0]
    return (str(tmpl.name), {'class': clean_wp_class(class_),
                             'importance': clean_wp_class(importance)})


def get_projects(page):
    """Extract WikiProject templates from a page"""
    return dict(parse_project(x) for x in page.filter_templates(matches="WikiProject"))


def get_wikiprojects(title, session=None):
    """Get all WikiProjects associated with a Wikipedia article"""
    # Problem: what if title doesn't exist
    # not sure if this handles cases where title is redirected
    page = get_talk_page(title, session=session)
    parsed = mwparser.parse(get_content(page))
    return get_projects(parsed)


def get_quality(title, session=None):
    """Get the Wikipedia quality assessment of an article"""
    CLASSES = {'fa': "FA", 'ga': "GA", 'b': "B", 'c': "C", 
               'start': "Start", 'stub': "Stub",
               'fl': "FL", 'list': "List", 'dab': "Disambig", 
               'book': "Book", 'template': "Template",
               'category': "Category", 'draft': "Draft", 'redirect': "Redirect",
               'na': "NA", "current": "Current", "future": "Future"}
    # current and future can be ignored
    projs = get_wikiprojects(title, session=session)
    classes_ = set()
    for _, vals in projs.items():
        k = vals['class']
        if k is not None:
            classes_.add(k)
    if not len(classes_):
        return None
    else:
        for k in CLASSES:
            if k in classes_:
                return CLASSES[k]
        return None

        