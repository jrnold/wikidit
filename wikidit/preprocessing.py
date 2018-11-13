"""Functions related to preprocessing revisions."""
import json
import os.path
from collections import Counter
from typing import Dict

import mwparserfromhell as mwparser
import pandas as pd
from spacy.lang.en import English
# import en_core_web_md

from .mw import match_template, wikilink_title_matches

# NLP = en_core_web_md.load(disable=["ner", "parser"])
# We're only using the tokenizer at the moment. However, leave it open to use
# word vectors.
NLP = English()


# Backlog data
def _load_backlog() -> Dict:
    filename = os.path.join(os.path.dirname(__file__), "backlog.json")
    with open(filename, "r") as f:
        return json.load(f)


backlog_templates: Dict = _load_backlog()


def _backlog_featurizer() -> Dict:
    backlog_issues = {}
    exclude = ("Too few wikilinks",)
    sections = (
        "Accuracy",
        "Content",
        "Style",
        "File Description",
        "File Quality",
        "Other",
        "Links",
    )
    for sec in sections:
        for subsection, templates in backlog_templates[sec].items():
            if subsection in exclude:
                continue
            # Combine file description and quality
            if sec in ("File Description", "File Quality"):
                secname = "File"
            else:
                secname = sec
            for name, redirs in templates.items():
                backlog_issues[name.lower()] = secname
                for r in redirs:
                    backlog_issues[r.lower()] = secname
    return backlog_issues


backlog_features: Dict = _backlog_featurizer()


def get_backlog_features(doc) -> Dict:
    """Get backlog features from templates."""
    template_counts = Counter(
        [str(x.name).strip().lower() for x in doc.filter_templates()]
    )
    features = {k: {} for k in set(backlog_features.values())}
    for name, count in template_counts.items():
        if name.lower() in backlog_features:
            features[backlog_features[name]][name] = count
    return features


WP10_LABELS: str = ("Stub", "Start", "C", "B", "GA", "FA")
"""Wikipeda WP10 Quality labels"""


WP10_DTYPE: str = pd.api.types.CategoricalDtype(categories=WP10_LABELS, ordered=True)
"""Categorical data type for WP10 quality labels."""


def is_word(token) -> bool:
    return not (token.is_space or token.is_punct)


class Featurizer:
    """Add common features to a revision."""
    # THis is implemented as a class rather than a function in order
    nlp = NLP

    def __init__(self) -> None:
        self.parser = mwparser.parser.Parser()

    def featurize(self, x: Dict, content: str="content") -> Dict:
        x = x.copy()
        features = self.parse_content(x[content])
        x.update(features)
        return x

    def parse_content(self, content: str) -> Dict:
        """Create features for each revision

        Parameters
        -----------
        content:
            The content of the revision.

        Returns
        --------
        dict:
            The revision with features as a dictionary.

        """
        text = self.parser.parse(content)

        revision = {}

        # Content characters are visible characters. Operationalized as characters after
        plaintext = text.strip_code()

        # Real Content

        # Sections
        words = [tok for tok in self.nlp(plaintext) if is_word(tok)]
        # always at least one word
        revision["words"] = len(words) + 1

        # Headings

        # Total number of headings
        headings = text.filter_headings()
        revision["headings"] = len([x for x in headings if x.level == 2])

        # Sub-headings
        revision["sub_headings"] = len([x for x in headings if x.level > 2])

        # Number of wikilinks
        wikilinks = text.filter_wikilinks()

        # number of images
        revision["images"] = sum(
            [wikilink_title_matches(r"file|image\:", link) for link in wikilinks]
        )

        # number of categories
        revision["categories"] = sum(
            [wikilink_title_matches("category\:", link) for link in wikilinks]
        )

        # Other wikilinks
        special_links = sum(revision[k] for k in ("images", "categories"))
        revision["wikilinks"] = len(wikilinks) - special_links

        # external links
        revision["external_links"] = len(text.filter_external_links())

        # Templates
        templates = text.filter_templates()

        # main templates
        revision["main_templates"] = sum(
            [match_template(x, "main$") for x in templates]
        )

        # citation templates
        revision["cite_templates"] = sum([match_template(x, "cite") for x in templates])

        # infoboxes
        revision["infoboxes"] = sum([match_template(x, "infobox") for x in templates])

        # other templates
        revision["templates"] = len(templates)

        backlog_sections = {
            "Accuracy": "backlog_accuracy",
            "Content": "backlog_content",
            "Style": "backlog_style",
            "File": "backlog_files",
            "Other": "backlog_other",
            "Links": "backlog_links",
        }
        backlog_issues = get_backlog_features(text)
        for k, v in backlog_sections.items():
            if len(backlog_issues[k]):
                revision[v] = sum(backlog_issues[k].values())
                revision[f"{v}_templates"] = " ".join(backlog_issues[k].keys())
            else:
                revision[v] = 0
                revision[f"{v}_templates"] = None

        # number of ref tags
        revision["ref"] = len([x for x in text.filter_tags() if x.tag == "ref"])

        # number of smartlists (e.g. wikitables)
        revision["smartlists"] = len(
            [x for x in text.nodes if isinstance(x, mwparser.smart_list.SmartList)]
        )

        # is there a map
        revision["coordinates"] = "#coordinates" in str(text).lower()

        # Add plaintext for more features
        revision["text"] = plaintext

        # Add vectors
        # revision['wordvec_{i}']

        return revision


def load_wp10(input_file: str) -> pd.DataFrame:
    """Load wp10 data with features from a csv file"""
    revisions = pd.read_csv(input_file, index_col="revid")
    revisions["wp10"] = pd.Categorical(
        revisions["wp10"], categories=WP10_LABELS, ordered=True
    )
    return revisions
