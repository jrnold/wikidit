"""Classes and methods for fitting and predicting models."""
from typing import List, Dict, Tuple
import os.path

import pandas as pd
import numpy as np
import dill
from sklearn.base import BaseEstimator, TransformerMixin

from .mw import Session, get_page
from .preprocessing import Featurizer, WP10_LABELS

_MODEL_FILE = 'xgboost-sequential.pkl'
_MODEL_PATH = os.path.join(os.path.dirname(__file__), _MODEL_FILE)


class RevisionPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer to preprocess revisions."""

    PER_WORD_COLS = [
        "headings",
        "sub_headings",
        "main_templates",
        "external_links",
        "wikilinks",
        "cite_templates",
        "templates",
        "ref",
        "images",
        "categories",
        "smartlists",
    ]
    """Columns which should be normalized to x per word."""

    BINARY_COLS = ["coordinates", "infoboxes"]
    """Columns which should be transformed to booleans."""

    KEEP = [
        "words",
        "backlog_accuracy",
        "backlog_content",
        "backlog_other",
        "backlog_style",
        "backlog_links",
        *PER_WORD_COLS,
        *(f"{x}_per_word" for x in PER_WORD_COLS),
        *BINARY_COLS,
    ]
    """Names of columns to keep"""

    def fit(self, X: Dict, y=None):
        """Does nothing."""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for feat in self.PER_WORD_COLS:
            X[f"{feat}_per_word"] = X[feat] / X["words"]
        for feat in self.BINARY_COLS:
            X[feat] = X[feat].astype(bool).astype(int)
        return X[self.KEEP]


def add_count(x: Dict, col: str, i: int) -> Dict:
    """Add ``i`` to a non-negative count variable ``x[col]``."""
    # this is needed so that subtracting 1 does not go below zero.
    x = x.copy()
    x[col] += i
    x[col] = max(x[col], 0)
    return x


def add_words(x: Dict, i: int) -> Dict:
    """Add ``i`` words to ``x[\"words\"]``."""
    x = x.copy()
    x["words"] += i
    x["words"] = max(x["words"], 1)
    return x


def add_per_word(x: Dict, col: str, i: int, w: int) -> Dict:
    """Add to a per word value in a revision.

    Parameters
    -----------
    x: dict
        Revision

    col: str
        Column name

    i: int
        Number to add to the column value

    w: int
        Number of words added/subtracted from a revision when it changes.

    Returns
    --------
    dict
        A copy of ``x`` with the appropriate changes. This does not
        alter ``x`` in place.
    """
    x = add_words(x, w)
    x[col] += i
    x[col] = max(x[col], 0)
    return x


def add_binary(x: Dict, col: str) -> Dict:
    ""
    x = x.copy()
    x[col] = bool(x[col]) or True
    return x


def make_edits(page: Dict) -> List[Tuple[str, Dict, str]]:
    return [
        ("sentence", add_words(page, 15), "Add a sentence (15 words)"),
        ("paragraph", add_words(page, 150), "Add a paragraph (150 words)"),
        (
            "headings",
            add_per_word(page, "headings", 1, 2),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style#'
            'Article_titles,_headings,_and_sections">Organize the article with a '
            "heading</a>",
        ),
        (
            "sub_headings",
            add_per_word(page, "sub_headings", 1, 2),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style#'
            'Article_titles,_headings,_and_sections">Organize the article with a '
            "sub-heading</a>",
        ),
        (
            "images",
            add_per_word(page, "images", 1, 0),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Images">'
            "Add an image</a>",
        ),
        (
            "categories",
            add_per_word(page, "categories", 1, 2),
            '<a href="https://en.wikipedia.org/wiki/Help:Category">Add another'
            " category.</a>",
        ),
        (
            "wikilinks",
            add_per_word(page, "wikilinks", 1, 1),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:External_links">'
            "Add a link to another page in Wikipedia</a>",
        ),
        (
            "external_links",
            add_per_word(page, "external_links", 1, 1),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:External_links">'
            "Add an external link</a>",
        ),
        (
            "citation",
            add_per_word(page, "cite_templates", 1, 5),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Citing_sources">Add a '
            "citation</a>",
        ),
        (
            "ref",
            add_per_word(page, "ref", 1, 15),
            '<a href="https://en.wikipedia.org/wiki/Help:Footnotes#'
            'Footnotes:_the_basics">Add a footnote</a>',
        ),
        (
            "coordinates",
            add_binary(page, "coordinates"),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:'
            'WikiProject_Geographical_coordinates#Coordinate_templates">'
            "Add coordinates.</a>",
        ),
        (
            "infoboxes",
            add_binary(page, "infoboxes"),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/'
            'Infoboxes">'
            "Add an infobox</a>",
        ),
        (
            "backlog_accuracy",
            add_count(page, "backlog_accuracy", -1),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Backlog">'
            "Fix a backlog issue related to accuracy</a>",
        ),
        (
            "backlog_other",
            add_count(page, "backlog_other", -1),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Backlog">'
            "Fix a backlog issue in the other category</a>",
        ),
        (
            "backlog_style",
            add_count(page, "backlog_style", -1),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Backlog">'
            "Fix a backlog issue relating to style</a>",
        ),
        (
            "backlog_links",
            add_count(page, "backlog_links", -1),
            '<a href="https://en.wikipedia.org/wiki/Wikipedia:Backlog">'
            "Fix a backlog issue relating to links</a>",
        ),
    ]


def predict_page_edits_api(
    title, model, mapper, featurizer=Featurizer(), session=Session()
):
    page = get_page(title, session)
    return predict_page_edits(page["content"], featurizer, model)


def qual_score(prob: np.array) -> float:
    return (prob * np.arange(prob.shape[1])).sum()


def predict_page_edits(content: str, featurizer: Featurizer, model) -> Dict:
    revision = featurizer.parse_content(content)

    del revision["text"]
    revision_df = pd.DataFrame.from_records([revision])

    # probabilities for current class
    prob = model.predict_proba(revision_df)
    best = model.predict(revision_df)[0]
    score = qual_score(prob)

    # Calc new probabilities for all types of edits
    edits = [
        (nm, description, pd.DataFrame.from_records([x]))
        for nm, x, description in make_edits(revision_df.to_dict("records")[0])
    ]
    edit_probs = [
        (nm, description, model.predict_proba(ed)) for nm, description, ed in edits
    ]
    edit_scores = [
        (nm, description, qual_score(p)) for nm, description, p in edit_probs
    ]
    edit_changes = [(n, d, s - score) for n, d, s in edit_scores]
    top_edits = sorted([x for x in edit_changes if x[2] > 0], key=lambda x: -x[2])

    return {
        "prob": list(zip(list(WP10_LABELS), list(prob.ravel()))),
        "score": score,
        "edit_probs": edit_probs,
        "edit_scores": edit_scores,
        "top_edits": top_edits,
        "edits": edits,
        "best": WP10_LABELS[best],
    }


def load_model():
    """Load the trained quality prediction model."""
    with open(_MODEL_PATH, 'rb') as f:
        model = dill.load(f)
    return model
