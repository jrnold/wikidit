import itertools
import re

import mwparserfromhell as mwparser
import mwapi
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from .mw import match_template, wikilink_title_matches, Session, get_page
from .preprocessing import Featurizer


def add_words(x, i):
    x = x.copy()
    x['words'] += i
    return x


def add_per_word(x, col, i, w):
    x = x.copy()
    words = x['words']
    x['words'] += w * i
    x[col] += i
    if x['words'] > 0:
        x[f"{col}_per_word"] = x[col] / x['words']
    return x


def add_count(x, col, i):
    x = x.copy()
    x[col] += i
    if x[col] < 0:
        x[col] = 0
    return x


def add_binary(x, col):
    x = x.copy()
    x[col] = bool(x[col]) or True
    return x


def make_edits(page):
    edits = [('sentence',
             add_count(page, 'words', 15),
              "Add a sentence (15 words)"),
             ('paragraph',
              add_count(page, 'words', 150),
               "Add a paragraph(150 words)"),
             ('headings',
              add_per_word(page, 'headings', 1, 2),
              "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style#Article_titles,_headings,_and_sections\">Organize the article with a heading</a>"),
             ('sub_headings',
              add_per_word(page, 'sub_headings', 1, 2),
              "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style#Article_titles,_headings,_and_sections\">Organize the article with a sub-heading</a>"),
            ('images',
             add_per_word(page, 'images', 1, 0),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Images\">Add an image</a>"),
            ('categories',
             add_per_word(page, 'categories', 1, 2),
             "<a href=\"https://en.wikipedia.org/wiki/Help:Category\">Add another category.</a>"),
            ('wikilinks',
             add_per_word(page, 'wikilinks', 1, 1),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:External_links\">Add a link to another page in Wikipedia</a>"),
            ('external_links',
             add_per_word(page, 'external_links', 1, 1),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:External_links\">Add an external link</a>"),
            ('citation',
             add_per_word(page, 'cite_templates', 1, 5),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Citing_sources\">Add a citation</a>"),
            ('ref',
             add_per_word(page, 'ref_per_word', 1, 15),
             "<a href=\"https://en.wikipedia.org/wiki/Help:Footnotes#Footnotes:_the_basics\">Add a footnote</a>"),
            ('coordinates',
             add_binary(page, 'coordinates'),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Geographical_coordinates#Coordinate_templates\">Add coordinates.</a>"),
            ('infoboxes',
             add_binary(page, 'infoboxes'),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Infoboxes\">Add an infobox</a>"),
            ('backlog_accuracy',
             add_count(page, 'backlog_accuracy', -1),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Backlog\">Fix a backlog issue related to accuracy</a>"),
            ('backlog_other',
             add_count(page, 'backlog_other', -1),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Backlog\">Fix a backlog issue in the other category</a>"),
            ('backlog_style',
             add_count(page, 'backlog_style', -1),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Backlog\">Fix a backlog issue relating to style</a>"),
            ('backlog_links',
             add_count(page, 'backlog_links', -1),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Backlog\">Fix a backlog issue relating to links</a>")
            ]
    return edits


def predict_page_edits_api(title, model, featurizer=Featurizer(),
                           session=Session()):
    page = get_page(title, session)
    return predict_page_edits(featurizer, page['content'], model)


def qual_score(prob):
    return (prob * np.arange(1, prob.shape[1] + 1)).sum()


def predict_page_edits(featurizer, content, model):
    revision = featurizer.parse_content(content)
    del revision['text']
    revision = pd.DataFrame.from_records([revision])
    # revision = create_features(revision)

    # probabilities for current class
    prob = model.predict_proba(revision)
    best = str(model.predict(revision)[0])
    score = qual_score(prob)

    # Calc new probabilities for all types of edits
    edits = [(nm, description, pd.DataFrame.from_records([x]))
             for nm, x, description in make_edits(revision.to_dict('records')[0])]
    edit_probs = [(nm, description, model.predict_proba(ed)) for nm, description, ed in edits]
    edit_scores = [(nm, description, qual_score(p)) for nm, description, p in edit_probs]
    edit_changes = [(n, d, s - score) for n, d, s in edit_scores]
    top_edits = sorted([x for x in edit_changes if x[2] > 0], key=lambda x: -x[2])
    
    return {
        'prob': list(zip(model.classes_, list(prob.ravel()))),
        'score': score,
        'edit_probs': edit_probs,
        'edit_scores': edit_scores,
        'top_edits': top_edits,
        'edits': edits,
        'best': best
    }


_COUNT_COLS = ['words',
             # infobox as a binary
             'backlog_accuracy',
             'backlog_content',
             'backlog_other',
             'backlog_style',
             'backlog_links']

_PER_WORD_COLS = [
             'headings_per_word',
             'sub_headings_per_word',
             # links
             'images_per_word',
             'categories_per_word',
             'wikilinks_per_word',
             'external_links_per_word',
             # templates
             'main_templates_per_word',
             'cite_templates_per_word',
             'ref_per_word'
]

_BINARY_COLS = ['coordinates', 'infoboxes']

_RESPONSE_COL = ['wp10']

def create_features(df):
    """Create features needed for the data frame"""
    # I do this once rather than multiple times for each data frame
    df = df.copy()
    for c in _COUNT_COLS:
        df[c] = np.sqrt(df[c])
    for c in _BINARY_COLS:
        df[c] = df[c].astype(bool)
    allcols = list(itertools.chain(_PER_WORD_COLS, _COUNT_COLS, _BINARY_COLS))
    return df.loc[:, allcols]
