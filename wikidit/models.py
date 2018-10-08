import re

import mwparserfromhell as mwparser
import mwapi
import pandas as pd
import numpy as np

from .mw import match_template, wikilink_title_matches
from .preprocessing import Featurizer

def add_words(x, i):
    x = x.copy()
    x['words'] += i
    return x

def add_per_word(x, col, i, w):
    x = x.copy()
    x[col] += i
    x['words'] += w * i
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
    x[col] = x[col] or True
    return x


def make_edits(page):
    edits = [('words',
             add_count(page, 'words', 14)),
             ('headings',
              add_per_word(page, 'headings', 1, 2)),
             ('sub_headings',
              add_per_word(page, 'sub_headings', 1, 2)),             
            ('images',
             add_per_word(page, 'images', 1, 0)),
            ('categories',
             add_per_word(page, 'categories', 1, 1)),
            ('wikilinks',
             add_per_word(page, 'wikilinks', 1, 1)),
            ('external_links',
             add_per_word(page, 'external_links', 1, 1)),
            ('citation', 
             add_per_word(page, 'cite_templates', 1, 3)),
            ('ref',
             add_per_word(page, 'ref_per_word', 1, 3)),
            ('coordinates', 
             add_binary(page, 'coordinates')),
            ('infoboxes',
             add_binary(page, 'infoboxes')),
            ('backlog_accuracy',
             add_count(page, 'backlog_accuracy', -1)),
            ('backlog_other',
             add_count(page, 'backlog_other', -1)),
            ('backlog_style',
             add_count(page, 'backlog_style', -1)),
            ('backlog_links',
             add_count(page, 'backlog_links', -1))
            ]
    return edits

def generate_predictions(model, page):

    def get_probs(X):
        return sort_wp10(list(model.predict_proba(X).squeeze()),
                         list(model.classes_))

    def get_wtavg(X):
        x = [x[1] * (i + 1) for i, x in enumerate(get_probs(X))]
        return sum(x)

    best_prediction = model.predict(page.data)
    pred_probs = get_probs(page.data)
    wt_avg = get_wtavg(page.data)
    interventions = {k: get_wtavg(v) - wt_avg
                     for k, v in dict(page.edits()).items()}
    return {
            'best': best_prediction[0],
            'avg': wt_avg,
            'proba': pred_probs,
            'interventions': interventions
            }
