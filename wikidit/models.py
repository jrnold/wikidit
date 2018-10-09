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
             add_count(page, 'words', 14),
              "Add a sentence (14 words)"),
             ('headings',
              add_per_word(page, 'headings', 1, 2),
              "Organize the article with a sub-heading"),
             ('sub_headings',
              add_per_word(page, 'sub_headings', 1, 2),
              "Organize the article with a sub-heading"),             
            ('images',
             add_per_word(page, 'images', 1, 0),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Images\"Add an image."),
            ('categories',
             add_per_word(page, 'categories', 1, 1),
             "<a href=\"https://en.wikipedia.org/wiki/Help:Category\">Add another category.</a>"),
            ('wikilinks',
             add_per_word(page, 'wikilinks', 1, 1),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:External_links\">Add a link to another page in Wikipedia.</a>"),
            ('external_links',
             add_per_word(page, 'external_links', 1, 1),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:External_links\">Add an external link.</a>"),
            ('citation', 
             add_per_word(page, 'cite_templates', 1, 3),
             "<a href=\"https://en.wikipedia.org/wiki/Wikipedia:Citing_sources\">Add a citation.</a>"),
            ('ref',
             add_per_word(page, 'ref_per_word', 1, 3),
             "<a href=\"https://en.wikipedia.org/wiki/Help:Footnotes#Footnotes:_the_basics\">Add a footnote.</a>"),
            ('coordinates', 
             add_binary(page, 'coordinates'),
             "Add coordinates."),
            ('infoboxes',
             add_binary(page, 'infoboxes'),
             "Add an infobox."),
            ('backlog_accuracy',
             add_count(page, 'backlog_accuracy', -1),
             "Fix a backlog issue related to accuracy."),
            ('backlog_other',
             add_count(page, 'backlog_other', -1),
             "Fix a backlog issue in the other category."),
            ('backlog_style',
             add_count(page, 'backlog_style', -1),
             "Fix a backlog issue relating to style."),
            ('backlog_links',
             add_count(page, 'backlog_links', -1),
             "Fix a backlog issue relating to links.")
            ]
    return edits


def predict_page_edits_api(title, model, featurizer=Featurizer(), session=None):
    if session is None:
        session = Session()
    page = get_page(session, title)
    return predict_page_edits(featurizer, page['content'], model)


def predict_page_edits(featurizer, content, pipeline):
    revision = featurizer.parse_content(content)
    del revision['text']

    revision = pd.DataFrame.from_records([revision])
    probs = list(pipeline.predict_proba(revision)[0, :])
    best_class = str(pipeline.predict(revision)[0])
    
    # If predicted to be FA - nothing else to do.
    if best_class == "FA":
        return {"predicted_class": best_class}
    
    # Create new pipeline for only that class
    pipe2 = Pipeline([('mapper', pipeline.named_steps['mapper']),
                      ('clf', pipeline.named_steps['clf'].named_estimators_[best_class])])

    # Predicted probability for > current predicted class
    prob_class = pipe2.predict_proba(revision)[0, 1]

    # Calc new probabilities for all types of edits
    edits = [(nm, pd.DataFrame.from_records([x])) 
             for nm, x in make_edits(revision.to_dict('records')[0])]
    new_probs = [(nm, pipe2.predict_proba(ed)[0, 1]) for nm, ed in edits]
    change_prob = [(nm, p - prob_class) for nm, p in new_probs]
    top_edits = sorted([(nm, p) for (nm, p) in change_prob if p > 0],
                       key=lambda x: -x[1])
    
    return {
        'predict': best_class,
        'proba': probs,
        'predicted_class_prob': prob_class,
        'change_prob': change_prob,
        'top_edits': top_edits
    }
