"""Functions related to preprocessing"""
import json
import os.path
import re
from collections import Counter
from functools import partial

import mwparserfromhell as mwparser
import mwapi
import pandas as pd
from joblib import Parallel, delayed

import spacy
from spacy.lang.en import English

from .mw import match_template, wikilink_title_matches

# Backlog data
def _load_backlog():
    filename = os.path.join(os.path.dirname(__file__), 'backlog.json')
    with open(filename, 'r') as f:
        return json.load(f)


backlog_templates = _load_backlog()

def _backlog_featurizer():
    backlog_issues = {}
    exclude = ("Too few wikilinks", )
    sections = ("Accuracy", "Content", "Style", "File Description", "File Quality", 
                "Other", "Links")
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


backlog_features = _backlog_featurizer()


def get_backlog_features(doc):
    """Get backlog features from templates"""
    template_counts = Counter([str(x.name).strip().lower() for x in doc.filter_templates()])
    features = {k: {} for k in set(backlog_features.values())}
    for name, count in template_counts.items():
        if name.lower() in backlog_features:
            features[backlog_features[name]][name] = count
    return features
    

WP10_LABELS = ("Stub", "Start", "C", "B", "GA", "FA")
"""Wikipeda WP10 Quality labels"""


WP10_DTYPE = pd.api.types.CategoricalDtype(categories=WP10_LABELS, ordered=True)
"""Categorical data type for WP10 quality labels."""

def is_word(token):
    return not (token.is_space or token.is_punct)
    

class Featurizer:
    
    def __init__(self, nlp_model='en'):
        self.nlp = English()
        self.parser = mwparser.parser.Parser()


    def featurize(self, x, content='content'):
        x = x.copy()
        features = self.parse_content(x[content])
        x.update(features)
        return x

    def parse_content(self, content):
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
        PAT_WORD = r"\b\w+\b"
        PAT_PARA = r"\n{2,}"
        
        revision = {}

        # Content characters are visible characters. Operationalized as characters after
        # mwparserfromhell calls the ``strip_code()`` method.

        # contentwords = [node.strip_code() for node in text.ifilter(content, forcetype=(Tag, Text, HTMLEntity))
        plaintext = text.strip_code()

        ## Real Content

        # Sections
        words = [tok for tok in self.nlp(plaintext) if is_word(tok)]
        word_count = len(words)
        # word_count = len(re.findall(PAT_WORD, plaintext))
        # parsed = nlp(plaintext)
        # words = [tok for tok in parsed if not (tok.is_space or tok.is_punct)]
        # word_count = len(words)
        # sections = text.get_sections(flat=True, include_lead=True, include_headings=False)
        # total_paras = 0
        # non_ref_paras = 0
        # word_count = 0
        # word_chars = 0
        #     for i, section in enumerate(sections):
        #         sectxt = section.strip_code()
        #         # split into paragraphs
        #         paras = re.split(PAT_PARA, sectxt)
        #         # count any paragraphs without references
        #         for p in paras:
        #             if not re.search("<ref>", p):
        #                 non_ref_paras += 1
        #         # count words
        #         parawords = re.findall(PAT_WORD, sectxt)
        #         word_count += len(parawords)
        #         word_chars += sum(len(w) for w in parawords)
        #         total_paras += len(paras)
        #         if (i == 0):
        #             lead_paras = len(paras)
        # revision['total_paras'] = total_paras
        # revision['lead_paras'] = lead_paras
        # revision['non_ref_paras'] = tnon_ref_paras
        revision['words'] = word_count
        # revision['average_word_length'] = word_count / word_chars if word_count > 0 else 0

        ## Headings

        # Total number of headings
        headings = text.filter_headings()
        revision['headings'] = len([x for x in headings if x.level == 2])

        # Sub-headings
        revision['sub_headings'] = len([x for x in headings if x.level > 2])

        ## Number of wikilinks
        wikilinks = text.filter_wikilinks()

        # number of images
        revision['images'] = sum([wikilink_title_matches(r"file|image\:", link) for link in wikilinks])

        # number of categories
        revision['categories'] = sum([wikilink_title_matches("category\:", link) for link in wikilinks])

        # Other wikilinks
        special_links = sum(revision[k] for k in ("images", "categories"))
        revision['wikilinks'] = len(wikilinks) - special_links

        # external links
        revision['external_links'] = len(text.filter_external_links())

        ## Templates
        templates = text.filter_templates()

        # main templates
        revision['main_templates'] = sum([match_template(x, "main$") for x in templates])

        # citation templates
        revision['cite_templates'] = sum([match_template(x, "cite") for x in templates])

        # infoboxes
        revision['infoboxes'] = sum([match_template(x, "infobox") for x in templates])

        # other templates
        revision['templates'] = len(templates)
      
        backlog_sections = {"Accuracy": "backlog_accuracy", 
                            "Content": "backlog_content", 
                            "Style":  "backlog_style",
                            "File": "backlog_files", 
                            "Other": "backlog_other", 
                            "Links": "backlog_links"}
        backlog_issues = get_backlog_features(text)
        for k, v in backlog_sections.items():
            if len(backlog_issues[k]):
                revision[v] = sum(backlog_issues[k].values())
                revision[f"{v}_templates"] = ' '.join(backlog_issues[k].keys())
            else:
                revision[v] = 0
                revision[f"{v}_templates"] = None
        
        # number of ref tags
        revision['ref'] = len([x for x in text.filter_tags() if x.tag == "ref"])

        # number of smartlists (e.g. wikitables)
        revision['smartlists'] = len([x for x in text.nodes
                                      if isinstance(x, mwparser.smart_list.SmartList)])

        # is there a map
        revision['coordinates'] = '#coordinates' in str(text).lower()
        
        # Add normalized versions for many of these
        for feat in ('headings', 'sub_headings', 'main_templates', 'external_links', 'wikilinks',
                     'cite_templates', 'templates', 'ref', 'images', 'categories',
                     'smartlists'):
            if word_count > 0:
                revision[f"{feat}_per_word"] = revision[feat] / word_count
            else:
                revision[f"{feat}_per_word"] = 0.

        # Add plaintext for more features
        revision['text'] = plaintext

        return revision


def load_wp10(input_file):
    """Load wp10 data with features from a csv file"""
    revisions = pd.read_csv(input_file, index_col='revid')
    revisions['wp10'] = pd.Categorical(revisions['wp10'],
                                       categories=WP10_LABELS, ordered=True)
    return revisions


def sort_wp10(x, classes):
    classes = list(classes)
    return sorted(list(zip(classes, x)), key=lambda x: WP10_LABELS.index(x[0]))
