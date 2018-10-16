from flask import Flask, render_template, request, Markup

import dill
import urllib.parse
import numpy as np
import os.path
import dill

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline

import pandas as pd

from wikidit.mw import Session, get_page, get_quality
from wikidit.models import Featurizer, predict_page_edits
from wikidit.preprocessing import WP10_LABELS

app = Flask(__name__)

# Load instances that should only be loaded once
MODEL_FILE = os.path.join("models", "xgboost-sequential.pkl")
with open(MODEL_FILE, "rb") as f:
    MODEL = dill.load(f)
    
featurizer = Featurizer()


def wikipedia_url(title, lang="en", revid=None):
    qtitle = urllib.parse.quote(title)
    if revid is None:
        out = f"https://{lang}.wikipedia.org/wiki/{title}"
    else:
        out = f"https://{lang}.wikipedia.org/w/index.php?title={title}&oldid={revid}"
    return out

# Create a single session instance to reuse
def get_next_quality_cat(cat):
    i = WP10_LABELS.index(cat)
    if i < (len(WP10_LABELS) - 1):
        return WP10_LABELS[i + 1]

@app.route('/')
def index():
    return render_template("index.html")

QA = {
    "FA": {"link": "https://en.wikipedia.org/wiki/Wikipedia:Featured_articles",
           "name": "Featured article",
           "tag": "FA"},
    "GA": {"href": "https://en.wikipedia.org/wiki/Wikipedia:Good_articles",
           "name": "Good article",
           "tag": "GA"},
    "B": {"href": "https://en.wikipedia.org/wiki/Category:B-Class_articles",
          "name": "B-class article",  "tag": "B"},
    "C": {"href": "https://en.wikipedia.org/wiki/Category:C-Class_articles",
          "name": "C-class article", "tag": "C"},
    "Start": {"href": "https://en.wikipedia.org/wiki/Category:Start-Class_articles",
             "name": "Start", "tag": "Start"},
    "Stub": {"href": "https://en.wikipedia.org/wiki/Category:Stub-Class_articles",
             "name": "Stub",  "tag": "Stub"},
    "FL": {"href": "https://en.wikipedia.org/wiki/Category:FL-Class_articles",
           "tag": "FL"},
    "List": {"tag": "List", "href": "https://en.wikipedia.org/wiki/Category:List-Class_articles"},
    "Disambig": {"tag": "Disambig", "href": "https://en.wikipedia.org/wiki/Category:Disambig-Class_articles"},
    "Book": {"tag": "Book", "href": "https://en.wikipedia.org/wiki/Category:Bookd-Class_articles"},
    "Template": {"tag": "Template", "href": "https://en.wikipedia.org/wiki/Category:Template-Class_articles"},
    "Category": {"tag": "Category", "href": "https://en.wikipedia.org/wiki/Category:Category-Class_articles"},
    "Draft": {"tag": "Draft", "href": "https://en.wikipedia.org/wiki/Category:Draft-Class_articles"},
    "Redirect": {"tag": "Redirect", "href": "https://en.wikipedia.org/wiki/Category:Redirect-Class_articles"}
}

@app.route('/page')
def wiki():
    session = Session()
    title = request.args.get('page-title')
    page = get_page(session, title)
    data = {
        'title': page['title'],
        'wikipedia_url': wikipedia_url(page['title']),
    }
    result = predict_page_edits(featurizer, page['content'], MODEL)
    data['probs'] = reversed([{'prob': round(p * 100), **QA[k]} for k, p in result['prob']])
    data['edits'] = [{'description': Markup(x[1]), 'value': round(x[2] * 100)} 
                     for x in result['top_edits']]
    data['best'] = QA[result['best']]
    return render_template("results.html", **data)

@app.route('/about')
def about():
    return render_template("about.html")

@app.errorhandler(404)
def page_not_found(e):
    return (render_template('404.html'), 404)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)