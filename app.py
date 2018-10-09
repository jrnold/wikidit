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

from wikidit.mw import Session, get_page
from wikidit.models import Featurizer, predict_page_edits
from wikidit.preprocessing import WP10_LABELS

app = Flask(__name__)

# Load instances that should only be loaded once
MODEL_FILE = os.path.join("models", "model.pkl")
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


@app.route('/page')
def wiki():
    title = request.args.get('page-title')
    session = Session()
    page = get_page(session, title)
    result = predict_page_edits(featurizer, page['content'], MODEL)
    class_prob = round(result["predicted_class_prob"] * 100)
    edits = [{'description': x[0], 'value': round(x[1] * 100)} for x in result['top_edits']]
    return render_template("results.html", 
                    quality = result["predict"],
                    next_level = get_next_quality_cat(result["predict"]),
                    class_prob = class_prob,
                    edits = edits, 
                    title = title,
                    wikipedia_url = wikipedia_url(title))

DESCRIPTIONS = {
    "ref": "Add a reference footnote.",
    "words": "Add a sentence (15 words).",
    "heading": "Organize the article with a new heading",
    "sub_heading": "Organize the article with a new sub_heading"
}

QA = {
    "FA": {"link": "https://en.wikipedia.org/wiki/Wikipedia:Featured_articles",
           "name": "Featured article",
           "color": "ffff66"},
    "GA": {"href": "https://en.wikipedia.org/wiki/Wikipedia:Good_articles",
           "name": "Good article",
           "color": "66ff66"},
    "B": {"href": "https://en.wikipedia.org/wiki/Category:B-Class_articles",
          "name": "B-class article", "color": "b2ff66"},
    "C": {"href": "https://en.wikipedia.org/wiki/Category:C-Class_articles",
          "name": "C-class article", "color": "ffff66"},
    "Start": {"href": "https://en.wikipedia.org/wiki/Category:Start-Class_articles",
             "name": "Start", "color": "ffaa66"},
    "Stub": {"href": "https://en.wikipedia.org/wiki/Category:Stub-Class_articles",
             "name": "Stub", "color": "ffa4a4"}
}


@app.errorhandler(404)
def page_not_found(e):
    return (render_template('404.html'), 404)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)