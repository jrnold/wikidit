from flask import Flask, render_template, request, Markup

import dill
import urllib.parse
import numpy as np
import os.path
import mwapi

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline

import pandas as pd

from wikdit.models import 


app = Flask(__name__)

# Create a single session instance to reuse
wikiapi = mwapi.Session(host=_WIKI_HOST, user_agent=_USER_AGENT)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/page')
def wiki():
    title = request.args.get('page-title')
    page = get_page(wikiapi, title)
    if page is None:
        out = render_template("not_found.html", title=title)
    else:
        wikipage = WikiPage(page['content'])
        preds = generate_predictions(MODEL, wikipage)
        fig = qual_prob_plot(preds['proba'])
        interventions = intervention_table(preds['interventions'])[:5]
        out = render_template('results.html', 
                              title=title,
                              wikipedia_url=wikipedia_url(title, "en", page['revid']),
                              best=Markup(html_article_quality(preds['best'])),
                              avg="{:.1f}".format(preds['avg']),
                              fig=qual_prob_plot(preds['proba']),
                              interventions=interventions)
        return str([page, wikipage, preds])


def html_article_quality(x):
    quality = {'FA': "Featured Article",
               'GA': "Good Article",
               "B": "B-class",
               "C": "C-class",
               "Start": "Start-class",
               "Stub": "Stub-class"}
    return f"<span=\"{x}\">{quality[x]}</span>"


def qual_prob_plot(proba):
    x, y = zip(*proba)
    layout = go.Layout(
        autosize=False,
        width=400,
        height=300,
        xaxis=dict(
            autorange=True,
            ticks='',
        ),
        yaxis=dict(range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1])
    )
    fig = go.Figure(data=[go.Bar(x=x, y=y)], layout=layout)
    div = plotly.offline.plot(fig, show_link=False, output_type="div",
                              include_plotlyjs=False)
    return div


def intervention_table(x):
    x = sorted(x.items(), key=lambda z: -z[1])
    x = [dict(zip(('name', 'value'), z)) for z in x]
    for i in x:
        i['value'] = "{:.1f}".format(i['value'])
    return x


def wikipedia_url(title, lang, revid=None):
    qtitle = urllib.parse.quote(title)
    if revid is None:
        out = f"https://{lang}.wikipedia.org/wiki/{title}"
    else:
        out = f"https://{lang}.wikipedia.org/w/index.php?title={title}&oldid={revid}"
    return out


@app.errorhandler(404)
def page_not_found(e):
    return (render_template('404.html'), 404)
