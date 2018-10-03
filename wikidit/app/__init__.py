from flask import Flask, render_template, request, Markup

import dill
import urllib.parse
import numpy as np
import os.path
import mwapi

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline


_USER_AGENT = "wikidit <jeffrey.arnold@gmail.com>"
_WIKI_HOST = "https://en.wikipedia.org"


# This needs to be replaced by config
MODEL_FILENAME = os.path.join(".", "models", "model.pkl")
with open(MODEL_FILENAME, "rb") as f:
    return dill.load(f)

app = Flask(__name__)


class WikiPage:
    """Class to represent a Wikipedia page for use in models.

    This handles transforming the text of the page into features,
    and

    """
    count_variables = (("words", 50, "Add one paragraph (50 words)"),
             ("headings", 1, "Split content with a  heading"),
             ("sub_headings", 1, "Split content with a subheading"),
             ("images", 1, "Add an image"),
             ("categories", 1, "Add a link to a category"),
             ("wikilinks", 1, "Add a link to another wiki page"),
             ("cite_templates", 1, "Add a citation"),
             ("who_templates", -1, "Remove a Who? template"),
             ("smartlists", 1, "Add a table"),
             ("ref", 1, "Add a reference"))

    binary_variables = (("coordinates", 1, "Add coordinates"),
                        ("infobox", 1, "Add an infobox"))

    def __init__(self, content: str):
        self.data = pd.DataFrame.from_records([featurize(content)])

    def add_count(self, variable, value):
        """Add value to count number to ensure that it is not greater than """
        df = self.data.copy()
        df[variable] = df[variable] + value
        df.loc[df[variable] < 0] = 0
        return df

    def set_value(self, variable, value):
        """Set all values of a column to the same value"""
        df = self.data.copy()
        df[variable] = value
        return df

    def edits(self):
        for x in self.count_variables:
            yield (x[2], self.add_count(*x[:2]))
        for x in self.binary_variables:
            yield (x[2], self.set_value(*x[:2]))


# Create a single session instance to reuse
wikiapi = mwapi.Session(host=_WIKI_HOST,
                 user_agent=_USER_AGENT)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/page')
def wiki():
    title = request.args.get('page-title')
    page = get_current_page(wikiapi, title)
    if page is None:
        out = render_template("not_found.html", title=title)
    else:
        preds = generate_predictions(MODEL, WikiPage(page))
        fig = qual_prob_plot(preds['proba'])
        interventions = intervention_table(preds['interventions'])[:5]
        out = render_template('results.html', title=title,
                              wikipedia_url=wikipedia_url(title, "en",
                                                          page['revid']),
                              best=Markup(html_article_quality(preds['best'])),
                              avg="{:.1f}".format(preds['avg']),
                              fig=qual_prob_plot(preds['proba']),
                              interventions=interventions)
    return out


def get_current_page(session, title):
    if title is None or title == '':
        return None
    r = session.get(action="query", titles=title, prop="revisions",
                    rvprop="ids|content|timestamp", rvslots="main")
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
