"""Flask application."""
import urllib.parse
import os.path

from flask import Flask, render_template, request, Markup

from wikidit.mw import get_page
from wikidit.models import Featurizer, predict_page_edits, load_model
from wikidit.preprocessing import WP10_LABELS

app = Flask(__name__)


# Load instances that should only be loaded once
MODEL = load_model()
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
    title = request.args.get('page-title')
    # if an empty title, return the original index
    if title is None or title.strip() == '':
        return render_template('index.html')
    page = get_page(title)
    if page is None:
        return render_template("not_found.html", title=title)
    data = {
        'title': page['title'],
        'wikipedia_url': wikipedia_url(page['title']),
    }
    result = predict_page_edits(page['content'], featurizer, MODEL)
    data['probs'] = reversed([{'prob': round(p * 100), **QA[k]} for k, p in result['prob']])
    data['edits'] = [{'description': Markup(x[1]), 'value': round(x[2] * 100)}
                     for x in result['top_edits'] if x[2] > 0.005]
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
