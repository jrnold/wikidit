"""Get names for all templates in the Wikipedia templates."""
import argparse
import json
import logging
import re

import yaml

from ..mw import Session

logger = logging.getLogger(__name__)


def get_redirects(x):
    pages = list(x['query']['pages'].values())[0]
    if 'redirects' in pages:
        return [r['title'] for r in pages['redirects']]
    else:
        return []


def template_name(x):
    return re.sub("^T(emplate)?:", "", x)


def run(input_file, output_file):
    logger.info(f"Reading from {input_file}")
    with open(input_file, "r") as f:
        backlog = yaml.load(f)
    session = Session()
    backlog_templates = {}
    for section, categories in backlog.items():
        backlog_templates[section] = {}
        for cat, templates in categories.items():
            backlog_templates[section][cat] = {}
            for tmpl in templates:
                result = session.get(action="query", titles=f"Template:{tmpl}", prop="redirects")
                tmpl_name = template_name(tmpl)
                if len(result):
                    redirects = [template_name(x) for x in get_redirects(result)]
                    backlog_templates[section][cat][tmpl_name] = redirects
                else:
                    backlog_templates[section][cat][tmpl_name] = []
    with open(output_file, "w") as f:
        logger.info(f"Writing to {input_file}")
        json.dump(backlog_templates, f)
                    

def main():
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()

