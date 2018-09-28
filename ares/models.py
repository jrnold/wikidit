import re

import mwparserfromhell as mwparser
from .mw import match_template

def featurize(content: str) -> dict:
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
    text = mwparser.parse(content)
    PAT_WORD = r"\b\w+\b"
    PAT_PARA = r"\n{2,}"

    revision = {}

    # Content characters are visible characters. Operationalized as characters after
    # mwparserfromhell calls the ``strip_code()`` method.

    # contentwords = [node.strip_code() for node in text.ifilter(content, forcetype=(Tag, Text, HTMLEntity))
    plaintext = text.strip_code()

    ## Real Content

    # Sections
    word_count = len(re.findall(PAT_WORD, plaintext))
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

    # number of who templates
    revision['who_templates'] = sum([match_template(x, 'who$') for x in templates])

    # main templates
    revision['main_templates'] = sum([match_template(x, "main$") for x in templates])

    # citation templates
    revision['cite_templates'] = sum([match_template(x, "cite") for x in templates])

    # infoboxes
    revision['infoboxes'] = sum([match_template(x, "infobox") for x in templates])

    # number of citation needed templates
    revision['citation_needed'] = sum(match_template(x, "citation_needed|cn|fact") for x in templates)

    # other templates
    special_templates = sum(revision[k] for k in 
                            ("infoboxes", "cite_templates", "citation_needed",
                             "main_templates", "who_templates"))

    revision['other_templates'] = len(templates) - special_templates
    
    # number of ref tags
    revision['ref'] = len([x for x in text.filter_tags() if x.tag == "ref"])

    # number of smartlists (e.g. wikitables)
    revision['smartlists'] = len([x for x in text.nodes
                                  if isinstance(x, mwparser.smart_list.SmartList)])

    # is there a map
    revision['coordinates'] = '#coordinates' in str(text).lower()

    # Add plaintext for more features
    revision['text'] = plaintext

    return revision
