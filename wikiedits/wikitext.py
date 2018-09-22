import re
import mwparserfromhell


def wiki2plaintext(parser, content):
    """Convert Wiki markup to plain text."""
    parser = mwparserfromhell.parser()
    wikicode = parser.parse(content)

    re_image_wl = re.compile(
        '^(?:File|Image|Media):', flags=re.IGNORECASE | re.UNICODE)
    bad_template_names = {
        'reflist', 'notelist', 'notelist-ua', 'notelist-lr', 'notelist-ur', 'notelist-lg'}
    bad_tags = {'ref', 'table'}

    def is_bad_wikilink(obj):
        return bool(re_image_wl.match(obj.title))

    def is_bad_tag(obj):
        return obj.tag in bad_tags

    def is_bad_template(obj):
        return obj.name.lower() in bad_template_names

    def is_bad_section(obj):
        return obj.

    texts = []
    # strip out references, tables, and file/image links
    # then concatenate the stripped text of each section
    for i, section in enumerate(wikicode.get_sections(flat=True, include_lead=True, include_headings=True)):
        for obj in section.ifilter_wikilinks(matches=is_bad_wikilink, recursive=True):
            try:
                section.remove(obj)
            except Exception:
                continue
        for obj in section.ifilter_templates(matches=is_bad_template, recursive=True):
            try:
                section.remove(obj)
            except Exception:
                continue
        for obj in section.ifilter_tags(matches=is_bad_tag, recursive=True):
            try:
                section.remove(obj)
            except Exception:
                continue
        texts.append(section.strip_code().strip())

    return '\n\n'.join(texts)
