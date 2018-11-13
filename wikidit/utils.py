import itertools


def split_seq(iterable, size):
    # From https://stackoverflow.com/a/312467/227406
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))
