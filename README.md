# Install

I try to have everything runnable through Docker.

## Development

To build the docker image
```
docker-compose build
docker-compose up
```

## Description

The file [enwiki.labeling_revisions.nettrom_30k.json](https://github.com/wikimedia/articlequality/blob/master/datasets/enwiki.labeling_revisions.nettrom_30k.json)
 is a sample of 30,000+ revisions, equally balanced between
the Stub, Start, C, B, and A categories. This is used for training Mediawiki's prediction model in the 
[articlequality](https://github.com/wikimedia/articlequality) package.


