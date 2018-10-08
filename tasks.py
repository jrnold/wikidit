from invoke import task

AWS_S3_BUCKET = "jrnold-data"
AWS_S3_PREFIX = "wikidit"
AWS_S3_REGION = "us-west-2"

@task
def push_aws():
    # aws s3 sync --region us-west-2 models/ s3://jrnold-data/wikidit/models/"
    # aws s3 cp --region us-west-2 data/enwiki.labeling*.ndjson.gz s3://jrnold-data/wikidit/data/
    pass