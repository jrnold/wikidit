from invoke import task

AWS_S3_BUCKET = "jrnold-data"
AWS_S3_PREFIX = "wikidit"
AWS_S3_REGION = "us-west-2"


@task
def download_spacy(c):
    c.run("python -m spacy download en_core_web_md")

    
@task
def gunicorn(c):
    c.run("gunicorn -w 4 -b 127.0.0.1:8000 run:app")
    
