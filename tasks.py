"""Task file for invoke."""
import os

from invoke import task

# AWS_S3_BUCKET = "jrnold-data"
# AWS_S3_PREFIX = "wikidit"
# AWS_S3_REGION = "us-west-2"

@task
def build(c):
     """Build Docker image."""
     c.run("docker build -t wikidit .")


@task
def setup(c):
    """Download SpaCy model."""
    c.run("conda env create --force -f environment.yml")

@task
def run_app(c, workers=4, port=8000):
    """Run the web application for production"""
    pwd = os.getcwd()
    # c.run(f"docker run -v {pwd}:/home/jovyan/work -p 0.0.0.0:{port}:{port} wikidit gunicorn -w {workers} app")
    c.run(f"gunicorn -p 0.0.0.0:{port} -w {workers} app:app")

@task
def dev_app(c):
    pwd = os.getcwd()
    # c.run("docker run -v {pwd}:/home/jovyan/work -p 0.0.0.0:{port}:{port} python app.py")
    c.run("python app.py")

@task
def jupyter(c):
    # c.run(f"docker run -v {pwd}:/home/jovyan/work -p 8888:8888 -e JUPYTER_LAB_ENABLE=true wikidit jupyter lab")
    c.run("jupyter lab")

@task
def format(c):
    c.run("black wikidit")
