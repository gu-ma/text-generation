# CPU image for the API
ARG BASE_IMAGE=python:3.8
FROM $BASE_IMAGE

RUN pip install -U pip
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN python -m spacy download en_core_web_sm