FROM python:3.9-slim as base

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./general ./general
COPY ./preprocess ./preprocess
COPY ./results_analysis ./results_analysis
COPY main.py main.py
COPY random_forest.py random_forest.py
COPY global_vars.py global_vars.py