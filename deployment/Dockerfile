FROM python:3.9-slim as base

COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
