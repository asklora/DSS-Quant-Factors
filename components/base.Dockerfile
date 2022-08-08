ARG BASE
FROM $BASE as installer

COPY components/data_preparation/requirements.txt components/data_preparation/requirements.txt
COPY components/model_training/requirements.txt components/model_training/requirements.txt
COPY components/model_evaluation/requirements.txt components/model_evaluation/requirements.txt

RUN pip install -r ./components/data_preparation/requirements.txt
RUN pip install -r ./components/model_training/requirements.txt
RUN pip install -r ./components/model_evaluation/requirements.txt

FROM installer as main

COPY components/data_preparation components/data_preparation
COPY components/model_training components/model_training
COPY components/model_evaluation components/model_evaluation

ENV DEBUG=False
ENV DB_USERNAME=quant_factor
ENV DB_PASSWORD=quant_factor
ENV PYTHONPATH=$PYTHONPATH:.