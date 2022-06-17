FROM python:3.9-slim as base

# install psycopg2
RUN apt-get clean \
    && apt-get update
RUN apt-get install -y libpq-dev gcc g++ curl
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install psycopg2
RUN cp /usr/lib/x86_64-linux-gnu/libpq* /opt/venv/lib/python3.9/site-packages/psycopg2/

FROM base as builder

COPY ./utils/packages ./utils/packages
RUN pip install -r ./utils/packages/date_process/requirements.txt
RUN pip install -r ./utils/packages/es_logger/requirements.txt
RUN pip install -r ./utils/packages/logger/requirements.txt
RUN pip install -r ./utils/packages/send_email/requirements.txt
RUN pip install -r ./utils/packages/send_slack/requirements.txt
RUN pip install -r ./utils/packages/sql/requirements.txt

COPY components/requirements.txt components/requirements.txt
RUN pip install -r components/requirements.txt

FROM builder as minimal

RUN addgroup app && adduser --home /app --ingroup app app
COPY --from=builder --chown=app:app /usr/lib/x86_64-linux-gnu/libpq* /usr/lib/
COPY --from=builder --chown=app:app /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
USER app

FROM builder as main
