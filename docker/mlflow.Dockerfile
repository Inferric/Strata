FROM python:3.12-slim

ARG MLFLOW_VERSION=3.14.0
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" "psycopg[binary]>=3.2,<4"

EXPOSE 5000
