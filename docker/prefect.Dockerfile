FROM python:3.12-slim

ARG PREFECT_VERSION=3.8.0
RUN pip install --no-cache-dir "prefect==${PREFECT_VERSION}" "asyncpg>=0.30,<1"

EXPOSE 4200
