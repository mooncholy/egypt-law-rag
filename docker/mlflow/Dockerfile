# The MLflow tracking server (Phase 8): the client's MLflow version, plus the
# Postgres driver for the backend store and boto3 for proxied S3 artifacts.
# The upstream ghcr.io/mlflow/mlflow image carries neither.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

RUN pip install --no-cache-dir \
        mlflow==3.16.1 \
        psycopg2-binary==2.9.13 \
        boto3==1.43.75 \
    && useradd --create-home --uid 1000 mlflow

USER mlflow
WORKDIR /home/mlflow
EXPOSE 5000
