# Airflow image with an ISOLATED virtualenv for the pipeline.
# Airflow 2.10.5 pins its own (older) pandas/numpy; the pipeline needs the exact
# versions in requirements.txt to reproduce the baseline. Installing them into a
# separate venv (/opt/pipeline-venv) keeps both sides correct. DAG tasks call
# /opt/pipeline-venv/bin/python -m pipeline.cli ...
FROM apache/airflow:2.10.5-python3.12

USER root
RUN python -m venv /opt/pipeline-venv && chown -R airflow:root /opt/pipeline-venv

USER airflow
COPY requirements.txt requirements-dev.txt /tmp/
# PIP_USER is set to true in the base image; it must be off inside a venv.
RUN PIP_USER=false /opt/pipeline-venv/bin/pip install --no-cache-dir -r /tmp/requirements-dev.txt

# pytest in Airflow's own environment, for tests/integration/test_dag_integrity.py (DagBag import test)
RUN pip install --no-cache-dir pytest==8.4.2
