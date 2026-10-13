# Pipeline image: runs the pipeline CLI, tests and scripts without Airflow.
#   docker compose run --rm pipeline python -m pipeline.cli run-all
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIPELINE_HOME=/opt/project \
    PYTHONPATH=/opt/project/src

WORKDIR /opt/project

# Dependencies first (layer cache): only re-installed when requirements change.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Code is also bind-mounted by docker-compose; copying it makes the image usable on its own.
COPY . .

CMD ["python", "-m", "pipeline.cli", "--help"]
