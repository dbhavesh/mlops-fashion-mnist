FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /mlflow

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

COPY docker/mlflow.requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt \
    && mkdir -p /mlflow/mlartifacts \
    && chown -R app:app /mlflow

USER app

EXPOSE 5000

CMD ["python", "-m", "mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/tracking.db", "--artifacts-destination", "/mlflow/mlartifacts", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--allowed-hosts", "mlflow:5000,localhost:*,127.0.0.1:*"]
