FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

COPY docker/ui.requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Deliberately copy only the package initializer and UI modules. The UI image
# contains no training, model, preprocessing, storage, tracking, or API code.
RUN mkdir -p /app/src/fashion_mnist_mlops
COPY --chown=app:app src/fashion_mnist_mlops/__init__.py /app/src/fashion_mnist_mlops/__init__.py
COPY --chown=app:app src/fashion_mnist_mlops/ui.py src/fashion_mnist_mlops/ui_client.py src/fashion_mnist_mlops/ui_config.py /app/src/fashion_mnist_mlops/

USER app

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/fashion_mnist_mlops/ui.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
