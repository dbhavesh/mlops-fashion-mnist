FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

COPY docker/training.requirements.txt /tmp/requirements.txt

# Training is CPU-first for this assignment. Explicit CPU wheels avoid pulling
# the CUDA/NVIDIA runtime stack into the one-shot job image.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

RUN mkdir -p /app/data /app/artifacts /app/scripts \
    && chown -R app:app /app

COPY --chown=app:app src/ /app/src/
COPY --chown=app:app configs/ /app/configs/
COPY --chown=app:app pyproject.toml /app/pyproject.toml
COPY --chown=app:app scripts/provision_minio_bucket.py /app/scripts/provision_minio_bucket.py

USER app

CMD ["python", "-m", "fashion_mnist_mlops.train", "--help"]
