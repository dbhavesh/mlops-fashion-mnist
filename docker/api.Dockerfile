FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

COPY docker/api.requirements.txt /tmp/requirements.txt

# Inference is CPU-only. Install PyTorch from its CPU wheel index explicitly so
# the image does not pull the multi-gigabyte CUDA/NVIDIA runtime stack.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Serving reuses accepted project modules. Training is never started by this image.
COPY --chown=app:app src/ /app/src/

USER app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "fashion_mnist_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
