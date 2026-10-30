# Fashion-MNIST MLOps

Train a LeNet-5-style classifier, track experiments with MLflow, publish
verified model artifacts to MinIO, and serve predictions through FastAPI
and Streamlit.

The model has 44,426 parameters and predicts the ten Fashion-MNIST classes.
Training and inference share grayscale conversion, resizing to 28 × 28,
and normalization using mean `0.2860406` and standard deviation `0.3530242`.

## Project layout

| Location | Purpose |
|---|---|
| `configs/train.toml` | Training configuration |
| `src/fashion_mnist_mlops/` | Training, storage, tracking, inference, API, and UI modules |
| `scripts/` | Provisioning and verification utilities |
| `docker/` | Service Dockerfiles and dependency lists |
| `compose.yaml` | Service configuration, health checks, network, and persistent volumes |
| `tests/unit/` | Module and configuration tests |
| `tests/integration/` | Real-data training and UI/API HTTP tests |
| `data/` | Local dataset cache, generated or copied separately |
| `artifacts/` | Local model files, logs, and verification evidence |

## Python setup

Use Python 3.12. The following commands are for Git Bash on Windows and
must be run from this repository's root.

```bash
python -m venv .venv
source .venv/Scripts/activate

python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e ".[dev]"
python -m pip check
```

On Linux or macOS, activate with `source .venv/bin/activate`.

Installation uses the dependency ranges in `pyproject.toml`. If a validated
environment snapshot is available, use it as a pip constraints file when
reproducing that environment.

## Local training

Permit the initial Fashion-MNIST download explicitly:

```bash
python -m fashion_mnist_mlops.train \
  --download-data \
  --epochs 1 \
  --max-train-samples 1024 \
  --max-validation-samples 256
```

Use `--offline` for subsequent runs with cached data. Offline operation is
the default.

The standard dataset contains 60,000 training images and 10,000 test images.
The configured validation split separates the training partition into
54,000 training images and 6,000 validation images.

Without an explicit output path, each run writes its model beneath
`artifacts/models/`. Existing model files are not overwritten. To select
a new output path, use `--artifact-path`.

```bash
python -m fashion_mnist_mlops.train --help
```

A bounded run checks the workflow. Assess model quality using an appropriate
training budget and a separate evaluation.

## Docker configuration

Use Docker Desktop with Linux containers on Windows.

```bash
cp .env.example .env
```

Set `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` in `.env` to your own local
credentials. Keep `.env` out of version control. Leave `MODEL_OBJECT_KEY`
empty until a model has been published.

Compose reads `.env` automatically. Direct Python commands require settings
in the process environment; they do not automatically load this file.

| Service | Host address |
|---|---|
| MLflow | http://localhost:15000 |
| MinIO API | http://localhost:19000 |
| MinIO console | http://localhost:19001 |
| FastAPI | http://localhost:18000 |
| FastAPI documentation | http://localhost:18000/docs |
| Streamlit | http://localhost:18501 |

These host ports are configurable in `.env`. Containers use service names
and internal ports, such as `http://mlflow:5000`, `minio:9000`, and
`http://api:8000`.

The Compose project is `mlops-fashion-mnist`. Its persistent volumes are
`mlops-fashion-mnist-minio-data` and `mlops-fashion-mnist-mlflow-data`.

## Build and initialize services

```bash
docker compose --profile training build
docker compose up -d --wait --wait-timeout 360 minio mlflow
```

Initialize the configured MLflow experiment:

```bash
docker compose run --rm --no-deps -T training python - <<'PYTHON'
from mlflow.tracking import MlflowClient
from fashion_mnist_mlops.config import load_infrastructure_config

settings = load_infrastructure_config()
client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
name = settings.mlflow.experiment_name
experiment = client.get_experiment_by_name(name)

if experiment is None:
    experiment_id = client.create_experiment(name)
else:
    experiment_id = experiment.experiment_id
    if experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment_id)

print(f"Active experiment: {name} ({experiment_id})")
PYTHON
```

Provision the required bucket:

```bash
docker compose run --rm --no-deps training \
  python scripts/provision_minio_bucket.py
```

The required bucket name is exactly `fashion-minst-bucket`.

## Tracked container training

```bash
docker compose run --rm training
```

The training service provisions the bucket, permits dataset download,
trains using the configuration, records the run in MLflow, and publishes
the model to MinIO. Training runs only when explicitly invoked.

Record the printed MLflow run ID and MinIO object key. The key follows:

```text
model/<mlflow-run-id>/model.pt
```

Open MLflow at http://localhost:15000, select the configured experiment,
and open a run to view its parameters, metrics, and artifacts. Select two
runs and choose **Compare** to compare their recorded results.

The default training container has no dataset or local-artifact bind mounts.
Its temporary local files disappear when the container is removed; the
published MinIO model and MLflow records persist. Mount `data/` and
`artifacts/` explicitly when retaining a local cache and model files is
required, such as for local-artifact run comparison.

## Select and serve a model

Set the exact key in `.env`:

```text
MODEL_OBJECT_KEY=model/<mlflow-run-id>/model.pt
```

Then start serving:

```bash
docker compose up -d --wait api ui
```

When selecting a different model, recreate the API so it receives the
updated environment:

```bash
docker compose up -d --force-recreate --wait api
```

The API does not choose a model automatically.

- `GET /health` checks that the API process responds.
- `GET /ready` retrieves and verifies the selected model on demand.
- `POST /predict` accepts one PNG or JPEG in the multipart field `file`.

The prediction response contains `class_index`, `class_name`, and
`model_run_id`. Uploads are limited to 5 MiB and 16 MiPixels; the filename,
declared media type, and decoded image format must agree.

Open Streamlit, upload a supported image, and select **Predict**. Streamlit
sends the image to FastAPI; model loading and inference run in the API.

## Run the API and UI directly

With the virtual environment active and the required settings exported:

```bash
python -m uvicorn fashion_mnist_mlops.api:app --host 127.0.0.1 --port 18000
```

In another terminal:

```bash
export FASTAPI_BASE_URL=http://127.0.0.1:18000
python -m streamlit run src/fashion_mnist_mlops/ui.py --server.port 18501
```

Stop containerized serving first if it is already using these host ports.

## Webhook notifications

Set `WEBHOOK_URL` to a destination reachable from the training process.
Successful tracked training with MinIO publication can then send a
`training.completed` notification. An empty URL disables delivery.

`WEBHOOK_TIMEOUT_SECONDS` controls the delivery timeout. For a local
receiver and captured-payload verification, inspect:

```bash
python -m scripts.webhook_receiver --help
python -m scripts.verify_webhook --help
```

Inside a container, `127.0.0.1` refers to that container. Configure the
receiver address according to where it actually runs.

A successful notification has this structure; the run ID, object key, and
accuracy below are illustrative:

```json
{
  "schema_version": 1,
  "event": "training.completed",
  "status": "success",
  "mlflow_run_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "minio_bucket": "fashion-minst-bucket",
  "minio_object_key": "model/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/model.pt",
  "validation_accuracy": 0.85
}
```

## Verification utilities

Run these from the repository root with the appropriate environment
settings:

```bash
python -m scripts.inspect_model_artifact --help
python -m scripts.download_model_artifact --help
python -m scripts.verify_model_inference --help
python -m scripts.verify_model_versions --help
python -m scripts.compare_runs --help
```

Artifact inspection checks a local model without uploading it. Downloads
verify the selected storage reference. Version verification checks two
completed runs and restores both remote models.

Run comparison checks parameters, metric histories, and provenance against
retained project-local model files. Those local files must be available
under the project root supplied to the utility.

## Tests

Use the installed project environment for consolidated validation:

```bash
python scripts/validate.py
```

This checks package location, syntax, dependencies, and both test suites.
Integration tests must complete without skips. Reports are written beneath
`artifacts/validation/`. These checks do not start Docker services or replace
live deployment verification.

After committing the reviewed project, also check repository cleanliness:

```bash
python scripts/validate.py --require-clean
```

The additional checks require this directory to be the Git repository root,
check whitespace, and reject staged, unstaged, or untracked changes. Child
processes do not inherit service settings, Git overrides, or pytest overrides
from the calling shell. The calling shell's environment is unchanged.

For individual checks:

```bash
python -m pip check
python -m pytest -q -rs tests/unit
python -m pytest -q -rs tests/integration
```

The training integration test uses a small subset of the real cached
dataset and invokes the CLI from outside the repository. It skips explicitly
if the dataset is unavailable; that skip is not successful training
integration verification.

The HTTP integration tests use a temporary loopback server and a simulated
model boundary. Live container checks separately exercise MLflow, MinIO,
artifact restoration, serving, webhook delivery, and persistence.

## Shutdown and troubleshooting

```bash
docker compose ps -a
docker compose logs --no-color --tail=200 minio mlflow api ui
docker compose stop
docker compose start
```

To remove service containers and the network while retaining named volumes:

```bash
docker compose down
```

An API can pass `/health` while failing `/ready`. Check the explicit model
key, MLflow run status, MinIO object, and credentials when readiness fails.

After a dependency or source change, rebuild the affected image and recreate
its service. Keep captured validation evidence under `artifacts/`.
