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
py -3.12 -m venv .venv
source .venv/Scripts/activate

python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e ".[dev]"
python -m pip check
```

On Linux or macOS, create the environment with `python3.12 -m venv .venv`
and activate with `source .venv/bin/activate`.

Installation uses the dependency ranges in `pyproject.toml`. If a validated
environment snapshot is available, use it as a pip constraints file when
reproducing that environment.

## Training configuration

Edit `configs/train.toml` to configure training without changing Python source.
The supplied defaults are:

| TOML setting | Default | Meaning |
|---|---|---|
| `training.epochs` | `10` | Complete passes over the training data |
| `training.batch_size` | `64` | Images per batch |
| `training.learning_rate` | `0.001` | Optimizer learning rate |
| `training.optimizer` | `"adam"` | Supported values: `"adam"`, `"sgd"` |
| `training.loss` | `"cross_entropy"` | Supported loss function |
| `training.seed` | `42` | Seed for initialization and data ordering |
| `validation.validation_split` | `0.1` | Fraction of training data reserved for validation |

Use `--config path/to/settings.toml` for another configuration file. The
`--epochs` argument overrides the configured epoch count. Sample limits are
optional CLI arguments for short checks; omit both to use the complete split.
Tracked runs record the effective settings, including overrides and sample
limits, in MLflow and retain training provenance with the model artifact.

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

For full-dataset local training with ten epochs:

```bash
python -m fashion_mnist_mlops.train --download-data --epochs 10
```

With the supplied split, this uses 54,000 training and 6,000 validation images.
The separate 10,000-image test partition is not the validation partition.
Local training without `--track-mlflow --upload-minio` does not publish a model
for the containerized API. Use the tracked container workflow below to publish.

## Docker configuration

Use Docker Desktop with Linux containers on Windows.

```bash
cp .env.example .env
```

Set `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` in `.env` to your own local
credentials. Keep `.env` out of version control. Leave `MODEL_OBJECT_KEY`
empty until a model has been published.

Exported shell variables take precedence over `.env` for Compose interpolation.
If you previously selected a model in the shell, clear that override before
using the value saved in `.env`:

```bash
unset MODEL_OBJECT_KEY
```

Clear any other stale exported overrides when switching deployments. Do not
print or share the real credentials. Log in to the MinIO console with the
`MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` values from your local `.env`.

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

The required bucket name is exactly `fashion-minst-bucket`. The spelling
`minst` is intentional and follows the assignment requirement.

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

### Retain the dataset cache and local artifacts

After initializing MLflow and the bucket, this Git Bash command runs the
configured training job while retaining downloaded data and local models:

```bash
mkdir -p data artifacts

MSYS_NO_PATHCONV=1 docker compose run --rm -T \
  --volume "$(pwd -W)/data:/app/data" \
  --volume "$(pwd -W)/artifacts:/app/artifacts" \
  training
```

`pwd -W` supplies an absolute Windows path; `MSYS_NO_PATHCONV=1` prevents
Git Bash from rewriting container paths. On Linux/macOS, omit that environment
assignment and use `$(pwd)` in place of `$(pwd -W)`.

Configuration and source files are copied into the training image at build
time. After editing `configs/train.toml`, rebuild with
`docker compose build training` before running that image again. Alternatively,
mount an external configuration and pass its container path with `--config`.

## Select and serve a model

Set the exact key in `.env`:

```text
MODEL_OBJECT_KEY=model/<mlflow-run-id>/model.pt
```

Then start serving:

```bash
docker compose up -d --wait --wait-timeout 240 api ui
```

When selecting a different model, recreate the API so it receives the
updated environment:

```bash
unset MODEL_OBJECT_KEY
docker compose up -d --force-recreate --wait --wait-timeout 240 api
```

Check readiness after startup or a model change:

```bash
curl --noproxy '*' -fsS http://localhost:18000/health
curl --noproxy '*' -fsS http://localhost:18000/ready
```

Expect `status: ready` and the selected MLflow run ID from `/ready`. Adjust
these URLs if you changed the host ports. The API does not choose a model
automatically.

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

### Capture and verify an actual notification

First complete the Docker build, MLflow initialization and bucket provisioning
above. Run the following from the project root in Git Bash. It starts an HTTP
receiver in a background thread inside the training container, then invokes
the training CLI as a separate process with a reachable receiver URL. No
additional host port or Windows firewall rule is required.

This example uses one epoch and limited samples to check delivery. It permits
the initial dataset download, retains the cache, and writes each captured
payload to a unique file under `artifacts/webhooks/`.

```bash
mkdir -p data artifacts

MSYS_NO_PATHCONV=1 docker compose run --rm -T \
  --volume "$(pwd -W)/data:/app/data" \
  --volume "$(pwd -W)/artifacts:/app/artifacts" \
  --volume "$(pwd -W)/scripts:/app/scripts:ro" \
  --entrypoint python \
  training - <<'PYTHON'
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from uuid import uuid4

from fashion_mnist_mlops.webhook_receiver import create_webhook_receiver

receiver = create_webhook_receiver(port=0)
worker = threading.Thread(target=receiver.serve_forever, daemon=True)
worker.start()
environment = os.environ.copy()
environment["WEBHOOK_URL"] = (
    f"http://127.0.0.1:{receiver.server_address[1]}/webhook"
)
capture = Path("/app/artifacts/webhooks") / f"{uuid4().hex}.json"

try:
    subprocess.run(
        [
            sys.executable, "-m", "fashion_mnist_mlops.train",
            "--project-root", "/app",
            "--epochs", "1",
            "--max-train-samples", "1024",
            "--max-validation-samples", "256",
            "--download-data", "--track-mlflow", "--upload-minio",
        ],
        env=environment,
        check=True,
    )
    if len(receiver.captured_payloads) != 1:
        raise RuntimeError("Expected exactly one accepted webhook.")
    capture.parent.mkdir(parents=True, exist_ok=True)
    with capture.open("x", encoding="utf-8") as output:
        json.dump(receiver.captured_payloads[0], output, indent=2)
        output.write("\n")
    subprocess.run(
        [sys.executable, "scripts/verify_webhook.py", "--payload", str(capture)],
        check=True,
    )
    print(f"Captured payload: {capture}")
    print(f"Host file: artifacts/webhooks/{capture.name}")
finally:
    receiver.shutdown()
    receiver.server_close()
    worker.join(timeout=5)
PYTHON
```

Expect `Webhook delivered: HTTP 204` and a verification result confirming a
FINISHED MLflow run and matching MinIO model bytes. This tests real local HTTP
delivery, not connectivity to an external webhook provider. It creates a new
model version but does not change the API's selected model.

Training, required tracking and upload failures must not emit a success
notification. The corresponding automated checks are:

```bash
python -m pytest -q -rs \
  tests/unit/test_webhook_failure_boundaries.py \
  tests/unit/test_webhook_runner.py \
  tests/unit/test_minio_upload_runner.py
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

### Run a verification helper inside Docker

The training image includes `provision_minio_bucket.py`; the other helper
scripts require a read-only scripts mount. After publishing a model, verify
its remote reference and CPU restoration using this Git Bash example:

```bash
read -r -p "Paste the model object key (model/<run-id>/model.pt): " model_key

MSYS_NO_PATHCONV=1 docker compose run --rm -T \
  --volume "$(pwd -W)/scripts:/app/scripts:ro" \
  --entrypoint python \
  training \
  scripts/verify_model_inference.py \
  --destination /tmp/verified-model.pt \
  --object-key "$model_key"
```

Paste the actual object key when prompted, then press Enter. Leave `model_key`
as the variable name in the command. The helper uses a synthetic image to
check restoration and inference; its prediction is not an accuracy measure.

## Tests

Use the installed project environment for consolidated validation:

```bash
python scripts/validate.py
```

This checks package location, syntax, dependencies, and both test suites.
Integration tests must complete without skips. Reports are written beneath
`artifacts/validation/`. These checks do not start Docker services or replace
live deployment verification.

A GitHub ZIP download has no `.git` directory. Use the command above for an
extracted ZIP. Do not initialize Git merely to run the tests.

For a Git checkout, after committing the reviewed project, also check
repository cleanliness:

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

For a fresh installation, explicitly download both dataset partitions before
running the integration suite or consolidated validation:

```bash
python - <<'PYTHON'
from torchvision.datasets import FashionMNIST

for training_partition in (True, False):
    dataset = FashionMNIST(
        root="data", train=training_partition, download=True
    )
    print("Training" if training_partition else "Test", len(dataset))
PYTHON
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
