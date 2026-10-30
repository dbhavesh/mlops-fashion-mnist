"""FastAPI health, readiness, and image prediction endpoints.

Run: python -m uvicorn fashion_mnist_mlops.api:app
"""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """A response means the API process can serve an HTTP request."""

    status: Literal["ok"] = "ok"


health_router = APIRouter(tags=["health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API process liveness",
    description=(
        "Reports only that the FastAPI process is responding. "
        "It does not check MLflow, MinIO, a artifact, or model readiness."
    ),
)
def get_health() -> HealthResponse:
    """Respond without loading a model, accessing storage, or making network calls."""
    return HealthResponse()

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Literal

from fashion_mnist_mlops.api_errors import (
    APIErrorResponse,
    APIServiceError,
    MODEL_UNAVAILABLE,
    READINESS_FAILED,
)
from fashion_mnist_mlops.model_loading import APIModelLoadError


class ReadinessResponse(BaseModel):
    status: Literal["ready"] = "ready"
    model_run_id: str = Field(min_length=1)


ready_router = APIRouter(tags=["readiness"])


def _resolve_readiness(request: Request) -> ReadinessResponse:
    """Load/cache the exact configured model only when readiness is requested.

    A cached version counts as ready even if remote services later go offline;
    switching MODEL_OBJECT_KEY requires successfully loading the new selection.
    """
    try:
        restored = request.app.state.model_loader.load()
    except (APIModelLoadError, ValueError) as exc:
        raise APIServiceError(503, "MODEL_UNAVAILABLE", MODEL_UNAVAILABLE) from exc
    try:
        return ReadinessResponse(model_run_id=restored.run_id)
    except Exception as exc:
        raise APIServiceError(500, "READINESS_FAILED", READINESS_FAILED) from exc


@ready_router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        503: {"model": APIErrorResponse, "description": "Selected model unavailable"},
        500: {"model": APIErrorResponse, "description": "Readiness evaluation failed"},
    },
    summary="Check selected model readiness",
    description=(
        "Unlike /health, explicitly resolves the selected run and loads its "
        "integrity-verified artifact on demand. A cached usable model "
        "satisfies readiness without re-contacting remote services."
    ),
)
async def get_readiness(request: Request) -> ReadinessResponse:
    """Avoid blocking the event loop during remote model retrieval."""
    try:
        return await run_in_threadpool(_resolve_readiness, request)
    except APIServiceError:
        raise
    except Exception as exc:
        raise APIServiceError(500, "READINESS_FAILED", READINESS_FAILED) from exc

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile

from fashion_mnist_mlops.api_errors import (
    APIErrorResponse,
    APIServiceError,
    MODEL_UNAVAILABLE,
    PREDICTION_FAILED,
)
from fashion_mnist_mlops.model_loading import APIModelLoadError
from fashion_mnist_mlops.prediction import (
    MAX_UPLOAD_BYTES,
    InvalidUploadedImage,
    UnsupportedUploadedImageType,
    canonical_class_name,
    decode_uploaded_image,
    predict_uploaded_image,
    validate_image_upload_metadata,
)

predict_router = APIRouter(tags=["prediction"])


class PredictionResponse(BaseModel):
    """Predicted class, canonical readable label, and exact serving run."""

    class_index: int = Field(ge=0, le=9, description="Fashion-MNIST class ID (0–9).")
    class_name: str = Field(min_length=1, description="Canonical name of the predicted class.")
    model_run_id: str = Field(min_length=1, description="MLflow run ID of the served model.")


def _predict_once(request: Request, content: bytes, expected_format: str) -> PredictionResponse:
    """Validate image before model retrieval; execute all blocking operations off-loop."""
    image = decode_uploaded_image(content, expected_format=expected_format)
    try:
        try:
            restored = request.app.state.model_loader.load()
        except (APIModelLoadError, ValueError) as exc:
            raise APIServiceError(503, "MODEL_UNAVAILABLE", MODEL_UNAVAILABLE) from exc
        try:
            result = predict_uploaded_image(restored, image)
            return PredictionResponse(
                class_index=result.class_index,
                class_name=canonical_class_name(result.class_index),
                model_run_id=restored.run_id,
            )
        except Exception as exc:
            raise APIServiceError(500, "PREDICTION_FAILED", PREDICTION_FAILED) from exc
    finally:
        image.close()


@predict_router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": APIErrorResponse, "description": "Invalid or unreadable upload"},
        413: {"model": APIErrorResponse, "description": "Upload exceeds the 5 MiB limit"},
        415: {"model": APIErrorResponse, "description": "Unsupported image format"},
        500: {"model": APIErrorResponse, "description": "Inference failed"},
        503: {"model": APIErrorResponse, "description": "Selected model unavailable"},
    },
    summary="Classify one PNG or JPEG Fashion-MNIST image",
    description=(
        "Accepts exactly one multipart/form-data PNG or JPEG image in `file` "
        "(maximum 5 MiB and 16 MiPixels). File extension, MIME and real format "
        "must agree. Returns canonical class name, class index and run ID. "
        "The `/ready` endpoint independently verifies model readiness."
    ),
)
async def post_predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """Return typed, sanitized client/server errors with stable category codes."""
    form = await request.form()
    entries = form.multi_items()
    if (len(entries) != 1 or entries[0][0] != "file"
            or not isinstance(entries[0][1], StarletteUploadFile)):
        raise APIServiceError(400, "INVALID_UPLOAD", "Exactly one image file is required.")

    try:
        expected_format = validate_image_upload_metadata(
            filename=file.filename, content_type=file.content_type
        )
    except UnsupportedUploadedImageType as exc:
        raise APIServiceError(415, "UNSUPPORTED_MEDIA_TYPE", str(exc)) from exc

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise APIServiceError(400, "INVALID_IMAGE", "Image file must not be empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise APIServiceError(413, "PAYLOAD_TOO_LARGE", "Image upload is too large.")

    try:
        return await run_in_threadpool(_predict_once, request, payload, expected_format)
    except APIServiceError:
        raise
    except UnsupportedUploadedImageType as exc:
        raise APIServiceError(415, "UNSUPPORTED_MEDIA_TYPE", str(exc)) from exc
    except InvalidUploadedImage as exc:
        raise APIServiceError(400, "INVALID_IMAGE", str(exc)) from exc
    except Exception as exc:
        raise APIServiceError(500, "PREDICTION_FAILED", PREDICTION_FAILED) from exc

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(ready_router)
api_router.include_router(predict_router)

from contextlib import asynccontextmanager

from fastapi import FastAPI

from fashion_mnist_mlops.model_loading import APISettings
from fashion_mnist_mlops.api_errors import APIServiceError, api_service_error_handler
from fashion_mnist_mlops.model_loading import APIModelLoader


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Release verified artifact cache on ASGI application shutdown."""
    try:
        yield
    finally:
        application.state.model_loader.clear()


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Build an independent app without network I/O or artifact loading."""
    if settings is None:
        settings = APISettings()
    if not isinstance(settings, APISettings):
        raise TypeError("settings must be an APISettings instance.")

    application = FastAPI(
        title=settings.title,
        version=settings.version,
        description="Fashion-MNIST image inference API with version-pinned model selection.",
        lifespan=_lifespan,
    )
    application.add_exception_handler(APIServiceError, api_service_error_handler)
    application.state.model_loader = APIModelLoader()
    application.include_router(api_router)
    return application


app = create_app()
