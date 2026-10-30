"""Stable, sanitized API errors for inference and model-readiness endpoints.

Expected API failures carry machine-readable codes. Internal exception text and
artifact, host, credential or traceback details must never enter HTTP bodies.
FastAPI's own request-schema errors (422) keep their standard validation shape.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class APIErrorResponse(BaseModel):
    """Common body for explicitly handled prediction/readiness errors."""

    detail: str = Field(description="Safe, human-readable explanation.")
    code: str = Field(description="Stable API error category.")


class APIServiceError(Exception):
    """Known safe HTTP failure; detail must not contain lower-layer exceptions."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        self.status_code = status_code
        self.code = code
        self.detail = detail
        super().__init__(detail)


async def api_service_error_handler(request: Request, exc: APIServiceError) -> JSONResponse:
    """Render only fixed messages; never serialize ``__cause__`` or tracebacks."""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIErrorResponse(detail=exc.detail, code=exc.code).model_dump(),
        headers={"Cache-Control": "no-store"},
    )


MODEL_UNAVAILABLE = "The selected prediction model is unavailable."
PREDICTION_FAILED = "Prediction could not be completed."
READINESS_FAILED = "Model readiness could not be determined."


# The endpoint's other validation codes are defined at each HTTP boundary.
