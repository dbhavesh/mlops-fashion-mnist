"""Streamlit image upload, prediction results, and safe HTTP errors.

Model loading, preprocessing, and inference belong to the API.
"""

from __future__ import annotations

import streamlit as st

from fashion_mnist_mlops.ui_client import (
    PredictionClientError,
    PredictionFailure,
    predict_image,
)
from fashion_mnist_mlops.ui_config import (
    FASTAPI_BASE_URL_ENV,
    UIConfigurationError,
    load_ui_settings,
)


PAGE_TITLE = "Fashion-MNIST Classifier"
SUPPORTED_IMAGE_TYPES = ("png", "jpg", "jpeg")

_HTTP_ERROR_MESSAGES = {
    400: "FastAPI rejected the image as invalid or unreadable. Choose a valid PNG or JPEG image.",
    413: "The image is too large for FastAPI. Choose an image smaller than 5 MiB.",
    415: "FastAPI does not support this image type. Choose a PNG or JPEG image.",
    422: "FastAPI could not validate the upload request. Re-select a PNG or JPEG image and try again.",
    500: "FastAPI encountered an internal prediction error. Try again; if it persists, check the API logs.",
    503: "The selected model is currently unavailable. Confirm FastAPI readiness and its MLflow/MinIO dependencies.",
}


def _prediction_error_message(error: PredictionClientError) -> str:
    """Convert sanitized client failures into actionable UI guidance."""
    if error.failure == PredictionFailure.TIMEOUT:
        return "The prediction request timed out. Confirm FastAPI is responsive and try again."
    if error.failure == PredictionFailure.CONNECTION:
        return (
            "Could not connect to the FastAPI backend. Confirm it is running and "
            f"{FASTAPI_BASE_URL_ENV} is correct."
        )
    if error.failure == PredictionFailure.TRANSPORT:
        return (
            "The prediction request could not reach FastAPI. Check the backend "
            "and network, then try again."
        )
    if error.failure == PredictionFailure.INVALID_UPLOAD:
        return "The selected file could not be sent. Choose a non-empty PNG or JPEG image."
    if error.failure == PredictionFailure.INVALID_RESPONSE:
        return (
            "FastAPI returned an unexpected prediction response. Confirm the API "
            "and UI versions are compatible, then try again."
        )
    if error.failure == PredictionFailure.HTTP:
        if error.status_code in _HTTP_ERROR_MESSAGES:
            return _HTTP_ERROR_MESSAGES[error.status_code]
        if error.status_code is not None:
            return (
                f"FastAPI returned HTTP {error.status_code}. Prediction could not "
                "be completed."
            )

    return "Prediction could not be completed. Check FastAPI and try again."


def render_app() -> None:
    """Render upload, progress feedback, prediction result, and safe errors."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="👕",
        layout="centered",
    )

    st.title(PAGE_TITLE)
    st.write(
        "Classify clothing images with the trained Fashion-MNIST model served "
        "by the FastAPI backend."
    )

    try:
        settings = load_ui_settings()
    except UIConfigurationError:
        st.error(
            "The FastAPI backend address is invalid. Set FASTAPI_BASE_URL to "
            "a valid HTTP(S) service origin."
        )
        st.stop()
        return

    st.subheader("How to use this application")
    st.markdown(
        "1. Choose a supported PNG or JPEG image.\n"
        "2. Review the image preview.\n"
        "3. Select **Predict** to send the image to FastAPI.\n"
        "4. Review the returned Fashion-MNIST class and model run identifier."
    )

    uploaded_file = st.file_uploader(
        "Upload a PNG or JPEG image",
        type=list(SUPPORTED_IMAGE_TYPES),
        accept_multiple_files=False,
        help=(
            "Choose one image only. The FastAPI backend performs authoritative "
            "file validation before inference."
        ),
    )

    if uploaded_file is None:
        st.caption("Choose one PNG or JPEG image to enable prediction.")
    else:
        st.image(uploaded_file, caption=f"Preview: {uploaded_file.name}")
        st.caption(
            "Preview only. FastAPI will perform final image validation before "
            "classification."
        )

    predict_clicked = st.button(
        "Predict",
        type="primary",
        disabled=uploaded_file is None,
    )
    if predict_clicked and uploaded_file is not None:
        try:
            with st.spinner("Requesting prediction from FastAPI..."):
                result = predict_image(
                    base_url=settings.fastapi_base_url,
                    filename=uploaded_file.name,
                    content=uploaded_file.getvalue(),
                    content_type=uploaded_file.type,
                )
        except PredictionClientError as exc:
            st.error(_prediction_error_message(exc))
        else:
            st.subheader("Prediction result")
            st.success(f"Predicted class: **{result.class_name}**")
            st.write(f"Class index: `{result.class_index}`")
            st.caption(f"Model run ID: `{result.model_run_id}`")

    st.caption(f"FastAPI backend: `{settings.fastapi_base_url}`")
    st.caption(f"Configure with the `{FASTAPI_BASE_URL_ENV}` environment variable.")


if __name__ == "__main__":
    render_app()
