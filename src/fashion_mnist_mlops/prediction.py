"""Decode uploaded images and reuse the canonical training preprocessing.

Decoding is independent of MLflow and MinIO. A single PNG/JPEG is decoded
into a PIL image, then the exact training/inference transform in the
data package is applied once. No competing API normalization is defined.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePath
from typing import TYPE_CHECKING

from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from fashion_mnist_mlops.inference import InferencePrediction, RestoredInferenceModel


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024

# Do not infer the actual image type from a user-controlled filename or header.
# Both are validated against the format detected by Pillow.
SUPPORTED_IMAGE_TYPES = {
    "image/png": ("PNG", frozenset({".png"})),
    "image/jpeg": ("JPEG", frozenset({".jpg", ".jpeg"})),
}
SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG"})


class InvalidUploadedImage(ValueError):
    """The uploaded bytes are not a usable single image."""


class UnsupportedUploadedImageType(InvalidUploadedImage):
    """The submitted file format, declared MIME type or extension is unsupported."""


class ImagePreprocessingError(RuntimeError):
    """The trusted shared preprocessing violated the serving tensor contract."""


class InvalidInferencePrediction(RuntimeError):
    """The predictor did not return a valid Fashion-MNIST class result."""


def canonical_class_name(class_index: int) -> str:
    """Return a class label from the single authoritative training class order.

    This check is deliberately strict: do not coerce bools, floats, or strings
    to a plausible class ID, even if an HTTP test double returns them.
    """
    from fashion_mnist_mlops.labels import FASHION_MNIST_CLASSES

    if type(class_index) is not int or not 0 <= class_index < len(FASHION_MNIST_CLASSES):
        raise InvalidInferencePrediction(
            "Inference returned an invalid Fashion-MNIST class index."
        )
    label = FASHION_MNIST_CLASSES[class_index]
    if not isinstance(label, str) or not label.strip():
        raise InvalidInferencePrediction(
            "The configured Fashion-MNIST class label is invalid."
        )
    return label


def validate_image_upload_metadata(*, filename: str | None, content_type: str | None) -> str:
    """Return the required Pillow format based on an explicit PNG/JPEG MIME.

    The filename extension must agree with the declared MIME. The decoder
    separately checks the *actual* image bytes, so neither is trusted alone.
    """
    mime = content_type.strip().lower() if isinstance(content_type, str) else ""
    if mime not in SUPPORTED_IMAGE_TYPES:
        raise UnsupportedUploadedImageType("Only PNG and JPEG image uploads are supported.")

    if not isinstance(filename, str) or not filename:
        raise UnsupportedUploadedImageType("Image filename must have a supported extension.")
    # PurePath handles forward slashes on both Windows and Linux; replacing
    # backslashes avoids platform-specific results for browser Windows paths.
    extension = PurePath(filename.replace("\\", "/")).suffix.lower()
    expected_format, valid_extensions = SUPPORTED_IMAGE_TYPES[mime]
    if extension not in valid_extensions:
        raise UnsupportedUploadedImageType(
            "Image filename extension does not match its declared media type."
        )
    return expected_format


def _check_opened_image(source: Image.Image, expected_format: str | None) -> None:
    if source.format not in SUPPORTED_IMAGE_FORMATS:
        raise UnsupportedUploadedImageType("Only PNG and JPEG image formats are supported.")
    if expected_format is not None and source.format != expected_format:
        raise UnsupportedUploadedImageType("Image content does not match its declared media type.")
    if getattr(source, "n_frames", 1) != 1:
        raise InvalidUploadedImage("Animated or multi-frame images are not supported.")
    if source.width < 1 or source.height < 1:
        raise InvalidUploadedImage("Image dimensions must be positive.")
    if source.width * source.height > MAX_IMAGE_PIXELS:
        raise InvalidUploadedImage("Image dimensions are too large.")


def decode_uploaded_image(content: bytes, *, expected_format: str | None = None) -> Image.Image:
    """Validate a bounded, single-frame PNG/JPEG and return a detached image.

    Verify the complete file structure, then reopen and fully decode pixels.
    Checking the image *before model loading* prevents invalid uploads from
    contacting infrastructure. PIL streams are closed on every failure path.
    """
    if not isinstance(content, bytes) or not content:
        raise InvalidUploadedImage("Upload must contain image bytes.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise InvalidUploadedImage("Uploaded file exceeds the size limit.")
    if expected_format is not None and expected_format not in SUPPORTED_IMAGE_FORMATS:
        raise UnsupportedUploadedImageType("Only PNG and JPEG image formats are supported.")

    try:
        # verify() detects damaged PNG chunks and malformed format metadata,
        # but makes the image unusable for pixel reads; reopen for load().
        with Image.open(BytesIO(content)) as source:
            _check_opened_image(source, expected_format)
            source.verify()
        with Image.open(BytesIO(content)) as source:
            _check_opened_image(source, expected_format)
            source.load()
            return source.copy()
    except (UnidentifiedImageError, OSError, SyntaxError, EOFError, ValueError,
            Image.DecompressionBombError) as exc:
        if isinstance(exc, InvalidUploadedImage):
            raise
        raise InvalidUploadedImage("Uploaded file is not a readable image.") from exc


def preprocess_uploaded_image(image: Image.Image):
    """Return the *shared training* preprocessing result for one uploaded image.

    Import dependencies on demand: ``create_app()`` and ``GET /health`` must
    not import torch, read model configuration or contact remote services.

    The shared data function owns grayscale conversion, 28 x 28 resizing,
    [0, 1] conversion and training-set normalization; the API never redefines
    those operations. Check their output contract before passing it to the
    saved LeNet5 model, so a drift in the preprocessing package fails closed.
    """
    import torch
    from fashion_mnist_mlops.preprocessing import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, preprocess_image_for_inference

    tensor = preprocess_image_for_inference(image)
    expected_shape = (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != expected_shape
        or tensor.dtype != torch.float32
        or tensor.device.type != "cpu"
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ImagePreprocessingError(
            "Shared image preprocessing produced an invalid inference tensor."
        )
    return tensor


def predict_uploaded_image(
    restored: RestoredInferenceModel, image: Image.Image
) -> InferencePrediction:
    """Use gradient-free inference and verify its numeric class result.

    The verified predictor owns ``torch.inference_mode()``, the
    evaluation-mode check, logits validation and argmax. Do not introduce a
    separate forward pass, optimizer, or prediction algorithm in FastAPI.
    """
    from fashion_mnist_mlops.inference import predict_preprocessed_image

    tensor = preprocess_uploaded_image(image)
    result = predict_preprocessed_image(restored, tensor)
    label = canonical_class_name(getattr(result, "class_index", None))
    # returns a label too. If it disagrees with the training taxonomy,
    # reject the inconsistent result rather than reporting the wrong garment.
    inferred_label = getattr(result, "class_name", None)
    if inferred_label is not None and inferred_label != label:
        raise InvalidInferencePrediction(
            "Inference returned an inconsistent Fashion-MNIST class label."
        )
    return result
