"""real Torch inference through , with an HTTP API boundary.

No MLflow/MinIO service, real artifact, subprocess or global networking monkeypatch.
The restored descriptor is constructed in memory solely for this test.
"""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch
from torch import nn

from fashion_mnist_mlops import prediction
from fashion_mnist_mlops.api import create_app
from fashion_mnist_mlops.labels import FASHION_MNIST_CLASSES
from fashion_mnist_mlops.preprocessing import preprocess_image_for_inference
from fashion_mnist_mlops.inference import (
    RestoredInferenceModel,
    predict_preprocessed_image,
)

RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"


class ObservedClassifier(nn.Module):
    """Real differentiable Torch model with 10 controlled Fashion-MNIST logits."""

    def __init__(self, target=7, *, tie=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)
        self.calls = []
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.bias[target] = 5.0
            if tie:
                self.linear.bias[2] = 5.0
        self.eval()

    def forward(self, tensor):
        self.calls.append({
            "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
            "training": self.training,
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "device": tensor.device.type,
        })
        return self.linear(self.flatten(tensor))


def _restored(tmp_path, *, target=7, tie=False):
    return RestoredInferenceModel(
        model=ObservedClassifier(target=target, tie=tie),
        run_id=RUN_ID,
        artifact_path=tmp_path / "test-only-artifact.pt",
        class_names=tuple(FASHION_MNIST_CLASSES),
        sha256="0" * 64,
    )


def _tensor():
    with Image.new("RGB", (37, 19), color=(40, 120, 230)) as image:
        return preprocess_image_for_inference(image)


def _png():
    stream = BytesIO()
    with Image.new("RGB", (37, 19), color=(40, 120, 230)) as image:
        image.save(stream, format="PNG")
    return stream.getvalue()


def _snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_unchanged(model, before):
    assert set(model.state_dict()) == set(before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_inference_inference_uses_argmax_and_returns_correct_index_and_label(tmp_path):
    restored = _restored(tmp_path, target=7)
    tensor = _tensor()
    with torch.inference_mode():
        logits = restored.model(tensor)
    expected = int(torch.argmax(logits, dim=1).item())
    result = predict_preprocessed_image(restored, tensor)
    assert expected == result.class_index == 7
    assert result.class_name == FASHION_MNIST_CLASSES[7]


def test_tied_logits_use_first_maximum_index(tmp_path):
    restored = _restored(tmp_path, target=8, tie=True)
    # target 8 and 2 both have value 5; argmax selects the lowest index.
    result = predict_preprocessed_image(restored, _tensor())
    assert result.class_index == 2


def test_no_gradient_tracking_even_when_caller_enables_gradients(tmp_path):
    restored = _restored(tmp_path)
    model = restored.model
    before = _snapshot(model)
    assert all(p.grad is None for p in model.parameters())
    with torch.enable_grad():
        assert torch.is_grad_enabled()
        result = predict_preprocessed_image(restored, _tensor())
        assert torch.is_grad_enabled()  # caller's previous state is restored
    assert result.class_index == 7
    call = model.calls[-1]
    assert call["grad_enabled"] is False
    assert call["inference_mode"] is True
    assert call["training"] is False
    assert call["shape"] == (1, 1, 28, 28)
    assert call["dtype"] == torch.float32
    assert call["device"] == "cpu"
    assert model.training is False
    assert all(parameter.grad is None for parameter in model.parameters())
    _assert_unchanged(model, before)


def test_existing_gradient_buffers_are_not_reset_or_overwritten(tmp_path):
    restored = _restored(tmp_path)
    model = restored.model
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.125)
    before_parameters = _snapshot(model)
    before_grads = [p.grad.clone() for p in model.parameters()]
    predict_preprocessed_image(restored, _tensor())
    _assert_unchanged(model, before_parameters)
    for p, g in zip(model.parameters(), before_grads):
        torch.testing.assert_close(p.grad, g, rtol=0, atol=0)


def test_repeated_inference_keeps_model_state_and_is_deterministic(tmp_path):
    restored = _restored(tmp_path, target=4)
    before = _snapshot(restored.model)
    outputs = [predict_preprocessed_image(restored, _tensor()) for _ in range(4)]
    assert all(out == outputs[0] for out in outputs)
    assert outputs[0].class_index == 4
    assert restored.model.training is False
    _assert_unchanged(restored.model, before)


def test_inference_rejects_training_mode_inference(tmp_path):
    restored = _restored(tmp_path)
    restored.model.train()
    before = _snapshot(restored.model)
    with pytest.raises(ValueError, match="evaluation mode"):
        predict_preprocessed_image(restored, _tensor())
    assert restored.model.calls == []
    _assert_unchanged(restored.model, before)


@pytest.mark.parametrize("bad_logits", [
    torch.zeros((1, 9)),
    torch.zeros((2, 10)),
    torch.full((1, 10), float("nan")),
    torch.full((1, 10), float("inf")),
])
def test_inference_rejects_invalid_logits(tmp_path, monkeypatch, bad_logits):
    restored = _restored(tmp_path)
    before = _snapshot(restored.model)
    monkeypatch.setattr(restored.model, "forward", lambda tensor: bad_logits)
    with pytest.raises(ValueError, match="logits"):
        predict_preprocessed_image(restored, _tensor())
    _assert_unchanged(restored.model, before)


def test_http_prediction_uses_real_inference_inference_without_mutation(tmp_path, monkeypatch):
    application = create_app()
    restored = _restored(tmp_path, target=6)
    before = _snapshot(restored.model)
    loads = []
    def load():
        loads.append(1)
        return restored
    monkeypatch.setattr(application.state.model_loader, "load", load)
    with TestClient(application) as client:
        for _ in range(2):
            response = client.post(
                "/predict", files={"file": ("shirt.png", _png(), "image/png")}
            )
            assert response.status_code == 200, response.text
            assert response.json() == {"class_index": 6, "class_name": "Shirt", "model_run_id": RUN_ID}
        assert client.get("/health").json() == {"status": "ok"}
    assert len(loads) == 2  # loader.load called each time; caching tested in 8.4
    assert len(restored.model.calls) == 2
    assert all(call["inference_mode"] for call in restored.model.calls)
    assert all(not call["grad_enabled"] for call in restored.model.calls)
    assert all(not call["training"] for call in restored.model.calls)
    assert all(p.grad is None for p in restored.model.parameters())
    _assert_unchanged(restored.model, before)


@pytest.mark.parametrize("index", [0, 3, 9])
def test_api_accepts_valid_integer_class_index(monkeypatch, index):
    import fashion_mnist_mlops.inference as inference
    monkeypatch.setattr(
        inference, "predict_preprocessed_image",
        lambda restored, tensor: SimpleNamespace(class_index=index),
    )
    with Image.new("L", (28, 28)) as image:
        result = prediction.predict_uploaded_image(SimpleNamespace(run_id=RUN_ID), image)
    assert result.class_index == index


@pytest.mark.parametrize("index", [-1, 10, True, False, "7", 3.0, None, [], 100])
def test_api_rejects_malformed_class_index_without_coercion(monkeypatch, index):
    import fashion_mnist_mlops.inference as inference
    monkeypatch.setattr(
        inference, "predict_preprocessed_image",
        lambda restored, tensor: SimpleNamespace(class_index=index),
    )
    with Image.new("L", (28, 28)) as image:
        with pytest.raises(prediction.InvalidInferencePrediction, match="invalid Fashion-MNIST"):
            prediction.predict_uploaded_image(SimpleNamespace(run_id=RUN_ID), image)


def test_api_rejects_result_without_class_index(monkeypatch):
    import fashion_mnist_mlops.inference as inference
    monkeypatch.setattr(
        inference, "predict_preprocessed_image",
        lambda restored, tensor: SimpleNamespace(other_field=3),
    )
    with Image.new("L", (28, 28)) as image:
        with pytest.raises(prediction.InvalidInferencePrediction):
            prediction.predict_uploaded_image(SimpleNamespace(run_id=RUN_ID), image)


def test_app_creation_and_health_do_not_trigger_inference(monkeypatch):
    application = create_app()
    def forbidden():
        raise AssertionError("loading model should not occur on GET /health")
    monkeypatch.setattr(application.state.model_loader, "load", forbidden)
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/openapi.json").status_code == 200
