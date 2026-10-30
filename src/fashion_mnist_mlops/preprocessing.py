import torch
from PIL import Image
from torchvision import transforms


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1

NORMALIZATION_MEAN = (0.2860406,)
NORMALIZATION_STD = (0.3530242,)


def build_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=IMAGE_CHANNELS,
            ),
            transforms.Resize(
                (IMAGE_HEIGHT, IMAGE_WIDTH),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN,
                std=NORMALIZATION_STD,
            ),
        ]
    )


def preprocess_image_for_inference(
    image: Image.Image,
) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be a PIL.Image.Image."
        )

    transform = build_image_transform()

    image_tensor = transform(image)

    return image_tensor.unsqueeze(0)
