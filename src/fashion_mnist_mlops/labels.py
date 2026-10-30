FASHION_MNIST_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

NUM_CLASSES = len(FASHION_MNIST_CLASSES)


def get_class_name(class_index: int) -> str:
    if (
        not isinstance(class_index, int)
        or isinstance(class_index, bool)
    ):
        raise TypeError(
            "class_index must be an integer."
        )

    if not 0 <= class_index < NUM_CLASSES:
        raise ValueError(
            f"class_index must be between 0 and "
            f"{NUM_CLASSES - 1}."
        )

    return FASHION_MNIST_CLASSES[class_index]
