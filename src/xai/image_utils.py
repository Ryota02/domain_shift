from pathlib import Path

import cv2
import numpy as np


IMAGENET_MEAN = np.array(
    [
        0.485,
        0.456,
        0.406,
    ],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [
        0.229,
        0.224,
        0.225,
    ],
    dtype=np.float32,
)


def denormalize_tensor(
    tensor,
):
    image = (
        tensor
        .detach()
        .cpu()
        .numpy()
    )

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    image = (
        image
        * IMAGENET_STD
        + IMAGENET_MEAN
    )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(
        np.float32
    )


def load_original_image(
    image_path,
):
    image_path = Path(
        image_path
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot load: "
            f"{image_path}"
        )

    return image


def resize_heatmap(
    heatmap,
    target_shape,
):
    if heatmap is None:
        return None

    height, width = (
        target_shape
    )

    if heatmap.shape == (
        height,
        width,
    ):
        return heatmap

    return cv2.resize(
        heatmap.astype(
            np.float32
        ),
        (
            width,
            height,
        ),
        interpolation=(
            cv2.INTER_LINEAR
        ),
    )