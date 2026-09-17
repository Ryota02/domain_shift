import numpy as np
import torch

from lime import lime_image

from src.xai.image_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def make_classifier(
    model,
    device,
    batch_size,
):
    mean = torch.tensor(
        IMAGENET_MEAN
    ).view(
        1,
        3,
        1,
        1,
    )

    std = torch.tensor(
        IMAGENET_STD
    ).view(
        1,
        3,
        1,
        1,
    )

    def classifier(
        images,
    ):
        images = np.asarray(
            images,
            dtype=np.float32,
        )

        if images.max() > 1.5:
            images /= 255.0

        tensor = (
            torch
            .from_numpy(
                images
            )
            .permute(
                0,
                3,
                1,
                2,
            )
        )

        tensor = (
            tensor - mean
        ) / std

        tensor = tensor.to(
            device
        )

        results = []

        with torch.no_grad():

            for start in range(
                0,
                len(tensor),
                batch_size,
            ):
                batch = tensor[
                    start:
                    start + batch_size
                ]

                logits = model(
                    batch
                ).reshape(
                    -1
                )

                p1 = torch.sigmoid(
                    logits
                )

                p0 = 1.0 - p1

                results.append(
                    torch.stack(
                        [p0, p1],
                        dim=1,
                    ).cpu()
                )

        return torch.cat(
            results
        ).numpy()

    return classifier


def generate_lime(
    model,
    image,
    target_class,
    device,
    num_samples=1000,
    num_features=10,
    batch_size=32,
):
    explainer = (
        lime_image
        .LimeImageExplainer()
    )

    explanation = (
        explainer.explain_instance(
            image,
            make_classifier(
                model,
                device,
                batch_size,
            ),
            labels=[
                target_class
            ],
            num_samples=(
                num_samples
            ),
            hide_color=0,
        )
    )

    segments = (
        explanation.segments
    )

    heatmap = np.zeros(
        segments.shape,
        dtype=np.float32,
    )

    weights = (
        explanation
        .local_exp
        .get(
            target_class,
            [],
        )
    )

    weights = sorted(
        weights,
        key=lambda x: abs(
            x[1]
        ),
        reverse=True,
    )[
        :num_features
    ]

    for segment, weight in weights:

        heatmap[
            segments == segment
        ] = weight

    maximum = np.max(
        np.abs(
            heatmap
        )
    )

    if maximum > 0:
        heatmap /= maximum

    return heatmap