import numpy as np
import shap
import torch
import torch.nn as nn


class BinarySHAPWrapper(
    nn.Module
):

    def __init__(
        self,
        model,
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        x,
    ):
        return self.model(
            x
        ).reshape(
            -1,
            1,
        )


def build_background(
    dataset,
    device,
    n_background=8,
):
    n = min(
        n_background,
        len(dataset),
    )

    tensors = [
        dataset[i][0]
        for i in range(n)
    ]

    return torch.stack(
        tensors
    ).to(
        device
    )


def generate_shap(
    model,
    input_tensor,
    background,
    target_class,
    nsamples=100,
):
    wrapper = (
        BinarySHAPWrapper(
            model
        )
    )

    wrapper.eval()

    explainer = (
        shap.GradientExplainer(
            wrapper,
            background,
        )
    )

    values = (
        explainer.shap_values(
            input_tensor,
            nsamples=nsamples,
        )
    )

    if isinstance(
        values,
        list,
    ):
        values = values[0]

    values = np.asarray(
        values
    )

    if (
        values.ndim == 5
        and values.shape[-1] == 1
    ):
        values = values[
            ...,
            0
        ]

    if (
        values.ndim == 4
        and values.shape[0] == 1
    ):
        values = values[0]

    if (
        values.ndim == 3
        and values.shape[0]
        in (1, 3)
    ):
        heatmap = np.mean(
            values,
            axis=0,
        )

    else:
        heatmap = np.squeeze(
            values
        )

    if target_class == 0:
        heatmap = -heatmap

    maximum = np.max(
        np.abs(
            heatmap
        )
    )

    if maximum > 0:
        heatmap /= maximum

    return heatmap