import numpy as np

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
)

from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)


class BinaryClassifierOutputTarget:

    def __init__(
        self,
        class_idx,
    ):
        self.class_idx = int(
            class_idx
        )

    def __call__(
        self,
        model_output,
    ):
        score = (
            model_output.squeeze()
        )

        if self.class_idx == 1:
            return score

        return -score


def vit_reshape_transform(
    tensor,
):
    # CLS tokenを除去
    tensor = tensor[
        :,
        1:,
        :
    ]

    n_tokens = tensor.size(
        1
    )

    side = int(
        np.sqrt(
            n_tokens
        )
    )

    if side * side != n_tokens:
        raise ValueError(
            f"Invalid ViT tokens: "
            f"{n_tokens}"
        )

    tensor = tensor.reshape(
        tensor.size(0),
        side,
        side,
        tensor.size(2),
    )

    return tensor.permute(
        0,
        3,
        1,
        2,
    )


def get_target_layer(
    model,
    backbone_name,
):
    backbone = getattr(
        model,
        "backbone",
        model,
    )

    name = (
        backbone_name.lower()
    )

    if "resnet" in name:

        return (
            [
                backbone.layer4[-1]
            ],
            None,
        )

    if "densenet" in name:

        return (
            [
                backbone.features
                .denseblock4
            ],
            None,
        )

    if "vit" in name:

        return (
            [
                backbone
                .encoder
                .layers[-1]
                .ln_1
            ],
            vit_reshape_transform,
        )

    raise ValueError(
        f"Unsupported backbone: "
        f"{backbone_name}"
    )


def generate_cam(
    model,
    input_tensor,
    display_image,
    target_class,
    backbone_name,
    method="gradcam",
):
    (
        target_layers,
        reshape_transform,
    ) = get_target_layer(
        model,
        backbone_name,
    )

    if method == "gradcam":
        cam_class = GradCAM

    elif method == "gradcam++":
        cam_class = (
            GradCAMPlusPlus
        )

    else:
        raise ValueError(
            f"Unknown CAM: "
            f"{method}"
        )

    targets = [
        BinaryClassifierOutputTarget(
            target_class
        )
    ]

    cam = cam_class(
        model=model,
        target_layers=(
            target_layers
        ),
        reshape_transform=(
            reshape_transform
        ),
    )

    heatmap = cam(
        input_tensor=(
            input_tensor
        ),
        targets=targets,
    )[0]

    visualization = (
        show_cam_on_image(
            np.clip(
                display_image,
                0,
                1,
            ).astype(
                np.float32
            ),
            heatmap,
            use_rgb=True,
        )
    )

    return (
        heatmap,
        visualization,
    )