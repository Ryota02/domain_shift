from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_xai_figure(
    original_image,
    model_input,
    gradcam_image=None,
    gradcam_pp_image=None,
    rollout_image=None,
    lime_heatmap=None,
    shap_heatmap=None,
    output_path=None,
    title="",
    dpi=200,
):
    plots = [
        (
            "Original",
            original_image,
            None,
        ),
        (
            "Model input",
            model_input,
            None,
        ),
    ]

    if gradcam_image is not None:
        plots.append(
            (
                "Grad-CAM",
                gradcam_image,
                None,
            )
        )

    if gradcam_pp_image is not None:
        plots.append(
            (
                "Grad-CAM++",
                gradcam_pp_image,
                None,
            )
        )
        
    if rollout_image is not None:
        plots.append(
            (
                "Attention Rollout",
                rollout_image,
                None,
            )
        )

    if lime_heatmap is not None:
        plots.append(
            (
                "LIME",
                model_input,
                lime_heatmap,
            )
        )

    if shap_heatmap is not None:
        plots.append(
            (
                "SHAP",
                model_input,
                shap_heatmap,
            )
        )

    n_cols = 3

    n_rows = int(
        np.ceil(
            len(plots)
            / n_cols
        )
    )

    fig = plt.figure(
        figsize=(
            18,
            5 * n_rows,
        )
    )

    for i, (
        name,
        image,
        heatmap,
    ) in enumerate(
        plots,
        1,
    ):
        ax = plt.subplot(
            n_rows,
            n_cols,
            i,
        )

        if image.ndim == 2:
            ax.imshow(
                image,
                cmap="gray",
            )
        else:
            ax.imshow(
                image
            )

        if heatmap is not None:

            overlay = ax.imshow(
                heatmap,
                cmap="coolwarm",
                alpha=0.5,
                vmin=-1,
                vmax=1,
            )

            plt.colorbar(
                overlay,
                ax=ax,
                fraction=0.046,
            )

        ax.set_title(
            name
        )

        ax.axis(
            "off"
        )

    fig.suptitle(
        title
    )

    plt.tight_layout()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )