import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

from PIL import Image


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_images(directory):
    directory = Path(directory)

    return sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def load_model(device):
    print(
        "[INFO] Loading ChestX-Det PSPNet..."
    )

    model = (
        xrv.baseline_models
        .chestx_det
        .PSPNet()
    )

    model = model.to(device)
    model.eval()

    print(
        "[INFO] Targets:"
    )

    for index, target in enumerate(
        model.targets
    ):
        print(
            index,
            target
        )

    return model


def predict_lung_mask(
    image_path,
    model,
    device,
    threshold=0.5,
):
    # --------------------------------
    # Original image information
    # --------------------------------

    original = Image.open(
        image_path
    ).convert("L")

    original_width, original_height = (
        original.size
    )

    # --------------------------------
    # TorchXRayVision preprocessing
    # --------------------------------
    #
    # load_image:
    #   single channel
    #   approximately [-1024, 1024]
    #
    # PSPNet itself performs
    # 512x512 resolution adjustment.
    # --------------------------------

    image = xrv.utils.load_image(
        str(image_path)
    )

    input_tensor = (
        torch.from_numpy(
            image
        )
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # [1, 1, H, W]

    # --------------------------------
    # Prediction
    # --------------------------------

    with torch.no_grad():
        logits = model(
            input_tensor
        )

    # [1, 14, 512, 512]

    probabilities = torch.sigmoid(
        logits
    )

    # --------------------------------
    # Lung channels
    # --------------------------------

    left_index = (
        model.targets.index(
            "Left Lung"
        )
    )

    right_index = (
        model.targets.index(
            "Right Lung"
        )
    )

    left_probability = (
        probabilities[
            0,
            left_index,
        ]
    )

    right_probability = (
        probabilities[
            0,
            right_index,
        ]
    )

    # --------------------------------
    # Left + Right lung
    # --------------------------------

    lung_probability = torch.maximum(
        left_probability,
        right_probability,
    )

    lung_mask = (
        lung_probability
        >= threshold
    ).float()

    # --------------------------------
    # Return to original resolution
    # --------------------------------

    lung_mask = (
        lung_mask
        .unsqueeze(0)
        .unsqueeze(0)
    )

    lung_mask = F.interpolate(
        lung_mask,
        size=(
            original_height,
            original_width,
        ),
        mode="nearest",
    )

    lung_mask = (
        lung_mask[
            0,
            0,
        ]
        .cpu()
        .numpy()
        .astype(
            np.uint8
        )
    )

    return lung_mask


def create_lung_only(
    image_path,
    lung_mask,
):
    image = Image.open(
        image_path
    ).convert("L")

    image_array = np.asarray(
        image,
        dtype=np.uint8,
    )

    lung_only = np.zeros_like(
        image_array
    )

    lung_only[
        lung_mask > 0
    ] = image_array[
        lung_mask > 0
    ]

    return lung_only


def save_mask(
    lung_mask,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        lung_mask * 255
    ).save(
        output_path
    )


def save_lung_only(
    lung_only,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        lung_only
    ).save(
        output_path
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    mask_dir = (
        output_dir
        / "mask"
    )

    lung_only_dir = (
        output_dir
        / "lung_only"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "[INFO] Device:",
        device
    )

    model = load_model(
        device
    )

    image_paths = find_images(
        input_dir
    )

    print(
        "[INFO] Images:",
        len(image_paths)
    )

    success = 0
    failed = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        try:
            relative_path = (
                image_path.relative_to(
                    input_dir
                )
            )

            lung_mask = predict_lung_mask(
                image_path=image_path,
                model=model,
                device=device,
                threshold=args.threshold,
            )

            # -------------------------
            # Sanity check
            # -------------------------

            mask_ratio = float(
                lung_mask.mean()
            )

            if mask_ratio < 0.05:
                print(
                    "[WARNING] Very small mask:",
                    image_path.name,
                    mask_ratio,
                )

            if mask_ratio > 0.70:
                print(
                    "[WARNING] Very large mask:",
                    image_path.name,
                    mask_ratio,
                )

            # -------------------------
            # Lung-only image
            # -------------------------

            lung_only = (
                create_lung_only(
                    image_path,
                    lung_mask,
                )
            )

            # -------------------------
            # Save mask
            # -------------------------

            mask_path = (
                mask_dir
                / relative_path
            )

            save_mask(
                lung_mask,
                mask_path,
            )

            # -------------------------
            # Save lung-only
            # -------------------------

            lung_only_path = (
                lung_only_dir
                / relative_path
            )

            save_lung_only(
                lung_only,
                lung_only_path,
            )

            success += 1

            print(
                f"[{index}/"
                f"{len(image_paths)}] "
                f"{image_path.name} "
                f"lung_ratio="
                f"{mask_ratio:.3f}"
            )

        except Exception as error:
            failed += 1

            print(
                "[ERROR]",
                image_path,
                error,
            )

    print(
        "\n[INFO] Finished"
    )

    print(
        "[INFO] Success:",
        success
    )

    print(
        "[INFO] Failed:",
        failed
    )

    print(
        "[INFO] Output:",
        output_dir
    )


if __name__ == "__main__":
    main()