import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from PIL import Image
from scipy.stats import (
    mannwhitneyu,
    ttest_ind,
)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def load_config(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def find_images(directory):
    directory = Path(directory)

    return sorted(
        p
        for p in directory.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def find_mask(
    mask_root,
    image_stem,
    mask_name="lung.png",
):
    """
    対応するmaskを検索する．

    対応形式:

    CXAS:
        mask/F0001002/lung.png
        mask/F0001002/left lung.png
        mask/F0001002/right lung.png

    Flat full-lung mask:
        mask/F0001002_mask.png
        mask/F0001002.png

    Flat left/right mask（存在する場合）:
        mask/F0001002_left_lung_mask.png
        mask/F0001002_right_lung_mask.png
    """

    mask_root = Path(
        mask_root
    )

    # --------------------------------
    # 1. CXAS original format
    # --------------------------------

    path = (
        mask_root
        / image_stem
        / mask_name
    )

    if path.exists():
        return path

    # --------------------------------
    # 2. Full lung mask
    # --------------------------------

    if mask_name == "lung.png":

        candidates = [
            mask_root
            / f"{image_stem}_mask.png",

            mask_root
            / f"{image_stem}.png",
        ]

        for path in candidates:
            if path.exists():
                return path

        return None

    # --------------------------------
    # 3. Left / Right lung masks
    # --------------------------------

    mask_stem = Path(
        mask_name
    ).stem

    normalized_name = (
        mask_stem
        .replace(" ", "_")
    )

    candidates = [
        # F0001002_left_lung.png
        mask_root
        / f"{image_stem}_{normalized_name}.png",

        # F0001002_left_lung_mask.png
        mask_root
        / f"{image_stem}_{normalized_name}_mask.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None

def load_mask(
    path,
    target_size=None,
):
    mask = Image.open(
        path
    ).convert("L")

    # 元画像サイズへ合わせる
    if target_size is not None:
        if mask.size != target_size:
            mask = mask.resize(
                target_size,
                resample=Image.Resampling.NEAREST,
            )

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return mask >= 128


def calculate_statistics(
    image_path,
    mask_path,
):
    # -------------------------
    # Original image
    # -------------------------

    image_pil = Image.open(
        image_path
    ).convert("L")

    original_size = (
        image_pil.size
    )

    image = np.asarray(
        image_pil,
        dtype=np.float64,
    )

    # 0 - 1
    image = image / 255.0

    # -------------------------
    # Lung mask
    # -------------------------

    mask = load_mask(
        mask_path,
        target_size=original_size,
    )

    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch after resize: "
            f"image={image.shape}, "
            f"mask={mask.shape}"
        )

    pixels = image[
        mask
    ]

    if pixels.size == 0:
        raise ValueError(
            "Empty lung mask"
        )

    # -------------------------
    # Lung size
    # -------------------------

    height, width = (
        mask.shape
    )

    lung_area_px = int(
        mask.sum()
    )

    image_area_px = (
        height * width
    )

    lung_area_ratio = (
        lung_area_px
        / image_area_px
    )

    # -------------------------
    # Bounding box
    # -------------------------

    ys, xs = np.where(
        mask
    )

    x_min = int(
        xs.min()
    )

    x_max = int(
        xs.max()
    )

    y_min = int(
        ys.min()
    )

    y_max = int(
        ys.max()
    )

    bbox_width = (
        x_max - x_min + 1
    )

    bbox_height = (
        y_max - y_min + 1
    )

    bbox_area = (
        bbox_width
        * bbox_height
    )

    bbox_area_ratio = (
        bbox_area
        / image_area_px
    )

    # 画像サイズが異なっても比較可能にする
    bbox_width_ratio = (
        bbox_width / width
    )

    bbox_height_ratio = (
        bbox_height / height
    )

    # -------------------------
    # Pixel statistics
    # -------------------------

    mean = float(
        np.mean(pixels)
    )

    median = float(
        np.median(pixels)
    )

    variance = float(
        np.var(
            pixels,
            ddof=0,
        )
    )

    std = float(
        np.std(
            pixels,
            ddof=0,
        )
    )

    # -------------------------
    # Moments
    # -------------------------

    first_moment = mean

    second_raw_moment = float(
        np.mean(
            pixels ** 2
        )
    )

    second_central_moment = float(
        np.mean(
            (
                pixels - mean
            ) ** 2
        )
    )

    return {
        "width": width,
        "height": height,

        "lung_area_px":
            lung_area_px,

        "lung_area_ratio":
            lung_area_ratio,

        "bbox_width":
            bbox_width,

        "bbox_height":
            bbox_height,

        "bbox_width_ratio":
            bbox_width_ratio,

        "bbox_height_ratio":
            bbox_height_ratio,

        "bbox_area_ratio":
            bbox_area_ratio,

        "mean":
            mean,

        "median":
            median,

        "variance":
            variance,

        "std":
            std,

        "first_moment":
            first_moment,

        "second_raw_moment":
            second_raw_moment,

        "second_central_moment":
            second_central_moment,
    }


def calculate_left_right_area(
    mask_root,
    image_stem,
    image_size,
):
    left_path = find_mask(
        mask_root,
        image_stem,
        "left lung.png",
    )

    right_path = find_mask(
        mask_root,
        image_stem,
        "right lung.png",
    )

    if (
        left_path is None
        or right_path is None
    ):
        return {}

    left = load_mask(
        left_path,
        target_size=image_size,
    )

    right = load_mask(
        right_path,
        target_size=image_size,
    )

    total_pixels = (
        left.shape[0]
        * left.shape[1]
    )

    left_area = int(
        left.sum()
    )

    right_area = int(
        right.sum()
    )

    return {
        "left_lung_area_px":
            left_area,

        "right_lung_area_px":
            right_area,

        "left_lung_area_ratio":
            left_area / total_pixels,

        "right_lung_area_ratio":
            right_area / total_pixels,

        "left_right_area_ratio":
            (
                left_area / right_area
                if right_area > 0
                else np.nan
            ),
    }

def collect_dataset(
    class_name,
    split,
    image_dir,
    mask_dir,
):
    rows = []

    image_paths = find_images(
        image_dir
    )

    print(
        f"[INFO] {class_name} "
        f"{split}: "
        f"{len(image_paths)} images"
    )

    for image_path in image_paths:

        stem = image_path.stem

        mask_path = find_mask(
            mask_dir,
            stem,
        )

        if mask_path is None:
            print(
                "[SKIP] Mask not found:",
                stem,
            )
            continue

        try:
            stats = (
                calculate_statistics(
                    image_path,
                    mask_path,
                )
            )

            with Image.open(
                image_path
            ) as image:
                image_size = image.size
            
            lr_stats = (
                calculate_left_right_area(
                    mask_dir,
                    stem,
                    image_size,
                )
            )

            row = {
                "class":
                    class_name,

                "split":
                    split,

                "image":
                    str(image_path),

                "mask":
                    str(mask_path),
            }

            row.update(
                stats
            )

            row.update(
                lr_stats
            )

            rows.append(
                row
            )

        except Exception as e:
            print(
                "[ERROR]",
                stem,
                e,
            )

    return rows


def cohens_d(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    nx = len(x)
    ny = len(y)

    pooled_var = (
        (
            (nx - 1)
            * np.var(
                x,
                ddof=1,
            )
        )
        +
        (
            (ny - 1)
            * np.var(
                y,
                ddof=1,
            )
        )
    ) / (
        nx + ny - 2
    )

    if pooled_var <= 0:
        return np.nan

    return float(
        (
            np.mean(x)
            - np.mean(y)
        )
        /
        np.sqrt(
            pooled_var
        )
    )


def statistical_tests(
    df,
    metrics,
):
    rows = []

    classes = sorted(
        df["class"].unique()
    )

    if len(classes) != 2:
        return pd.DataFrame()

    for split in sorted(
        df["split"].unique()
    ):

        split_df = df[
            df["split"] == split
        ]

        for metric in metrics:

            x = split_df[
                split_df["class"]
                == classes[0]
            ][metric].dropna()

            y = split_df[
                split_df["class"]
                == classes[1]
            ][metric].dropna()

            if (
                len(x) < 2
                or len(y) < 2
            ):
                continue

            t_result = ttest_ind(
                x,
                y,
                equal_var=False,
            )

            u_result = mannwhitneyu(
                x,
                y,
                alternative="two-sided",
            )

            rows.append({
                "split":
                    split,

                "metric":
                    metric,

                "class_0":
                    classes[0],

                "class_1":
                    classes[1],

                "mean_0":
                    float(x.mean()),

                "mean_1":
                    float(y.mean()),

                "median_0":
                    float(x.median()),

                "median_1":
                    float(y.median()),

                "welch_t_p":
                    float(
                        t_result.pvalue
                    ),

                "mannwhitney_p":
                    float(
                        u_result.pvalue
                    ),

                "cohens_d":
                    cohens_d(
                        x,
                        y,
                    ),
            })

    return pd.DataFrame(
        rows
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    rows = []

    for class_name, paths in (
        cfg["dataset"].items()
    ):

        for split in [
            "train",
            "test",
        ]:

            rows.extend(
                collect_dataset(
                    class_name=(
                        class_name
                    ),
                    split=split,
                    image_dir=paths[
                        f"{split}_images"
                    ],
                    mask_dir=paths[
                        f"{split}_masks"
                    ],
                )
            )

    df = pd.DataFrame(
        rows
    )

    output_dir = Path(
        cfg["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Per-image
    # -------------------------

    df.to_csv(
        output_dir
        / "per_image_statistics.csv",
        index=False,
    )

    metrics = [
        # Raw size
        "lung_area_px",
        "bbox_width",
        "bbox_height",
    
        # Normalized lung size
        "lung_area_ratio",
        "bbox_width_ratio",
        "bbox_height_ratio",
        "bbox_area_ratio",
    
        # Pixel statistics
        "mean",
        "median",
        "variance",
        "std",
    
        # Moments
        "first_moment",
        "second_raw_moment",
        "second_central_moment",
    ]

    optional_metrics = [
        "left_lung_area_ratio",
        "right_lung_area_ratio",
        "left_right_area_ratio",
    ]

    metrics += [
        m
        for m in optional_metrics
        if m in df.columns
    ]

    # -------------------------
    # Summary
    # -------------------------

    summary = (
        df.groupby(
            [
                "split",
                "class",
            ]
        )[metrics]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ])
    )
    
    # MultiIndex columnsを1段にする
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in summary.columns
    ]
    
    summary = (
        summary
        .reset_index()
    )
    
    summary.to_csv(
        output_dir
        / "class_summary.csv",
        index=False,
    )

    # -------------------------
    # Statistical tests
    # -------------------------

    tests = statistical_tests(
        df,
        metrics,
    )

    tests.to_csv(
        output_dir
        / "statistical_tests.csv",
        index=False,
    )

    print(
        "\n[INFO] Saved:"
    )

    print(
        output_dir
        / "per_image_statistics.csv"
    )

    print(
        output_dir
        / "class_summary.csv"
    )

    print(
        output_dir
        / "statistical_tests.csv"
    )


if __name__ == "__main__":
    main()