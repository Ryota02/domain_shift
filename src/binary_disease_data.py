# src/binary_disease_dataset.py

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Utility
# ============================================================

def find_images(
    directory,
):
    """
    指定ディレクトリ以下の画像を再帰的に取得する．
    """

    directory = Path(
        directory
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )

    image_paths = sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    return image_paths


# ============================================================
# Mask
# ============================================================

def find_mask(
    mask_root,
    image_stem,
):
    """
    元画像に対応するlung maskを検索する．

    以下の形式に対応:

    1. CXAS original
       mask/F0001002/lung.png

    2. flat mask
       mask/F0001002_mask.png

    3. same name
       mask/F0001002.png
    """

    if mask_root is None:
        return None

    mask_root = Path(
        mask_root
    )

    if not mask_root.exists():
        return None

    candidates = [
        # F0001002_mask.png
        mask_root
        / f"{image_stem}_mask.png",

        # F0001002.png
        mask_root
        / f"{image_stem}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_lung_mask(
    mask_path,
    target_size,
    threshold=128,
):
    """
    Lung maskを読み込み，
    元画像のサイズに合わせる．

    target_size:
        PIL形式 (width, height)

    return:
        bool ndarray [H, W]
    """

    mask_pil = Image.open(
        mask_path
    ).convert("L")

    # --------------------------------
    # Mask size -> Original image size
    # --------------------------------

    if mask_pil.size != target_size:

        mask_pil = mask_pil.resize(
            target_size,
            resample=(
                Image.Resampling.NEAREST
            ),
        )

    mask = np.asarray(
        mask_pil,
        dtype=np.uint8,
    )

    mask = (
        mask >= threshold
    )

    return mask

def lung_only_image(
    image,
    mask_path,
    mask_threshold=128,
):
    """
    元画像の肺野内だけを残す．

    Lung:
        original pixel

    Outside lung:
        0
    """

    image = image.convert(
        "L"
    )

    original_size = image.size

    image_array = np.asarray(
        image,
        dtype=np.uint8,
    )

    mask = load_lung_mask(
        mask_path=mask_path,
        target_size=original_size,
        threshold=mask_threshold,
    )

    if image_array.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"image={image_array.shape}, "
            f"mask={mask.shape}"
        )

    output = np.zeros_like(
        image_array,
        dtype=np.uint8,
    )

    output[mask] = (
        image_array[mask]
    )

    output_image = Image.fromarray(
        output
    ).convert(
        "RGB"
    )

    return output_image


# ============================================================
# Moment standardization
# ============================================================

def moment_standardize_image(
    image,
    mask_path,
    mask_threshold=128,
    clip_z=3.0,
    eps=1e-8,
):
    """
    肺野内画素を用いて画像ごとにmoment standardizationを行う．

    First raw moment:
        m1 = E[X]

    Second raw moment:
        m2 = E[X^2]

    Variance:
        Var(X) = m2 - m1^2

    Standardization:
        z = (x - m1) / sqrt(Var(X))

    肺野外は0にする．
    """

    # ========================================================
    # Original image
    # ========================================================

    image = image.convert(
        "L"
    )

    original_size = (
        image.size
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # 0 - 255
    # ↓
    # 0 - 1

    image_array = (
        image_array
        / 255.0
    )

    # ========================================================
    # Lung mask
    # ========================================================

    mask = load_lung_mask(
        mask_path=mask_path,
        target_size=original_size,
        threshold=mask_threshold,
    )

    if (
        image_array.shape
        != mask.shape
    ):
        raise ValueError(
            f"Shape mismatch: "
            f"image={image_array.shape}, "
            f"mask={mask.shape}"
        )

    lung_pixels = (
        image_array[
            mask
        ]
    )

    if lung_pixels.size == 0:
        raise ValueError(
            f"Empty lung mask: "
            f"{mask_path}"
        )

    # ========================================================
    # First moment
    #
    # m1 = E[X]
    # ========================================================

    first_moment = float(
        np.mean(
            lung_pixels
        )
    )

    # ========================================================
    # Second raw moment
    #
    # m2 = E[X^2]
    # ========================================================

    second_raw_moment = float(
        np.mean(
            lung_pixels ** 2
        )
    )

    # ========================================================
    # Variance
    #
    # Var(X)
    # =
    # E[X^2] - E[X]^2
    # ========================================================

    variance = (
        second_raw_moment
        - first_moment ** 2
    )

    variance = max(
        variance,
        eps,
    )

    std = float(
        np.sqrt(
            variance
        )
    )

    # ========================================================
    # Standardization
    # ========================================================

    standardized = np.zeros_like(
        image_array,
        dtype=np.float32,
    )

    standardized[
        mask
    ] = (
        image_array[
            mask
        ]
        - first_moment
    ) / std

    # ========================================================
    # Clip
    # ========================================================

    if clip_z is not None:

        standardized[
            mask
        ] = np.clip(
            standardized[
                mask
            ],
            -clip_z,
            clip_z,
        )

    # ========================================================
    # PNG/PILとして扱える0-1へ変換
    #
    # -clip_z -> 0
    # 0       -> 0.5
    # +clip_z -> 1
    # ========================================================

    normalized = np.zeros_like(
        standardized,
        dtype=np.float32,
    )

    if clip_z is not None:

        normalized[
            mask
        ] = (
            standardized[
                mask
            ]
            + clip_z
        ) / (
            2.0
            * clip_z
        )

    else:
        # clipしない場合の安全なmin-max
        lung_standardized = (
            standardized[
                mask
            ]
        )

        minimum = float(
            lung_standardized.min()
        )

        maximum = float(
            lung_standardized.max()
        )

        denominator = max(
            maximum - minimum,
            eps,
        )

        normalized[
            mask
        ] = (
            lung_standardized
            - minimum
        ) / denominator

    # --------------------------------
    # Outside lung = black
    # --------------------------------

    normalized[
        ~mask
    ] = 0.0

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    normalized_uint8 = (
        normalized
        * 255.0
    ).round().astype(
        np.uint8
    )

    # --------------------------------
    # ViT / ResNet用にRGBへ
    # --------------------------------

    output_image = Image.fromarray(
        normalized_uint8
    ).convert(
        "RGB"
    )

    return output_image


# ============================================================
# Dataset
# ============================================================

class BinaryDiseaseDataset(
    Dataset
):
    """
    Pneumoconiosis vs Pneumonia用Dataset.

    label:
        0 = Pneumoconiosis
        1 = Pneumonia

    sample:

        {
            "image": Path(...),
            "mask": Path(...) or None,
            "label": 0 or 1
        }
    """

    def __init__(
        self,
        samples,
        transform=None,
        preprocessing_mode="original",
        mask_threshold=128,
        clip_z=3.0,
        eps=1e-8,
    ):
        self.samples = (
            samples
        )

        self.transform = (
            transform
        )

        self.preprocessing_mode = (
            preprocessing_mode
        )

        self.mask_threshold = int(
            mask_threshold
        )

        self.clip_z = (
            None
            if clip_z is None
            else float(
                clip_z
            )
        )

        self.eps = float(
            eps
        )

    def __len__(
        self,
    ):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        sample = (
            self.samples[
                index
            ]
        )

        image_path = Path(
            sample["image"]
        )

        mask_path = sample.get(
            "mask",
            None,
        )

        label = float(
            sample["label"]
        )

        # ====================================================
        # Load image
        # ====================================================

        with Image.open(
            image_path
        ) as image_pil:

            image = (
                image_pil.copy()
            )

        # ====================================================
        # Preprocessing
        # ====================================================
        
        if self.preprocessing_mode == "original":
            # --------------------------------
            # Original
            #
            # Mask: 使用しない
            # Moment: 使用しない
            # --------------------------------
        
            image = image.convert(
                "RGB"
            )
        
        
        elif self.preprocessing_mode== "lung_only":
            # --------------------------------
            # Lung-only
            #
            # Mask: 使用する
            # Moment: 使用しない
            # --------------------------------
        
            if mask_path is None:
                raise RuntimeError(
                    "Lung-only preprocessing "
                    "requires a lung mask: "
                    f"{image_path}"
                )
        
            image = lung_only_image(
                image=image,
                mask_path=mask_path,
                mask_threshold=(
                    self.mask_threshold
                ),
            )
        
        
        elif self.preprocessing_mode == "moment_standardized":
            # --------------------------------
            # Lung-only
            # +
            # Moment standardization
            #
            # Mask: 使用する
            # Moment: 使用する
            # --------------------------------
        
            if mask_path is None:
                raise RuntimeError(
                    "Moment standardization "
                    "requires a lung mask: "
                    f"{image_path}"
                )
        
            image = (
                moment_standardize_image(
                    image=image,
                    mask_path=mask_path,
                    mask_threshold=(
                        self.mask_threshold
                    ),
                    clip_z=(
                        self.clip_z
                    ),
                    eps=self.eps,
                )
            )

        # ====================================================
        # Transform
        # ====================================================

        if self.transform is not None:

            image = (
                self.transform(
                    image
                )
            )

        label = torch.tensor(
            label,
            dtype=torch.float32,
        )

        return (
            image,
            label,
        )


# ============================================================
# Sampling
# ============================================================

def sample_paths(
    paths,
    n_samples,
    seed,
):
    """
    指定数をrandom samplingする．
    """

    paths = list(
        paths
    )

    if (
        len(paths)
        < n_samples
    ):
        raise ValueError(
            f"Requested "
            f"{n_samples} samples, "
            f"but only "
            f"{len(paths)} images "
            f"are available."
        )

    rng = random.Random(
        seed
    )

    selected = rng.sample(
        paths,
        n_samples,
    )

    return sorted(
        selected
    )


# ============================================================
# Samples
# ============================================================

def make_samples(
    image_paths,
    label,
    mask_dir=None,
    require_mask=False,
):
    """
    image path + mask path + label
    をまとめる．
    """

    samples = []

    missing_masks = []

    for image_path in image_paths:

        image_path = Path(
            image_path
        )

        mask_path = None

        if mask_dir is not None:

            mask_path = find_mask(
                mask_root=mask_dir,
                image_stem=(
                    image_path.stem
                ),
            )

        if (
            require_mask
            and mask_path is None
        ):
            missing_masks.append(
                str(
                    image_path
                )
            )

            continue

        samples.append({
            "image":
                image_path,

            "mask":
                mask_path,

            "label":
                int(
                    label
                ),
        })

    # --------------------------------
    # Moment standardizationを使う場合
    # mask不足はエラーにする
    # --------------------------------

    if (
        require_mask
        and missing_masks
    ):

        print(
            f"[WARNING] "
            f"{len(missing_masks)} masks "
            f"were not found."
        )

        for path in (
            missing_masks[:10]
        ):
            print(
                "  [MISSING]",
                path,
            )

        if len(
            missing_masks
        ) > 10:
            print(
                "  ..."
            )

    return samples


# ============================================================
# Stratified train / validation split
# ============================================================

def split_train_val(
    samples,
    val_ratio,
    seed,
):
    """
    1クラス分のsamplesをtrain/valに分割する．

    各クラスごとにこの関数を呼ぶことで
    stratified splitになる．
    """

    samples = list(
        samples
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        samples
    )

    n_total = len(
        samples
    )

    if (
        val_ratio <= 0
    ):
        return (
            samples,
            [],
        )

    n_val = int(
        round(
            n_total
            * val_ratio
        )
    )

    n_val = max(
        1,
        n_val,
    )

    n_val = min(
        n_val,
        n_total - 1,
    )

    val_samples = (
        samples[
            :n_val
        ]
    )

    train_samples = (
        samples[
            n_val:
        ]
    )

    return (
        train_samples,
        val_samples,
    )


# ============================================================
# Transform
# ============================================================

def build_transforms(
    cfg,
):
    transform_cfg = (
        cfg.get(
            "transform",
            {},
        )
    )

    train_cfg = (
        cfg.get(
            "train",
            {},
        )
    )

    resize_size = int(
        transform_cfg.get(
            "resize_size",
            256,
        )
    )

    image_size = int(
        train_cfg.get(
            "img_size",
            224,
        )
    )

    use_random_crop = bool(
        transform_cfg.get(
            "random_crop",
            False,
        )
    )

    horizontal_flip = float(
        transform_cfg.get(
            "horizontal_flip",
            0.5,
        )
    )

    # ========================================================
    # Train
    # ========================================================

    train_steps = []

    if use_random_crop:

        train_steps.append(
            transforms.Resize(
                (
                    resize_size,
                    resize_size,
                )
            )
        )

        train_steps.append(
            transforms.RandomCrop(
                image_size
            )
        )

    else:

        train_steps.append(
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            )
        )

    if horizontal_flip > 0:

        train_steps.append(
            transforms.RandomHorizontalFlip(
                p=horizontal_flip
            )
        )

    train_steps.extend([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    train_transform = (
        transforms.Compose(
            train_steps
        )
    )

    # ========================================================
    # Validation / Test
    # ========================================================

    if use_random_crop:

        eval_transform = (
            transforms.Compose([
                transforms.Resize(
                    (
                        resize_size,
                        resize_size,
                    )
                ),

                transforms.CenterCrop(
                    image_size
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])
        )

    else:

        eval_transform = (
            transforms.Compose([
                transforms.Resize(
                    (
                        image_size,
                        image_size,
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])
        )

    return (
        train_transform,
        eval_transform,
    )

def filter_images_with_masks(
    image_paths,
    mask_dir,
):
    valid_paths = []
    missing_paths = []

    for image_path in image_paths:

        image_path = Path(
            image_path
        )

        mask_path = find_mask(
            mask_root=mask_dir,
            image_stem=image_path.stem,
        )

        if mask_path is None:
            missing_paths.append(
                image_path
            )

        else:
            valid_paths.append(
                image_path
            )

    print(
        "[INFO] Mask availability: "
        f"{len(valid_paths)} / "
        f"{len(image_paths)}"
    )

    if missing_paths:

        print(
            "[WARNING] Missing masks:",
            len(
                missing_paths
            ),
        )

        for path in (
            missing_paths[:10]
        ):
            print(
                "  [MISSING]",
                path.name,
            )

    return valid_paths


# ============================================================
# Build datasets
# ============================================================

def build_binary_disease_datasets(
    cfg,
    seed
):
    dataset_cfg = (
        cfg["dataset"]
    )

    train_cfg = (
        cfg.get(
            "train",
            {},
        )
    )

    preprocessing_cfg = cfg.get(
        "preprocessing",
        {},
    )
    
    preprocessing_mode = str(
        preprocessing_cfg.get(
            "mode",
            "original",
        )
    ).lower()
    
    valid_modes = {
        "original",
        "lung_only",
        "moment_standardized",
    }
    
    if preprocessing_mode not in valid_modes:
        raise ValueError(
            "Unknown preprocessing mode: "
            f"{preprocessing_mode}"
        )
    
    # maskが画像処理に必要か
    needs_mask = (
        preprocessing_mode
        in {
            "lung_only",
            "moment_standardized",
        }
    )
    
    mask_threshold = int(
        preprocessing_cfg.get(
            "mask_threshold",
            128,
        )
    )
    
    clip_z = preprocessing_cfg.get(
        "clip_z",
        3.0,
    )
    
    if clip_z is not None:
        clip_z = float(
            clip_z
        )
    
    eps = float(
        preprocessing_cfg.get(
            "eps",
            1e-8,
        )
    )
    
    require_mask_for_comparison = bool(
        preprocessing_cfg.get(
            "require_mask_for_comparison",
            True,
        )
    )

    train_samples_per_class = int(
        dataset_cfg.get(
            "train_samples_per_class",
            70,
        )
    )

    test_samples_per_class = int(
        dataset_cfg.get(
            "test_samples_per_class",
            21,
        )
    )

    val_ratio = float(
        dataset_cfg.get(
            "val_ratio",
            0.2,
        )
    )

    # ========================================================
    # Paths
    # ========================================================

    pneumo_cfg = (
        dataset_cfg[
            "pneumoconiosis"
        ]
    )

    pneumonia_cfg = (
        dataset_cfg[
            "pneumonia"
        ]
    )

    # --------------------------------
    # Images
    # --------------------------------

    pneumo_train_paths = find_images(
        pneumo_cfg[
            "train"
        ]
    )

    pneumo_test_paths = find_images(
        pneumo_cfg[
            "test"
        ]
    )

    pneumonia_train_paths = find_images(
        pneumonia_cfg[
            "train"
        ]
    )

    pneumonia_test_paths = find_images(
        pneumonia_cfg[
            "test"
        ]
    )

    print(
        "\n[INFO] Available images"
    )

    print(
        "Pneumoconiosis train:",
        len(
            pneumo_train_paths
        ),
    )

    print(
        "Pneumoconiosis test:",
        len(
            pneumo_test_paths
        ),
    )

    print(
        "Pneumonia train:",
        len(
            pneumonia_train_paths
        ),
    )

    print(
        "Pneumonia test:",
        len(
            pneumonia_test_paths
        ),
    )

    # ========================================================
    # Sample same number from each class
    # ========================================================

    pneumo_train_mask_dir = (
        pneumo_cfg.get(
            "train_masks",
            None,
        )
    )
    
    pneumo_test_mask_dir = (
        pneumo_cfg.get(
            "test_masks",
            None,
        )
    )
    
    pneumonia_train_mask_dir = (
        pneumonia_cfg.get(
            "train_masks",
            None,
        )
    )
    
    pneumonia_test_mask_dir = (
        pneumonia_cfg.get(
            "test_masks",
            None,
        )
    )
    
    if require_mask_for_comparison:
        pneumo_train_paths = (
            filter_images_with_masks(
                pneumo_train_paths,
                pneumo_train_mask_dir,
            )
        )
    
        pneumo_test_paths = (
            filter_images_with_masks(
                pneumo_test_paths,
                pneumo_test_mask_dir,
            )
        )
    
        pneumonia_train_paths = (
            filter_images_with_masks(
                pneumonia_train_paths,
                pneumonia_train_mask_dir,
            )
        )
    
        pneumonia_test_paths = (
            filter_images_with_masks(
                pneumonia_test_paths,
                pneumonia_test_mask_dir,
            )
        )

    pneumo_train_selected = (
        sample_paths(
            pneumo_train_paths,
            train_samples_per_class,
            seed=seed,
        )
    )

    pneumonia_train_selected = (
        sample_paths(
            pneumonia_train_paths,
            train_samples_per_class,
            seed=seed,
        )
    )

    pneumo_test_selected = (
        sample_paths(
            pneumo_test_paths,
            test_samples_per_class,
            seed=seed,
        )
    )

    pneumonia_test_selected = (
        sample_paths(
            pneumonia_test_paths,
            test_samples_per_class,
            seed=seed,
        )
    )

    # ========================================================
    # Mask directories
    # ========================================================

    pneumo_train_mask_dir = (
        pneumo_cfg.get(
            "train_masks",
            None,
        )
    )

    pneumo_test_mask_dir = (
        pneumo_cfg.get(
            "test_masks",
            None,
        )
    )

    pneumonia_train_mask_dir = (
        pneumonia_cfg.get(
            "train_masks",
            None,
        )
    )

    pneumonia_test_mask_dir = (
        pneumonia_cfg.get(
            "test_masks",
            None,
        )
    )

    # ========================================================
    # Make samples
    #
    # 0 = Pneumoconiosis
    # 1 = Pneumonia
    # ========================================================

    pneumo_train_samples = (
        make_samples(
            image_paths=pneumo_train_selected, 
            label=0,
            mask_dir=pneumo_train_mask_dir,
            require_mask=needs_mask,
        )
    )

    pneumonia_train_samples = (
        make_samples(
            image_paths=pneumonia_train_selected,
            label=1,
            mask_dir=pneumonia_train_mask_dir,
            require_mask = needs_mask,
        )
    )

    pneumo_test_samples = (
        make_samples(
            image_paths=pneumo_test_selected,
            label=0,
            mask_dir=pneumo_test_mask_dir,
            require_mask = needs_mask
        )
    )

    pneumonia_test_samples = (
        make_samples(
            image_paths=pneumonia_test_selected,
            label=1,
            mask_dir=pneumonia_test_mask_dir,
            require_mask=needs_mask
        )
    )

    # --------------------------------
    # Mask missing check
    # --------------------------------

    if (needs_mask):

        if (
            len(
                pneumo_train_samples
            )
            != train_samples_per_class
        ):
            raise RuntimeError(
                "Some Pneumoconiosis "
                "training masks are missing."
            )

        if (
            len(
                pneumonia_train_samples
            )
            != train_samples_per_class
        ):
            raise RuntimeError(
                "Some Pneumonia "
                "training masks are missing."
            )

        if (
            len(
                pneumo_test_samples
            )
            != test_samples_per_class
        ):
            raise RuntimeError(
                "Some Pneumoconiosis "
                "test masks are missing."
            )

        if (
            len(
                pneumonia_test_samples
            )
            != test_samples_per_class
        ):
            raise RuntimeError(
                "Some Pneumonia "
                "test masks are missing."
            )

    # ========================================================
    # Stratified train / validation
    # ========================================================

    (
        pneumo_train,
        pneumo_val,
    ) = split_train_val(
        samples=(
            pneumo_train_samples
        ),
        val_ratio=(
            val_ratio
        ),
        seed=seed,
    )

    (
        pneumonia_train,
        pneumonia_val,
    ) = split_train_val(
        samples=(
            pneumonia_train_samples
        ),
        val_ratio=(
            val_ratio
        ),
        seed=seed + 1,
    )

    # ========================================================
    # Combine
    # ========================================================

    train_samples = (
        pneumo_train
        + pneumonia_train
    )

    val_samples = (
        pneumo_val
        + pneumonia_val
    )

    test_samples = (
        pneumo_test_samples
        + pneumonia_test_samples
    )

    # --------------------------------
    # Shuffle train / val / test
    # --------------------------------

    random.Random(
        seed
    ).shuffle(
        train_samples
    )

    random.Random(
        seed + 1
    ).shuffle(
        val_samples
    )

    random.Random(
        seed + 2
    ).shuffle(
        test_samples
    )

    # ========================================================
    # Transform
    # ========================================================

    (
        train_transform,
        eval_transform,
    ) = build_transforms(
        cfg
    )

    # ========================================================
    # Dataset
    # ========================================================

    train_dataset = (
        BinaryDiseaseDataset(
            samples=train_samples,
            transform=train_transform,
            preprocessing_mode=preprocessing_mode,
            mask_threshold=mask_threshold,
            clip_z=clip_z,
            eps=eps,
        )
    )

    val_dataset = (
        BinaryDiseaseDataset(
            samples=val_samples,
            transform=eval_transform,
            preprocessing_mode=preprocessing_mode,
            mask_threshold=mask_threshold,
            clip_z=clip_z,
            eps=eps,
        )
    )

    test_dataset = (
        BinaryDiseaseDataset(
            samples=test_samples,
            transform=eval_transform,
            preprocessing_mode=preprocessing_mode, 
            mask_threshold=mask_threshold,
            clip_z=clip_z,
            eps=eps,
        )
    )

    # ========================================================
    # Information
    # ========================================================

    print(
        "\n[INFO] Dataset split"
    )

    print(
        "Train:",
        len(
            train_dataset
        ),
    )

    print(
        "Validation:",
        len(
            val_dataset
        ),
    )

    print(
        "Test:",
        len(
            test_dataset
        ),
    )

    print(
        "\n[INFO] Preprocessing mode:",
        preprocessing_mode,
    )
    
    print(
        "[INFO] Uses lung mask:",
        needs_mask
    )
    
    print(
        "[INFO] Uses moment standardization:",
        preprocessing_mode
        == "moment_standardized",
    )

    datasets_dict = {
        "train":
            train_dataset,

        "val":
            val_dataset,

        "test":
            test_dataset,

        "class_names": [
            "Pneumoconiosis",
            "Pneumonia",
        ],

        "class_to_idx": {
            "Pneumoconiosis": 0,
            "Pneumonia": 1,
        },

        "counts": {
            "train": {
                "Pneumoconiosis":
                    len(
                        pneumo_train
                    ),

                "Pneumonia":
                    len(
                        pneumonia_train
                    ),
            },

            "val": {
                "Pneumoconiosis":
                    len(
                        pneumo_val
                    ),

                "Pneumonia":
                    len(
                        pneumonia_val
                    ),
            },

            "test": {
                "Pneumoconiosis":
                    len(
                        pneumo_test_samples
                    ),

                "Pneumonia":
                    len(
                        pneumonia_test_samples
                    ),
            },
        },

        # 再現性確認用
        "train_samples":
            train_samples,

        "val_samples":
            val_samples,

        "test_samples":
            test_samples,
    }

    return datasets_dict


# ============================================================
# DataLoader
# ============================================================

def build_binary_disease_loaders(
    datasets_dict,
    cfg,
):
    train_cfg = (
        cfg.get(
            "train",
            {},
        )
    )

    batch_size = int(
        train_cfg.get(
            "batch_size",
            8,
        )
    )

    num_workers = int(
        train_cfg.get(
            "num_workers",
            0,
        )
    )

    pin_memory = bool(
        train_cfg.get(
            "pin_memory",
            True,
        )
    )

    train_loader = DataLoader(
        datasets_dict[
            "train"
        ],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        datasets_dict[
            "val"
        ],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        datasets_dict[
            "test"
        ],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train":
            train_loader,

        "val":
            val_loader,

        "test":
            test_loader,
    }