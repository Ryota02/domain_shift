from pathlib import Path
from PIL import Image

import numpy as np

from sklearn.model_selection import (
    train_test_split,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)

from src.data import build_transforms


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


class BinaryDiseaseDataset(Dataset):
    def __init__(
        self,
        samples,
        transform=None,
    ):
        """
        samples:
            [
                (image_path, label),
                ...
            ]

        label:
            Pneumoconiosis -> 0
            Pneumonia      -> 1
        """
        self.samples = samples
        self.transform = transform

        self.classes = [
            "Pneumoconiosis",
            "Pneumonia",
        ]

        self.class_to_idx = {
            "Pneumoconiosis": 0,
            "Pneumonia": 1,
        }

        self.targets = [
            label
            for _, label in samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(
                image
            )

        return image, label


def find_images(directory):
    """
    指定されたディレクトリ以下の画像を
    再帰的に取得する．
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )

    images = sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not images:
        raise RuntimeError(
            f"No images found in: "
            f"{directory}"
        )

    return images


def sample_images(
    image_paths,
    n_samples,
    seed,
):
    """
    n_samples枚を重複なしでランダム抽出する．
    """
    if len(image_paths) < n_samples:
        raise ValueError(
            f"Requested {n_samples} images, "
            f"but only {len(image_paths)} "
            "images are available."
        )

    rng = np.random.default_rng(
        seed
    )

    selected_indices = rng.choice(
        len(image_paths),
        size=n_samples,
        replace=False,
    )

    return [
        image_paths[index]
        for index in selected_indices
    ]


def make_samples(
    pneumoconiosis_paths,
    pneumonia_paths,
):
    """
    Pneumoconiosis -> 0
    Pneumonia      -> 1
    """
    samples = []

    samples.extend(
        [
            (path, 0)
            for path
            in pneumoconiosis_paths
        ]
    )

    samples.extend(
        [
            (path, 1)
            for path
            in pneumonia_paths
        ]
    )

    return samples


def build_binary_disease_datasets(
    cfg,
):
    dataset_cfg = cfg["dataset"]

    seed = int(
        cfg["train"].get(
            "seed",
            42,
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

    # -------------------------
    # Paths
    # -------------------------

    pneumoconiosis_train = (
        find_images(
            dataset_cfg[
                "pneumoconiosis"
            ]["train"]
        )
    )

    pneumoconiosis_test = (
        find_images(
            dataset_cfg[
                "pneumoconiosis"
            ]["test"]
        )
    )

    pneumonia_train = (
        find_images(
            dataset_cfg[
                "pneumonia"
            ]["train"]
        )
    )

    pneumonia_test = (
        find_images(
            dataset_cfg[
                "pneumonia"
            ]["test"]
        )
    )

    print(
        "[INFO] Available data:"
    )

    print(
        "Pneumoconiosis train:",
        len(pneumoconiosis_train),
    )

    print(
        "Pneumoconiosis test:",
        len(pneumoconiosis_test),
    )

    print(
        "Pneumonia train:",
        len(pneumonia_train),
    )

    print(
        "Pneumonia test:",
        len(pneumonia_test),
    )

    # -------------------------
    # Sampling
    # -------------------------

    pneumoconiosis_train = (
        sample_images(
            pneumoconiosis_train,
            train_samples_per_class,
            seed,
        )
    )

    pneumonia_train = (
        sample_images(
            pneumonia_train,
            train_samples_per_class,
            seed + 1,
        )
    )

    pneumoconiosis_test = (
        sample_images(
            pneumoconiosis_test,
            test_samples_per_class,
            seed + 2,
        )
    )

    pneumonia_test = (
        sample_images(
            pneumonia_test,
            test_samples_per_class,
            seed + 3,
        )
    )

    # -------------------------
    # Train/Validation split
    # -------------------------

    all_train_samples = (
        make_samples(
            pneumoconiosis_train,
            pneumonia_train,
        )
    )

    labels = np.array(
        [
            label
            for _, label
            in all_train_samples
        ]
    )

    indices = np.arange(
        len(all_train_samples)
    )

    train_indices, val_indices = (
        train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
    )

    # -------------------------
    # Transform
    # -------------------------

    img_size = cfg.get("train", {}).get("img_size", 224)
    resize_size = cfg.get("transform", {}).get("resize_size", 256)
    random_crop = cfg.get("transform", {}).get("random_crop", True)
    
    train_transform, eval_transform = build_transforms(
        img_size=img_size, 
        resize_size=resize_size, 
        random_crop=random_crop
    )

    # trainとvalで同じ画像リストを持つが，
    # Transformは別々にする
    train_base = BinaryDiseaseDataset(
        all_train_samples,
        transform=train_transform,
    )

    val_base = BinaryDiseaseDataset(
        all_train_samples,
        transform=eval_transform,
    )

    train_dataset = Subset(
        train_base,
        train_indices.tolist(),
    )

    val_dataset = Subset(
        val_base,
        val_indices.tolist(),
    )

    # -------------------------
    # Test
    # -------------------------

    test_samples = make_samples(
        pneumoconiosis_test,
        pneumonia_test,
    )

    test_dataset = (
        BinaryDiseaseDataset(
            test_samples,
            transform=eval_transform,
        )
    )

    # -------------------------
    # Counts
    # -------------------------

    train_labels = labels[
        train_indices
    ]

    val_labels = labels[
        val_indices
    ]

    test_labels = np.array(
        test_dataset.targets
    )

    counts = {
        "train": {
            "Pneumoconiosis": int(
                np.sum(
                    train_labels == 0
                )
            ),
            "Pneumonia": int(
                np.sum(
                    train_labels == 1
                )
            ),
        },
        "val": {
            "Pneumoconiosis": int(
                np.sum(
                    val_labels == 0
                )
            ),
            "Pneumonia": int(
                np.sum(
                    val_labels == 1
                )
            ),
        },
        "test": {
            "Pneumoconiosis": int(
                np.sum(
                    test_labels == 0
                )
            ),
            "Pneumonia": int(
                np.sum(
                    test_labels == 1
                )
            ),
        },
    }

    print(
        "[INFO] Class mapping:"
    )

    print(
        {
            "Pneumoconiosis": 0,
            "Pneumonia": 1,
        }
    )

    print(
        "[INFO] Dataset counts:"
    )

    print(counts)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "class_names": [
            "Pneumoconiosis",
            "Pneumonia",
        ],
        "class_to_idx": {
            "Pneumoconiosis": 0,
            "Pneumonia": 1,
        },
        "counts": counts,
    }


def build_binary_disease_loaders(
    cfg,
    datasets_dict,
):
    train_cfg = cfg["train"]

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

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": (
            num_workers > 0
        ),
    }

    return {
        "train": DataLoader(
            datasets_dict["train"],
            shuffle=True,
            **common,
        ),

        "val": DataLoader(
            datasets_dict["val"],
            shuffle=False,
            **common,
        ),

        "test": DataLoader(
            datasets_dict["test"],
            shuffle=False,
            **common,
        ),
    }