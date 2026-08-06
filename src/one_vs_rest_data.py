from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from src.data import build_transforms


class OneVsRestDataset(Dataset):
    """ImageFolderの12クラスを，指定クラス対その他へ変換する．"""

    def __init__(self, root, target_class, transform=None):
        self.dataset = datasets.ImageFolder(
            root=Path(root),
            transform=transform,
        )

        if target_class not in self.dataset.class_to_idx:
            raise ValueError(
                f"Unknown target class: {target_class}\n"
                f"Available classes: {self.dataset.classes}"
            )

        self.target_class = target_class
        self.target_class_idx = self.dataset.class_to_idx[target_class]
        self.original_classes = self.dataset.classes
        self.original_class_to_idx = self.dataset.class_to_idx
        self.classes = [f"Not_{target_class}", target_class]
        self.class_to_idx = {
            f"Not_{target_class}": 0,
            target_class: 1,
        }

        self.binary_targets = np.asarray(
            [
                int(label == self.target_class_idx)
                for label in self.dataset.targets
            ],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, original_label = self.dataset[index]
        binary_label = int(original_label == self.target_class_idx)
        return image, binary_label


def _require_directory(path, name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{name} directory not found: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")

    return path


def _check_mapping(reference, compared, reference_name, compared_name):
    if reference.original_class_to_idx != compared.original_class_to_idx:
        raise RuntimeError(
            f"{reference_name}/{compared_name} class mapping mismatch:\n"
            f"{reference_name}={reference.original_class_to_idx}\n"
            f"{compared_name}={compared.original_class_to_idx}"
        )


def _count_targets(dataset):
    if isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices)
        targets = dataset.dataset.binary_targets[indices]
    else:
        targets = dataset.binary_targets

    positive = int(np.sum(targets == 1))
    negative = int(np.sum(targets == 0))

    return {
        "positive": positive,
        "negative": negative,
        "total": positive + negative,
    }


def build_one_vs_rest_datasets(cfg, target_class):
    transform_cfg = cfg["transform"]
    dataset_cfg = cfg["dataset"]

    root = Path(dataset_cfg["root"])
    target_class = target_class

    train_dir = _require_directory(
        root / dataset_cfg.get("train", "source/train"),
        "Train",
    )
    test_dir = _require_directory(
        root / dataset_cfg.get("test", "target/test"),
        "Test",
    )

    img_size = cfg.get("train", {}).get("img_size", 224)
    resize_size = transform_cfg.get("resize_size", 256)
    random_crop = transform_cfg.get("random_crop", True)
    train_transform, eval_transform = build_transforms(
        img_size=img_size, 
        resize_size=resize_size, 
        random_crop=random_crop
    )

    train_augmented = OneVsRestDataset(
        train_dir,
        target_class,
        train_transform,
    )
    train_evaluation = OneVsRestDataset(
        train_dir,
        target_class,
        eval_transform,
    )
    test_dataset = OneVsRestDataset(
        test_dir,
        target_class,
        eval_transform,
    )

    _check_mapping(
        train_augmented,
        test_dataset,
        "Train",
        "Test",
    )

    val_relative = dataset_cfg.get("val")

    if val_relative:
        val_dir = _require_directory(
            root / val_relative,
            "Validation",
        )
        val_dataset = OneVsRestDataset(
            val_dir,
            target_class,
            eval_transform,
        )
        _check_mapping(
            train_augmented,
            val_dataset,
            "Train",
            "Validation",
        )
        train_dataset = train_augmented
    else:
        val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
        seed = int(cfg["train"].get("seed", 42))

        if not 0.0 < val_ratio < 1.0:
            raise ValueError("dataset.val_ratio must be between 0 and 1.")

        indices = np.arange(len(train_augmented))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=train_augmented.binary_targets,
        )

        train_dataset = Subset(
            train_augmented,
            train_indices.tolist(),
        )
        val_dataset = Subset(
            train_evaluation,
            val_indices.tolist(),
        )

    result = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "class_names": test_dataset.classes,
        "class_to_idx": test_dataset.class_to_idx,
        "original_classes": test_dataset.original_classes,
        "original_class_to_idx": test_dataset.original_class_to_idx,
        "target_class": target_class,
        "counts": {
            "train": _count_targets(train_dataset),
            "val": _count_targets(val_dataset),
            "test": _count_targets(test_dataset),
        },
    }

    print("[INFO] One-vs-Rest target:", target_class)
    print("[INFO] Binary classes:", result["class_names"])
    print("[INFO] Counts:", result["counts"])

    return result


def build_one_vs_rest_loaders(cfg, datasets_dict):
    train_cfg = cfg["train"]

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", True))

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    return {
        "train": DataLoader(
            datasets_dict["train"],
            shuffle=True,
            drop_last=False,
            **common,
        ),
        "val": DataLoader(
            datasets_dict["val"],
            shuffle=False,
            drop_last=False,
            **common,
        ),
        "test": DataLoader(
            datasets_dict["test"],
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }


def calculate_pos_weight(datasets_dict):
    counts = datasets_dict["counts"]["train"]
    positive = counts["positive"]
    negative = counts["negative"]

    if positive == 0:
        raise ValueError("Training data contains no positive samples.")

    return float(negative / positive)
