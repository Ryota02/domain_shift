from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def build_datasets(cfg):
    workdir = cfg["workdir"]
    img_size = cfg["train"]["img_size"]

    train_transform, eval_transform = build_transforms(img_size)

    source_train_datasets = []
    source_val_datasets = []

    reference_class_to_idx = None

    for source in cfg["sources"]:
        source_dir = workdir / source["path"]

        train_dataset = datasets.ImageFolder(
            root=source_dir / "train",
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=source_dir / "test",
            transform=eval_transform
        )

        if reference_class_to_idx is None:
            reference_class_to_idx = train_dataset.class_to_idx
        else:
            assert train_dataset.class_to_idx == reference_class_to_idx, (
                f"class_to_idx mismatch in {source['name']}: "
                f"{train_dataset.class_to_idx} != {reference_class_to_idx}"
            )

        source_train_datasets.append(train_dataset)
        source_val_datasets.append(val_dataset)

    target_dir = workdir / cfg["target"]["path"]

    target_adapt_dataset = datasets.ImageFolder(
        root=target_dir / "train",
        transform=train_transform
    )

    target_test_dataset = datasets.ImageFolder(
        root=target_dir / "test",
        transform=eval_transform
    )

    assert target_test_dataset.class_to_idx == reference_class_to_idx, (
        f"Target class_to_idx mismatch: "
        f"{target_test_dataset.class_to_idx} != {reference_class_to_idx}"
    )

    source_train_dataset = ConcatDataset(source_train_datasets)
    source_val_dataset = ConcatDataset(source_val_datasets)

    return {
        "source_train": source_train_dataset,
        "source_val": source_val_dataset,
        "target_adapt": target_adapt_dataset,
        "target_test": target_test_dataset,
        "target_classes": target_test_dataset.classes,
        "target_class_to_idx": target_test_dataset.class_to_idx,
    }


def build_loaders(cfg, datasets_dict):
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"]["num_workers"]

    loaders = {
        "source_train": DataLoader(
            datasets_dict["source_train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "source_val": DataLoader(
            datasets_dict["source_val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "target_adapt": DataLoader(
            datasets_dict["target_adapt"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "target_test": DataLoader(
            datasets_dict["target_test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    return loaders