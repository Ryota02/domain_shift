from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def build_datasets(cfg):
    train_transform, eval_transform = build_transforms(cfg.img_size)

    doha_dir = cfg.workdir / cfg.doha_name
    nigeria_dir = cfg.workdir / cfg.nigeria_name
    china_dir = cfg.workdir / cfg.china_name

    doha_train = datasets.ImageFolder(doha_dir / "train", transform=train_transform)
    china_train = datasets.ImageFolder(china_dir / "train", transform=train_transform)

    doha_val = datasets.ImageFolder(doha_dir / "test", transform=eval_transform)
    china_val = datasets.ImageFolder(china_dir / "test", transform=eval_transform)

    target_adapt = datasets.ImageFolder(nigeria_dir / "train", transform=train_transform)
    target_test = datasets.ImageFolder(nigeria_dir / "test", transform=eval_transform)

    assert doha_train.class_to_idx == china_train.class_to_idx, f"Doha and China class_to_idx mismatch: {         oha_train.class_to_idx}, {china_train.class_to_idx}"

    assert doha_train.class_to_idx == target_test.class_to_idx, f"Source and Target class_to_idx mismatch: {doha_train.class_to_idx}, {target_test.class_to_idx}"

    source_train = ConcatDataset([doha_train, china_train])
    source_val = ConcatDataset([doha_val, china_val])

    return {
        "doha_train": doha_train,
        "china_train": china_train,
        "doha_val": doha_val,
        "china_val": china_val,
        "source_train": source_train,
        "source_val": source_val,
        "target_adapt": target_adapt,
        "target_test": target_test,
    }


def build_loaders(cfg, datasets_dict):
    loaders = {
        "source_train": DataLoader(
            datasets_dict["source_train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True
        ),
        "source_val": DataLoader(
            datasets_dict["source_val"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        ),
        "target_adapt": DataLoader(
            datasets_dict["target_adapt"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True
        ),
        "target_test": DataLoader(
            datasets_dict["target_test"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        ),
    }

    return loaders