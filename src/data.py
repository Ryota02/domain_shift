from pathlib import Path

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def check_class_mapping(
    reference_dataset,
    compared_dataset,
    reference_name,
    compared_name,
):
    """
    2つのImageFolderでclass_to_idxが一致するか確認する．
    """
    reference_mapping = (
        reference_dataset.class_to_idx
    )

    compared_mapping = (
        compared_dataset.class_to_idx
    )

    if reference_mapping != compared_mapping:
        raise RuntimeError(
            f"{reference_name}/{compared_name} "
            "class_to_idx mismatch:\n"
            f"{reference_name}="
            f"{reference_mapping}\n"
            f"{compared_name}="
            f"{compared_mapping}"
        )


def build_transforms(
    img_size, 
    resize_size=None, 
    random_crop=False
):
    if resize_size is None:
        resize_size = img_size

    train_ops = [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((resize_size, resize_size)),
    ]
    eval_ops = [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((resize_size, resize_size)),
    ]

    if random_crop:
        train_ops.append(
            transforms.RandomCrop((img_size, img_size))
        )
        eval_ops.append(
            transforms.CenterCrop((img_size, img_size))
        )

    train_ops.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])
    eval_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return (
        transforms.Compose(train_ops),
        transforms.Compose(eval_ops),
    )

def build_chestxray8_datasets(
    cfg,
    train_transform,
    eval_transform,
):
    """
    想定構成：
    root/
    ├── source/train/<class>/*.png
    └── target/test/<class>/*.png

    target/testは，
    ・target_adapt：学習用Transform
    ・target_test：評価用Transform
    として2回読み込む．
    """
    workdir = Path(cfg["workdir"])

    source_train_dir = workdir / cfg.get(
        "sources",
        "ChestXray8/images/source",
    )
    target_test_dir = workdir / cfg.get(
        "target",
        "ChestXray8/images/target",
    )

    source_train_dataset = datasets.ImageFolder(
        root=source_train_dir,
        transform=train_transform,
    )
    # 同じSource train画像へ評価用Transformを適用する
    source_tsne_dataset = datasets.ImageFolder(
        root=source_train_dir,
        transform=eval_transform,
    )
    
    check_class_mapping(
        reference_dataset=source_train_dataset,
        compared_dataset=source_tsne_dataset,
        reference_name="ChestXray8 source train",
        compared_name="ChestXray8 source t-SNE",
    )
    target_adapt_dataset = datasets.ImageFolder(
        root=target_test_dir,
        transform=train_transform,
    )
    target_test_dataset = datasets.ImageFolder(
        root=target_test_dir,
        transform=eval_transform,
    )

    if (
        source_train_dataset.class_to_idx
        != target_test_dataset.class_to_idx
    ):
        raise RuntimeError(
            "Source/Target class_to_idx mismatch:\n"
            f"source={source_train_dataset.class_to_idx}\n"
            f"target={target_test_dataset.class_to_idx}"
        )

    result = {
        "source_train": source_train_dataset,
        "source_tsne": source_tsne_dataset,
        "target_adapt": target_adapt_dataset,
        "target_test": target_test_dataset,
        "target_classes": target_test_dataset.classes,
        "target_class_to_idx": (
            target_test_dataset.class_to_idx
        ),
    }

    # source_val = cfg.get("source_val", {})
    # if source_val is not None:
    #     source_val_dir = workdir / source_val

    #     if source_val_dir.exists():
    #         source_val_dataset = datasets.ImageFolder(
    #             root=source_val_dir,
    #             transform=eval_transform,
    #         )

    #         if (
    #             source_val_dataset.class_to_idx
    #             != source_train_dataset.class_to_idx
    #         ):
    #             raise RuntimeError(
    #                 "Source train/val class_to_idx mismatch."
    #             )

    #         result["source_val"] = source_val_dataset

    return result

def build_datasets(cfg):
    workdir = cfg["workdir"]
    img_size = cfg["train"]["img_size"]

    transform_cfg = cfg.get("transform")
    train_transform, eval_transform = build_transforms(
        img_size=img_size,
        resize_size=transform_cfg.get("resize_size"), 
        random_crop=transform_cfg.get(
            "random_crop",
            False
        )
    )

    dataset_type = cfg.get("dataset_type")

    if dataset_type == "chestxray8": 
        return build_chestxray8_datasets(
            cfg,
            train_transform,
            eval_transform,
        )
    
    source_train_datasets = []
    source_val_datasets = []
    source_tsne_datasets = []

    reference_class_to_idx = None

    for source in cfg["sources"]:
        source_dir = workdir / source["path"]

        train_dataset = datasets.ImageFolder(
            root=source_dir / "train",
            transform=train_transform
        )

        source_tsne_dataset = datasets.ImageFolder(
            root=source_train_dir,
            transform=eval_transform,
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
        source_tsne_datasets.append(source_tsne_dataset)
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
    source_tsne_dataset = ConcatDataset(source_tsne_datasets)
    source_val_dataset = ConcatDataset(source_val_datasets)

    return {
        "source_train": source_train_dataset,
        "source_val": source_val_dataset,
        "source_tsne": source_tsne_dataset,
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

    if "source_val" in datasets_dict:
        loaders["source_val"] = DataLoader(
            datasets_dict["source_val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),

        print(
            "[INFO] Source validation loader:",
            len(datasets_dict["source_val"]),
            "images",
        )
    else:
        print(
            "[INFO] Source validation dataset is not used."
        )

    print(
        "[INFO] Loader keys:",
        list(loaders.keys()),
    )

    return loaders