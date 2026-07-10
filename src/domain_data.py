from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


class DomainImageFolder(Dataset):
    """
    ImageFolderにdomain labelを付けるDataset。
    疾患ラベルは読み込むが，domain分類の学習では使わない。
    """

    def __init__(self, root, transform, domain_label):
        self.dataset = datasets.ImageFolder(
            root=str(root),
            transform=transform,
        )
        self.domain_label = domain_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, disease_label = self.dataset[idx]
        return image, self.domain_label, disease_label


class MultiDomainImageDataset(Dataset):
    """
    複数domainのDatasetを1つにまとめる。
    """

    def __init__(self, domain_datasets):
        self.domain_datasets = domain_datasets
        self.index_map = []

        for domain_dataset_idx, dataset in enumerate(domain_datasets):
            for sample_idx in range(len(dataset)):
                self.index_map.append((domain_dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        domain_dataset_idx, sample_idx = self.index_map[idx]
        return self.domain_datasets[domain_dataset_idx][sample_idx]


def get_domain_path(workdir, domain_name, split):
    workdir = Path(workdir)

    if domain_name == "china":
        return workdir / "ZhangLabData_binary_dataset" / split

    if domain_name == "doha":
        return workdir / "COVID-19_Radiography_binary_dataset_clean" / split

    if domain_name == "nigeria":
        return workdir / "nigerian_pneumonia_binary_dataset" / split

    raise ValueError(f"Unknown domain: {domain_name}")


def build_domain_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_domain_loader(
    workdir,
    domains,
    split,
    img_size,
    batch_size,
    num_workers,
    shuffle,
):
    transform = build_domain_transform(img_size)

    domain_datasets = []

    for domain_label, domain_name in enumerate(domains):
        root = get_domain_path(
            workdir=workdir,
            domain_name=domain_name,
            split=split,
        )

        print(
            f"[INFO] domain={domain_name}, "
            f"label={domain_label}, "
            f"split={split}, "
            f"path={root}"
        )

        dataset = DomainImageFolder(
            root=root,
            transform=transform,
            domain_label=domain_label,
        )

        print(f"[INFO] num images: {len(dataset)}")

        domain_datasets.append(dataset)

    merged_dataset = MultiDomainImageDataset(domain_datasets)

    loader = DataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def build_train_val_test_loaders(
    workdir,
    domains,
    img_size,
    batch_size,
    num_workers,
    val_ratio,
    split_train="train",
    split_test="test",
    seed=42,
):
    """
    各domainのtrainフォルダをtrain/valに分割し，
    testフォルダを最終評価に使う。
    """

    transform = build_domain_transform(img_size)

    train_domain_datasets = []
    val_domain_datasets = []

    generator = torch.Generator().manual_seed(seed)

    for domain_label, domain_name in enumerate(domains):
        root = get_domain_path(
            workdir=workdir,
            domain_name=domain_name,
            split=split_train,
        )

        print(
            f"[INFO] domain={domain_name}, "
            f"label={domain_label}, "
            f"split={split_train}, "
            f"path={root}"
        )

        full_dataset = DomainImageFolder(
            root=root,
            transform=transform,
            domain_label=domain_label,
        )

        n_total = len(full_dataset)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        if n_train <= 0 or n_val <= 0:
            raise ValueError(
                f"Invalid train/val split for {domain_name}: "
                f"total={n_total}, train={n_train}, val={n_val}"
            )

        train_subset, val_subset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=generator,
        )

        print(
            f"[INFO] {domain_name}: "
            f"total={n_total}, train={n_train}, val={n_val}"
        )

        train_domain_datasets.append(train_subset)
        val_domain_datasets.append(val_subset)

    train_dataset = MultiDomainImageDataset(train_domain_datasets)
    val_dataset = MultiDomainImageDataset(val_domain_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = build_domain_loader(
        workdir=workdir,
        domains=domains,
        split=split_test,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader