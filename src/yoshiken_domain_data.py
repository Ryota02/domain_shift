from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


class FlatDomainImageDataset(Dataset):
    """
    画像が直下，またはサブフォルダ内に置かれているデータセット用。

    Domain分類では疾患ラベルは使わないため，
    disease_label は -1 とする。
    """

    def __init__(self, root, transform, domain_label):
        self.root = Path(root)
        self.transform = transform
        self.domain_label = domain_label

        self.image_paths = []

        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                self.image_paths.append(path)

        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        disease_label = -1

        return image, self.domain_label, disease_label


class MultiDomainImageDataset(Dataset):
    """
    複数domainのDatasetを1つにまとめる。
    """

    def __init__(self, datasets_list):
        self.datasets_list = datasets_list
        self.index_map = []

        for dataset_idx, dataset in enumerate(datasets_list):
            for sample_idx in range(len(dataset)):
                self.index_map.append((dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index_map[idx]
        return self.datasets_list[dataset_idx][sample_idx]


def get_yoshiken_domain_path(workdir, domain_name):
    workdir = Path(workdir)

    if domain_name == "km":
        return workdir / "azumadata/じん肺画像/高知大学で使っているもの/KM_dicom_dataset"

    if domain_name == "nihcc":
        return workdir / "azumadata/じん肺画像/高知大学で使っているもの/nih"

    if domain_name == "niosh":
        return workdir / "azumadata/じん肺画像/高知大学で使っているもの/NIOSH_practice" 

    raise ValueError(f"Unknown Yoshiken domain: {domain_name}")


def split_dataset(dataset, train_ratio, val_ratio, seed):
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split: total={n_total}, "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_yoshiken_train_val_test_loaders(
    workdir,
    domains,
    transform,
    batch_size,
    num_workers,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
):
    """
    Yoshiken Data用のDomain分類loader。

    KM, NIHCC, NIOSHの各ディレクトリから全画像を読み込み，
    domainごとに train / val / test に分割する。
    """

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for domain_label, domain_name in enumerate(domains):
        root = get_yoshiken_domain_path(
            workdir=workdir,
            domain_name=domain_name,
        )

        full_dataset = FlatDomainImageDataset(
            root=root,
            transform=transform,
            domain_label=domain_label,
        )

        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset=full_dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed + domain_label,
        )

        print(
            f"[INFO] {domain_name}: "
            f"total={len(full_dataset)}, "
            f"train={len(train_dataset)}, "
            f"val={len(val_dataset)}, "
            f"test={len(test_dataset)}, "
            f"domain_label={domain_label}, "
            f"path={root}"
        )

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    train_dataset = MultiDomainImageDataset(train_datasets)
    val_dataset = MultiDomainImageDataset(val_datasets)
    test_dataset = MultiDomainImageDataset(test_datasets)

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = make_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    test_loader = make_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    print(f"[INFO] total train images: {len(train_dataset)}")
    print(f"[INFO] total val images: {len(val_dataset)}")
    print(f"[INFO] total test images: {len(test_dataset)}")

    return train_loader, val_loader, test_loader