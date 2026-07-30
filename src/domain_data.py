from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


class DomainFolderDataset(Dataset):
    def __init__(self, root, domains, transform=None):
        self.root = Path(root)
        self.domains = domains
        self.transform = transform

        self.domain_to_idx = {
            name: idx for idx, name in enumerate(domains)
        }

        self.samples = []

        for domain_name in domains:
            domain_dir = self.root / domain_name

            if not domain_dir.exists():
                raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

            domain_label = self.domain_to_idx[domain_name]

            image_paths = sorted([
                path for path in domain_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ])

            if len(image_paths) == 0:
                raise ValueError(f"No images found in {domain_dir}")

            for image_path in image_paths:
                self.samples.append((image_path, domain_label))

            print(
                f"[INFO] {domain_name}: "
                f"label={domain_label}, "
                f"num_images={len(image_paths)}, "
                f"path={domain_dir}"
            )

        print(f"[INFO] total images in {self.root}: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, domain_label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        disease_label = -1

        return image, domain_label, disease_label


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
    data_root,
    split,
    domains,
    img_size,
    batch_size,
    num_workers,
    shuffle,
):
    split_root = Path(data_root) / split

    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    transform = build_domain_transform(img_size)

    dataset = DomainFolderDataset(
        root=split_root,
        domains=domains,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader