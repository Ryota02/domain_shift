import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class DomainImageFolder(Dataset):
    def __init__(self, root, transform, domain_label, domain_name):
        self.dataset = datasets.ImageFolder(
            root=str(root),
            transform=transform,
        )
        self.domain_label = domain_label
        self.domain_name = domain_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, disease_label = self.dataset[idx]
        return image, self.domain_label, disease_label


class MultiDomainImageDataset(Dataset):
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workdir",
        type=str,
        default="/media/share/Member/ueki/datasets",
    )

    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        required=True,
        choices=["china", "doha", "nigeria"],
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "resnet50", "vgg16"],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Trained CNN domain classifier checkpoint.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split used for t-SNE visualization.",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_samples_per_domain",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cnn_domain_tsne",
    )

    return parser.parse_args()


def get_domain_path(workdir, domain_name, split):
    workdir = Path(workdir)

    if domain_name == "china":
        return workdir / "ZhangLabData_binary_dataset" / split

    if domain_name == "doha":
        return workdir / "COVID-19_Radiography_binary_dataset_clean" / split

    if domain_name == "nigeria":
        return workdir / "nigerian_pneumonia_binary_dataset" / split

    raise ValueError(f"Unknown domain: {domain_name}")


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_loader(args):
    transform = build_transform(args.img_size)

    domain_datasets = []

    for domain_label, domain_name in enumerate(args.domains):
        root = get_domain_path(
            workdir=args.workdir,
            domain_name=domain_name,
            split=args.split,
        )

        print(
            f"[INFO] domain={domain_name}, "
            f"label={domain_label}, "
            f"split={args.split}, "
            f"path={root}"
        )

        dataset = DomainImageFolder(
            root=root,
            transform=transform,
            domain_label=domain_label,
            domain_name=domain_name,
        )

        if args.max_samples_per_domain is not None:
            n = min(len(dataset), args.max_samples_per_domain)
            indices = list(range(n))
            dataset = torch.utils.data.Subset(dataset, indices)

        print(f"[INFO] num images used: {len(dataset)}")

        domain_datasets.append(dataset)

    merged_dataset = MultiDomainImageDataset(domain_datasets)

    loader = DataLoader(
        merged_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return loader


def build_model(model_name, num_domains):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_domains)
        return model

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_domains)
        return model

    if model_name == "vgg16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_domains)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def extract_features_resnet(model, images):
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    return x


def extract_features_vgg(model, images):
    x = model.features(images)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    # 最後の分類層の手前まで通す
    for layer in model.classifier[:-1]:
        x = layer(x)

    return x


def extract_features(model, model_name, loader, device):
    model.eval()

    all_features = []
    all_domain_labels = []
    all_disease_labels = []

    with torch.no_grad():
        for images, domain_labels, disease_labels in loader:
            images = images.to(device)

            if model_name.startswith("resnet"):
                features = extract_features_resnet(model, images)
            elif model_name == "vgg16":
                features = extract_features_vgg(model, images)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            all_features.append(features.cpu().numpy())
            all_domain_labels.append(domain_labels.numpy())
            all_disease_labels.append(disease_labels.numpy())

    features = np.concatenate(all_features, axis=0)
    domain_labels = np.concatenate(all_domain_labels, axis=0)
    disease_labels = np.concatenate(all_disease_labels, axis=0)

    return features, domain_labels, disease_labels


def plot_tsne(
    tsne_features,
    labels,
    label_names,
    title,
    save_path,
):
    plt.figure(figsize=(8, 7))

    for label_id, label_name in enumerate(label_names):
        mask = labels == label_id

        plt.scatter(
            tsne_features[mask, 0],
            tsne_features[mask, 1],
            s=10,
            alpha=0.7,
            label=label_name,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_domains = len(args.domains)

    print("[INFO] model:", args.model)
    print("[INFO] domains:", args.domains)
    print("[INFO] checkpoint:", args.checkpoint)

    loader = build_loader(args)

    model = build_model(
        model_name=args.model,
        num_domains=num_domains,
    )

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    features, domain_labels, disease_labels = extract_features(
        model=model,
        model_name=args.model,
        loader=loader,
        device=device,
    )

    print("[INFO] features shape:", features.shape)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    tsne_features = tsne.fit_transform(features)

    plot_tsne(
        tsne_features=tsne_features,
        labels=domain_labels,
        label_names=args.domains,
        title=f"t-SNE by domain ({args.model})",
        save_path=output_dir / "tsne_by_domain.png",
    )

    disease_label_names = ["NORMAL", "PNEUMONIA"]

    plot_tsne(
        tsne_features=tsne_features,
        labels=disease_labels,
        label_names=disease_label_names,
        title=f"t-SNE by disease label ({args.model})",
        save_path=output_dir / "tsne_by_disease.png",
    )

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "domain_labels.npy", domain_labels)
    np.save(output_dir / "disease_labels.npy", disease_labels)
    np.save(output_dir / "tsne_features.npy", tsne_features)

    print("[INFO] Saved t-SNE figures to:", output_dir)


if __name__ == "__main__":
    main()