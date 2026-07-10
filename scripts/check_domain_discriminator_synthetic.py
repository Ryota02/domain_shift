import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm

from src.config import load_config
from src.utils import set_seed, get_device
from src.ada_model import SupervisedADAModel


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = tensor + noise
        return torch.clamp(tensor, 0.0, 1.0)


class SyntheticDomainPairDataset(Dataset):
    """
    同じ画像を2つのdomainとして使うDataset。

    indexが偶数:
        original domain, label=0

    indexが奇数:
        shifted domain, label=1
    """

    def __init__(self, base_dataset, transform_original, transform_shifted):
        self.base_dataset = base_dataset
        self.transform_original = transform_original
        self.transform_shifted = transform_shifted

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):
        base_idx = idx // 2
        domain_label = idx % 2

        image, class_label = self.base_dataset[base_idx]

        if domain_label == 0:
            image = self.transform_original(image)
        else:
            image = self.transform_shifted(image)

        return image, domain_label, class_label


def build_synthetic_domain_loader(cfg):
    img_size = cfg["train"].get("img_size", 224)
    batch_size = cfg["train"].get("batch_size", 32)
    num_workers = cfg["train"].get("num_workers", 0)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 通常domain
    transform_original = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 人工的に作った別domain
    transform_shifted = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(
            brightness=0.02,
            contrast=0.02,
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 0.5),
        ),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.002),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 例: target train/adaptを使う
    data_root = cfg["workdir"] / cfg["target"]["path"] / "train"

    base_dataset = datasets.ImageFolder(
        root=str(data_root),
        transform=None,
    )

    synthetic_dataset = SyntheticDomainPairDataset(
        base_dataset=base_dataset,
        transform_original=transform_original,
        transform_shifted=transform_shifted,
    )

    loader = DataLoader(
        synthetic_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def train_domain_discriminator_only(
    model,
    loader,
    device,
    epochs=10,
    lr=1e-4,
):
    """
    Feature Extractorを固定して，
    Domain Discriminatorだけ学習する。
    """

    model.eval()

    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = False

    for p in model.domain_discriminator.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        model.domain_discriminator.parameters(),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "domain_loss": [],
        "domain_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model.domain_discriminator.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, domain_labels, _ in tqdm(loader, desc=f"ADA {epoch}/{epochs}"):
            images = images.to(device)
            domain_labels = domain_labels.to(device)

            with torch.no_grad():
                _, features = model.forward_class(images)

            # GRLは使わない
            domain_logits = model.domain_discriminator(features.detach())

            loss = criterion(domain_logits, domain_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = domain_logits.argmax(dim=1)

            total_loss += loss.item() * domain_labels.size(0)
            correct += (preds == domain_labels).sum().item()
            total += domain_labels.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total

        history["domain_loss"].append(epoch_loss)
        history["domain_accuracy"].append(epoch_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Domain Loss: {epoch_loss:.4f} "
            f"Domain Acc: {epoch_acc:.4f}"
        )

    return history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to source_pretrained_model.pth or best_model.pth",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(42)
    device = get_device(require_cuda=True)

    model = SupervisedADAModel(
        num_classes=len(cfg["classes"]),
        feature_dim=cfg["model"].get("feature_dim", 1920),
        pretrained=False,
        grl_lambda=cfg["ada"].get("grl_lambda", 1.0),
    )

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    loader = build_synthetic_domain_loader(cfg)

    history = train_domain_discriminator_only(
        model=model,
        loader=loader,
        device=device,
        epochs=10,
        lr=1e-4,
    )

    print("[RESULT]")
    print("Final synthetic domain acc:", history["domain_accuracy"][-1])


if __name__ == "__main__":
    main()