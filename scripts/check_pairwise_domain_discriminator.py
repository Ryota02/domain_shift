import argparse
import json
import sys
from pathlib import Path
from itertools import cycle

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.config import load_config
from src.utils import set_seed, get_device
from src.ada_model import SupervisedADAModel


class DomainDataset(Dataset):
    """
    ImageFolderにdomain labelを付けるDataset。
    疾患ラベルは使わず，domain labelだけを返す。
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


class MultiDomainDataset(Dataset):
    """
    複数DomainDatasetをまとめる。
    """

    def __init__(self, domain_datasets):
        self.domain_datasets = domain_datasets

        self.index_map = []
        for domain_dataset_idx, dset in enumerate(domain_datasets):
            for sample_idx in range(len(dset)):
                self.index_map.append((domain_dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        domain_dataset_idx, sample_idx = self.index_map[idx]
        return self.domain_datasets[domain_dataset_idx][sample_idx]


class SimpleDomainDiscriminator(nn.Module):
    """
    任意のdomain数に対応するDomain Discriminator。
    2クラスでも3クラスでも使える。
    """

    def __init__(self, feature_dim=1920, num_domains=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_domains),
        )

    def forward(self, x):
        return self.net(x)


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
        help="source_pretrained_model.pth or best_model.pth",
    )

    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        required=True,
        choices=["china", "doha", "nigeria"],
        help="Domains to classify. Example: --domains doha nigeria",
    )

    parser.add_argument(
        "--split_train",
        type=str,
        default="train",
        help="Split used for training domain discriminator.",
    )

    parser.add_argument(
        "--split_eval",
        type=str,
        default="test",
        help="Split used for evaluating domain discriminator.",
    )

    return parser.parse_args()


def build_transform(cfg):
    img_size = cfg["train"].get("img_size", 224)

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_domain_path(cfg, domain_name, split):
    workdir = cfg["workdir"]

    if domain_name == "china":
        return workdir / "ZhangLabData_binary_dataset" / split

    if domain_name == "doha":
        return workdir / "COVID-19_Radiography_binary_dataset_clean" / split

    if domain_name == "nigeria":
        return workdir / "nigerian_pneumonia_binary_dataset" / split

    raise ValueError(f"Unknown domain: {domain_name}")


def build_domain_loader(cfg, domain_names, split, shuffle):
    transform = build_transform(cfg)

    domain_datasets = []

    for domain_label, domain_name in enumerate(domain_names):
        root = get_domain_path(
            cfg=cfg,
            domain_name=domain_name,
            split=split,
        )

        print(
            f"[INFO] domain={domain_name}, "
            f"label={domain_label}, "
            f"split={split}, "
            f"path={root}"
        )

        dset = DomainDataset(
            root=root,
            transform=transform,
            domain_label=domain_label,
        )

        print(f"[INFO] num images: {len(dset)}")

        domain_datasets.append(dset)

    merged_dataset = MultiDomainDataset(domain_datasets)

    loader = DataLoader(
        merged_dataset,
        batch_size=cfg["train"].get("batch_size", 32),
        shuffle=shuffle,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=True,
    )

    return loader


def load_feature_model(cfg, checkpoint_path, device):
    model = SupervisedADAModel(
        num_classes=len(cfg["classes"]),
        feature_dim=cfg["model"].get("feature_dim", 1920),
        pretrained=False,
        grl_lambda=cfg["ada"].get("grl_lambda", 1.0),
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = False

    model.eval()

    return model


def train_one_epoch(
    feature_model,
    domain_discriminator,
    loader,
    criterion,
    optimizer,
    device,
):
    feature_model.eval()
    domain_discriminator.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, domain_labels, _ in loader:
        images = images.to(device)
        domain_labels = domain_labels.to(device)

        with torch.no_grad():
            _, features = feature_model.forward_class(images)

        logits = domain_discriminator(features.detach())
        loss = criterion(logits, domain_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * domain_labels.size(0)
        correct += (preds == domain_labels).sum().item()
        total += domain_labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def evaluate(
    feature_model,
    domain_discriminator,
    loader,
    criterion,
    device,
    num_domains,
):
    feature_model.eval()
    domain_discriminator.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0 for _ in range(num_domains)]
    class_total = [0 for _ in range(num_domains)]
    pred_count = [0 for _ in range(num_domains)]

    confusion = torch.zeros(num_domains, num_domains, dtype=torch.long)

    with torch.no_grad():
        for images, domain_labels, _ in loader:
            images = images.to(device)
            domain_labels = domain_labels.to(device)

            _, features = feature_model.forward_class(images)

            logits = domain_discriminator(features)
            loss = criterion(logits, domain_labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * domain_labels.size(0)
            correct += (preds == domain_labels).sum().item()
            total += domain_labels.size(0)

            for d in range(num_domains):
                mask = domain_labels == d
                class_correct[d] += (preds[mask] == d).sum().item()
                class_total[d] += mask.sum().item()
                pred_count[d] += (preds == d).sum().item()

            for t, p in zip(domain_labels.cpu(), preds.cpu()):
                confusion[t.long(), p.long()] += 1

    domain_acc_by_class = {}

    for d in range(num_domains):
        if class_total[d] == 0:
            domain_acc_by_class[d] = None
        else:
            domain_acc_by_class[d] = class_correct[d] / class_total[d]

    pred_ratio = {
        d: pred_count[d] / total
        for d in range(num_domains)
    }

    return {
        "loss": total_loss / total,
        "domain_acc": correct / total,
        "domain_acc_by_class": domain_acc_by_class,
        "pred_ratio": pred_ratio,
        "confusion_matrix": confusion.tolist(),
        "total": total,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    epochs = 20
    lr = 1e-4

    set_seed(42)
    device = get_device(require_cuda=True)

    domain_names = args.domains
    num_domains = len(domain_names)

    if num_domains < 2:
        raise ValueError("Please specify at least two domains.")

    print("[INFO] domains:", domain_names)
    print("[INFO] num_domains:", num_domains)
    print("[INFO] checkpoint:", args.checkpoint)

    output_dir = cfg["output_dir"] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # if args.output_name is None:
    #     output_name = "domain_probe_" + "_vs_".join(domain_names) + ".json"
    # else:
    #     output_name = args.output_name

    train_loader = build_domain_loader(
        cfg=cfg,
        domain_names=domain_names,
        split=args.split_train,
        shuffle=True,
    )

    eval_loader = build_domain_loader(
        cfg=cfg,
        domain_names=domain_names,
        split=args.split_eval,
        shuffle=False,
    )

    feature_model = load_feature_model(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    domain_discriminator = SimpleDomainDiscriminator(
        feature_dim=cfg["model"].get("feature_dim", 1920),
        num_domains=num_domains,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        domain_discriminator.parameters(),
        lr=lr,
    )

    history = {
        "train_loss": [],
        "train_domain_acc": [],
        "eval_loss": [],
        "eval_domain_acc": [],
    }

    print("[INFO] Start training Domain Discriminator only")
    print("[INFO] Feature Extractor is frozen")
    print("[INFO] GRL is not used")

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            feature_model=feature_model,
            domain_discriminator=domain_discriminator,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        eval_result = evaluate(
            feature_model=feature_model,
            domain_discriminator=domain_discriminator,
            loader=eval_loader,
            criterion=criterion,
            device=device,
            num_domains=num_domains,
        )

        history["train_loss"].append(train_result["loss"])
        history["train_domain_acc"].append(train_result["accuracy"])
        history["eval_loss"].append(eval_result["loss"])
        history["eval_domain_acc"].append(eval_result["domain_acc"])

        by_class_text = " ".join([
            f"{domain_names[d]}Acc={eval_result['domain_acc_by_class'][d]:.4f}"
            for d in range(num_domains)
            if eval_result["domain_acc_by_class"][d] is not None
        ])

        pred_ratio_text = " ".join([
            f"Pred{domain_names[d]}={eval_result['pred_ratio'][d]:.4f}"
            for d in range(num_domains)
        ])

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"TrainLoss={train_result['loss']:.4f} "
            f"TrainAcc={train_result['accuracy']:.4f} "
            f"EvalLoss={eval_result['loss']:.4f} "
            f"EvalAcc={eval_result['domain_acc']:.4f} "
            f"{by_class_text} "
            f"{pred_ratio_text}"
        )

    final_result = evaluate(
        feature_model=feature_model,
        domain_discriminator=domain_discriminator,
        loader=eval_loader,
        criterion=criterion,
        device=device,
        num_domains=num_domains,
    )

    result_log = {
        "domains": domain_names,
        "domain_label_map": {
            domain_name: i
            for i, domain_name in enumerate(domain_names)
        },
        "checkpoint": args.checkpoint,
        "split_train": args.split_train,
        "split_eval": args.split_eval,
        "epochs": epochs,
        "lr": lr,
        "setting": {
            "feature_extractor": "frozen",
            "classifier": "frozen",
            "domain_discriminator": "newly_initialized_and_trained",
            "grl": "not_used",
        },
        "final_result": final_result,
        "history": history,
    }

    # output_path = output_dir / output_name

    # with open(output_path, "w") as f:
    #     json.dump(result_log, f, indent=2)

    print("[RESULT]")
    print(f"Domain Acc: {final_result['domain_acc']:.4f}")

    for d, name in enumerate(domain_names):
        acc = final_result["domain_acc_by_class"][d]
        ratio = final_result["pred_ratio"][d]
        print(f"{name} Domain Acc: {acc:.4f}")
        print(f"Pred {name} Ratio: {ratio:.4f}")

    print("Confusion Matrix:")
    print(final_result["confusion_matrix"])
    # print("[INFO] Saved result to:", output_path)


if __name__ == "__main__":
    main()