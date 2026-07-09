import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


class DomainImageFolder(Dataset):
    """
    ImageFolderにdomain labelを付けるDataset。
    疾患ラベルは読み込むが，学習には使わない。
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
    複数domainのDatasetを結合する。
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
        default="resnet18",
        choices=["resnet18", "resnet50", "vgg16"],
    )

    parser.add_argument(
        "--split_train",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--split_test",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--val_ratio", 
        type=float,
        default=0.2,
        help="Raiot of train split used for validation."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cnn_domain_classifier",
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="If set, freeze CNN backbone and train only final classifier.",
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


def build_loader(args, split, shuffle):
    transform = build_transform(args.img_size)

    domain_datasets = []

    for domain_label, domain_name in enumerate(args.domains):
        root = get_domain_path(
            workdir=args.workdir,
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
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return loader

def build_train_val_loaders(args):
    """
    各domainのtrainフォルダをtrain/valに分割する。
    domainごとに分割するので，China/Doha/Nigeriaの比率を保ちやすい。
    """

    transform = build_transform(args.img_size)

    train_domain_datasets = []
    val_domain_datasets = []

    generator = torch.Generator().manual_seed(42)

    for domain_label, domain_name in enumerate(args.domains):
        root = get_domain_path(
            workdir=args.workdir,
            domain_name=domain_name,
            split=args.split_train,
        )

        print(
            f"[INFO] domain={domain_name}, "
            f"label={domain_label}, "
            f"split={args.split_train}, "
            f"path={root}"
        )

        full_dataset = DomainImageFolder(
            root=root,
            transform=transform,
            domain_label=domain_label,
        )

        n_total = len(full_dataset)
        n_val = int(n_total * args.val_ratio)
        n_train = n_total - n_val

        if n_train <= 0 or n_val <= 0:
            raise ValueError(
                f"Invalid split for {domain_name}: "
                f"n_total={n_total}, n_train={n_train}, n_val={n_val}"
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

def build_model(model_name, num_domains, freeze_backbone):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features

        if freeze_backbone:
            print("[INFO] Freeze Backbone")
            for p in model.parameters():
                p.requires_grad = False

        model.fc = nn.Linear(in_features, num_domains)
        return model

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features

        if freeze_backbone:
            print("[INFO] Freeze Backbone")
            for p in model.parameters():
                p.requires_grad = False

        model.fc = nn.Linear(in_features, num_domains)
        return model

    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            print("[INFO] Freeze Backbone")
            for p in model.features.parameters():
                p.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_domains)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, domain_labels, _ in loader:
        images = images.to(device)
        domain_labels = domain_labels.to(device)

        logits = model(images)
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


def evaluate(model, loader, criterion, device, num_domains):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0 for _ in range(num_domains)]
    class_total = [0 for _ in range(num_domains)]
    pred_count = [0 for _ in range(num_domains)]

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, domain_labels, _ in loader:
            images = images.to(device)
            domain_labels = domain_labels.to(device)

            logits = model(images)
            loss = criterion(logits, domain_labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * domain_labels.size(0)
            correct += (preds == domain_labels).sum().item()
            total += domain_labels.size(0)

            y_true.extend(domain_labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            for d in range(num_domains):
                mask = domain_labels == d
                class_correct[d] += (preds[mask] == d).sum().item()
                class_total[d] += mask.sum().item()
                pred_count[d] += (preds == d).sum().item()

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

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "loss": total_loss / total,
        "domain_acc": correct / total,
        "domain_acc_by_class": domain_acc_by_class,
        "pred_ratio": pred_ratio,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    args = parse_args()

    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_domains = len(args.domains)

    if num_domains < 2:
        raise ValueError("Please specify at least two domains.")

    print("[INFO] domains:", args.domains)
    print("[INFO] model:", args.model)
    print("[INFO] num_domains:", num_domains)
    print("[INFO] freeze_backbone:", args.freeze_backbone)

    train_loader, val_loader = build_train_val_loaders(args)

    test_loader = build_loader(
        args=args,
        split=args.split_test,
        shuffle=False,
    )

    model = build_model(
        model_name=args.model,
        num_domains=num_domains,
        freeze_backbone=args.freeze_backbone,
    )

    model = model.to(device)

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_domain_acc": [],
        "eval_loss": [],
        "eval_domain_acc": [],
    }

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_result = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_domains=num_domains,
        )

        history["train_loss"].append(train_result["loss"])
        history["train_domain_acc"].append(train_result["accuracy"])
        history["eval_loss"].append(val_result["loss"])
        history["eval_domain_acc"].append(val_result["domain_acc"])

        by_class_text = " ".join([
            f"{args.domains[d]}Acc={val_result['domain_acc_by_class'][d]:.4f}"
            for d in range(num_domains)
            if val_result["domain_acc_by_class"][d] is not None
        ])

        pred_ratio_text = " ".join([
            f"Pred{args.domains[d]}={val_result['pred_ratio'][d]:.4f}"
            for d in range(num_domains)
        ])

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"TrainLoss={train_result['loss']:.4f} "
            f"TrainAcc={train_result['accuracy']:.4f} "
            f"EvalLoss={val_result['loss']:.4f} "
            f"EvalAcc={val_result['domain_acc']:.4f} "
            f"{by_class_text} "
            f"{pred_ratio_text}"
        )

        if val_result["domain_acc"] > best_val_acc:
            best_val_acc = val_result["domain_acc"]
            best_state_dict = {
                k: v.cpu()
                for k, v in model.state_dict().items()
            }

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_result = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_domains=num_domains,
    )

    result_log = {
        "domains": args.domains,
        "domain_label_map": {
            name: i
            for i, name in enumerate(args.domains)
        },
        "model": args.model,
        "freeze_backbone": args.freeze_backbone,
        "split_train": args.split_train,
        "split_test": args.split_test,
        "epochs": args.epochs,
        "lr": args.lr,
        "test_result": test_result,
        "history": history,
    }

    output_name = (
        f"cnn_domain_{args.model}_"
        f"{'_vs_'.join(args.domains)}.json"
    )

    output_path = output_dir / output_name

    with open(output_path, "w") as f:
        json.dump(result_log, f, indent=2)

    torch.save(
        model.state_dict(),
        output_dir / output_name.replace(".json", ".pth"),
    )

    print("[RESULT]")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Final Test Domain Acc: {test_result['domain_acc']:.4f}")

    for d, name in enumerate(args.domains):
        print(
            f"{name} Test Domain Acc: "
            f"{test_result['domain_acc_by_class'][d]:.4f}"
        )
        print(
            f"Pred {name} Ratio: "
            f"{test_result['pred_ratio'][d]:.4f}"
        )

    print("Confusion Matrix:")
    print(test_result["confusion_matrix"])
    print("[INFO] Saved result to:", output_path)


if __name__ == "__main__":
    main()