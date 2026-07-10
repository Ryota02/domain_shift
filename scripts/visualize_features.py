import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config import load_config
from src.model import build_model
from src.ada_model import SupervisedADAModel
from src.utils import get_device


class DomainWrapper(torch.utils.data.Dataset):
    """
    ImageFolderのデータにdomain名を追加するためのDataset。
    戻り値:
        image, label, domain_name
    """

    def __init__(self, dataset, domain_name):
        self.dataset = dataset
        self.domain_name = domain_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, self.domain_name


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def limit_dataset(dataset, max_samples):
    """
    各domainから使うサンプル数を制限する。
    t-SNEは重いので、500程度に制限するのがおすすめ。
    """

    if max_samples is None:
        return dataset

    n = min(len(dataset), max_samples)
    indices = list(range(n))
    return Subset(dataset, indices)


def load_checkpoint_to_model(model, checkpoint_path, device):
    """
    checkpointの保存形式が多少違っても読み込めるようにする。
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def extract_features_from_ada(model, images):
    """
    ADAモデル用。
    SupervisedADAModelには forward_class があるので、
    そこから特徴量を取り出す。
    """

    _, features = model.forward_class(images)
    return features


def extract_features_from_source_only(model, images):
    """
    source-onlyモデル用。
    torchvisionのDenseNet / ResNetを想定して、
    分類層の直前特徴を取り出す。
    """

    # DenseNetの場合
    if hasattr(model, "features"):
        features = model.features(images)
        features = torch.relu(features)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features

    # ResNetの場合
    if all(hasattr(model, name) for name in ["conv1", "bn1", "relu", "maxpool", "layer1"]):
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        features = torch.flatten(x, 1)
        return features

    raise ValueError(
        "このsource-onlyモデルから特徴量を抽出できません。"
        "DenseNetまたはResNetを想定しています。"
    )


def extract_features(model, loader, device, setting):
    model.eval()

    all_features = []
    all_labels = []
    all_domains = []

    with torch.no_grad():
        for images, labels, domains in loader:
            images = images.to(device)

            if setting == "supervised_ada":
                features = extract_features_from_ada(model, images)
            else:
                features = extract_features_from_source_only(model, images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_domains.extend(list(domains))

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels, all_domains


def reduce_features(features, method):
    method = method.lower()

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=42,
        )
        return reducer.fit_transform(features)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(features)

    raise ValueError(f"Unknown visualization method: {method}")


def plot_by_domain(features_2d, domains, save_path):
    plt.figure(figsize=(8, 6))

    unique_domains = sorted(set(domains))

    for domain in unique_domains:
        idx = np.array([d == domain for d in domains])
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=10,
            alpha=0.7,
            label=domain,
        )

    plt.title("Feature visualization by domain")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_by_class(features_2d, labels, class_names, save_path):
    plt.figure(figsize=(8, 6))

    for class_idx, class_name in enumerate(class_names):
        idx = labels == class_idx
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=10,
            alpha=0.7,
            label=class_name,
        )

    plt.title("Feature visualization by class")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def build_visualization_dataset(cfg):
    workdir = Path(cfg["workdir"])
    img_size = cfg["train"]["img_size"]

    vis_cfg = cfg.get("visualization", {})
    split = vis_cfg.get("use_split", "test")
    max_samples = vis_cfg.get("max_samples_per_domain", 500)

    transform = build_eval_transform(img_size)

    wrapped_datasets = []

    # Source domain
    for source in cfg["sources"]:
        source_dir = workdir / source["path"] / split

        if not source_dir.exists():
            raise FileNotFoundError(f"Source split not found: {source_dir}")

        ds = datasets.ImageFolder(source_dir, transform=transform)
        ds = limit_dataset(ds, max_samples)

        wrapped_datasets.append(
            DomainWrapper(
                dataset=ds,
                domain_name=source["name"],
            )
        )

    # Target domain
    target_dir = workdir / cfg["target"]["path"] / split

    if not target_dir.exists():
        raise FileNotFoundError(f"Target split not found: {target_dir}")

    target_ds = datasets.ImageFolder(target_dir, transform=transform)
    target_ds = limit_dataset(target_ds, max_samples)

    wrapped_datasets.append(
        DomainWrapper(
            dataset=target_ds,
            domain_name=cfg["target"]["name"],
        )
    )

    combined_dataset = ConcatDataset(wrapped_datasets)

    loader = DataLoader(
        combined_dataset,
        batch_size=cfg["train"].get("batch_size", 32),
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )

    return loader


def build_visualization_model(cfg, device):
    num_classes = len(cfg["classes"])
    setting = cfg["setting"]

    if setting == "supervised_ada":
        model = SupervisedADAModel(
            num_classes=num_classes,
            feature_dim=cfg["model"].get("feature_dim", 1920),
            pretrained=False,
            grl_lambda=cfg["ada"].get("grl_lambda", 1.0),
        )
    elif setting == "source_only":
        model = build_model(cfg)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    vis_cfg = cfg.get("visualization", {})
    if not vis_cfg.get("enabled", False):
        print("visualization.enabled is false. Skip feature visualization.")
        return

    device = get_device(require_cuda=True)

    method = vis_cfg.get("method", "tsne")
    color_by = vis_cfg.get("color_by", ["domain", "class"])
    output_dirname = vis_cfg.get("output_dirname", "feature_visualization")
    checkpoint = Path(cfg["output_dir"]) / vis_cfg["checkpoint"]

    output_dir = Path(cfg["output_dir"]) / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building dataset...")
    loader = build_visualization_dataset(cfg)

    print("[INFO] Building model...")
    model = build_visualization_model(cfg, device)
    
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    model = load_checkpoint_to_model(model, checkpoint, device)

    print("[INFO] Extracting features...")
    features, labels, domains = extract_features(
        model=model,
        loader=loader,
        device=device,
        setting=cfg["setting"],
    )

    print(f"[INFO] Feature shape: {features.shape}")

    print(f"[INFO] Reducing features by {method}...")
    features_2d = reduce_features(features, method=method)

    if "domain" in color_by:
        save_path = output_dir / f"{method}_by_domain.png"
        plot_by_domain(
            features_2d=features_2d,
            domains=domains,
            save_path=save_path,
        )
        print(f"[INFO] Saved: {save_path}")

    if "class" in color_by:
        save_path = output_dir / f"{method}_by_class.png"
        plot_by_class(
            features_2d=features_2d,
            labels=labels,
            class_names=cfg["classes"],
            save_path=save_path,
        )
        print(f"[INFO] Saved: {save_path}")

    print(f"[INFO] Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()