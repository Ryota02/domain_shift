import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


from src.config import load_config
from src.data import build_datasets
from src.model import build_model
from src.utils import (
    get_device,
    parse_args,
    set_seed,
)


def resolve_path(path):
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = ROOT_DIR / path

    return path.resolve()


def sample_dataset(dataset, max_samples, seed):
    if (
        max_samples is None
        or max_samples <= 0
        or max_samples >= len(dataset)
    ):
        return dataset

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(dataset),
        size=max_samples,
        replace=False,
    )

    return Subset(
        dataset,
        indices.tolist(),
    )


def make_loader(
    dataset,
    batch_size,
    num_workers,
    device,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # DataParallelで保存された場合への対応
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "[INFO] Loaded checkpoint:",
        checkpoint_path,
    )


def extract_features(
    model,
    loader,
    device,
):
    model.eval()

    features = []
    labels = []

    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            batch_features = (
                model.forward_features(images)
            )

            if batch_features.ndim > 2:
                batch_features = (
                    torch.flatten(
                        batch_features,
                        start_dim=1,
                    )
                )

            features.append(
                batch_features.cpu().numpy()
            )

            labels.append(
                batch_labels.numpy()
            )

    return (
        np.concatenate(features),
        np.concatenate(labels),
    )


def calculate_tsne(
    source_features,
    target_features,
    config,
):
    features = np.concatenate(
        [
            source_features,
            target_features,
        ]
    )

    features = StandardScaler().fit_transform(
        features
    )

    pca_dim = min(
        config.get("pca_dim", 50),
        features.shape[1],
        features.shape[0] - 1,
    )

    features = PCA(
        n_components=pca_dim,
        random_state=config["seed"],
    ).fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=config.get(
            "perplexity",
            30.0,
        ),
        max_iter=config.get(
            "max_iter",
            1000,
        ),
        learning_rate="auto",
        init="pca",
        random_state=config["seed"],
    )

    return tsne.fit_transform(
        features
    )


def plot_by_domain(
    embedding,
    domain_labels,
    output_path,
):
    plt.figure(figsize=(8, 7))

    for label, name, marker in [
        (0, "Source", "o"),
        (1, "Target", "^"),
    ]:
        mask = domain_labels == label

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            alpha=0.55,
            marker=marker,
            label=name,
        )

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE by domain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
    )
    plt.close()


def plot_by_class(
    embedding,
    class_labels,
    class_names,
    output_path,
):
    plt.figure(figsize=(11, 8))

    color_map = plt.get_cmap(
        "tab20",
        len(class_names),
    )

    for class_id, class_name in enumerate(
        class_names
    ):
        mask = class_labels == class_id

        if not np.any(mask):
            continue

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.55,
            color=color_map(class_id),
            label=class_name,
        )

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE by class")

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_csv(
    embedding,
    domain_labels,
    class_labels,
    class_names,
    output_path,
):
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "tsne_1",
            "tsne_2",
            "domain",
            "class_id",
            "class_name",
        ])

        for point, domain, class_id in zip(
            embedding,
            domain_labels,
            class_labels,
        ):
            writer.writerow([
                float(point[0]),
                float(point[1]),
                (
                    "Source"
                    if domain == 0
                    else "Target"
                ),
                int(class_id),
                class_names[int(class_id)],
            ])


def main():
    args = parse_args()
    cfg = load_config(args.config)

    tsne_cfg = cfg["tsne"]

    seed = tsne_cfg.get(
        "seed",
        cfg["train"].get("seed", 42),
    )

    tsne_cfg["seed"] = seed

    set_seed(seed)

    device = get_device(
        require_cuda=tsne_cfg.get(
            "require_cuda",
            True,
        )
    )

    checkpoint_path = resolve_path(
        tsne_cfg["checkpoint"]
    )

    output_dir = resolve_path(
        tsne_cfg.get(
            "output_dir",
            Path(cfg["output_dir"])
            / "figures"
            / "tsne",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets_dict = build_datasets(cfg)

    source_key = tsne_cfg.get(
        "source_dataset",
        "source_tsne",
    )

    target_key = tsne_cfg.get(
        "target_dataset",
        "target_test",
    )

    source_dataset = datasets_dict[
        source_key
    ]

    target_dataset = datasets_dict[
        target_key
    ]

    source_dataset = sample_dataset(
        source_dataset,
        tsne_cfg.get("max_source", 2000),
        seed,
    )

    target_dataset = sample_dataset(
        target_dataset,
        tsne_cfg.get("max_target", 2000),
        seed + 1,
    )

    batch_size = tsne_cfg.get(
        "batch_size",
        64,
    )

    num_workers = tsne_cfg.get(
        "num_workers",
        4,
    )

    source_loader = make_loader(
        source_dataset,
        batch_size,
        num_workers,
        device,
    )

    target_loader = make_loader(
        target_dataset,
        batch_size,
        num_workers,
        device,
    )

    model = build_model(cfg).to(device)

    load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    source_features, source_labels = (
        extract_features(
            model,
            source_loader,
            device,
        )
    )

    target_features, target_labels = (
        extract_features(
            model,
            target_loader,
            device,
        )
    )

    embedding = calculate_tsne(
        source_features,
        target_features,
        tsne_cfg,
    )

    class_labels = np.concatenate([
        source_labels,
        target_labels,
    ])

    domain_labels = np.concatenate([
        np.zeros(
            len(source_labels),
            dtype=np.int64,
        ),
        np.ones(
            len(target_labels),
            dtype=np.int64,
        ),
    ])

    class_names = datasets_dict[
        "target_classes"
    ]

    plot_by_domain(
        embedding,
        domain_labels,
        output_dir / "tsne_domain.png",
    )

    plot_by_class(
        embedding,
        class_labels,
        class_names,
        output_dir / "tsne_class.png",
    )

    save_csv(
        embedding,
        domain_labels,
        class_labels,
        class_names,
        output_dir / "tsne_embedding.csv",
    )

    np.save(
        output_dir / "tsne_embedding.npy",
        embedding,
    )

    print(
        "[INFO] Source features:",
        source_features.shape,
    )
    print(
        "[INFO] Target features:",
        target_features.shape,
    )
    print(
        "[INFO] Results saved to:",
        output_dir,
    )


if __name__ == "__main__":
    main()