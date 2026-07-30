import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from torchvision import transforms

from src.yoshiken_domain_data import (
    FlatDomainImageDataset,
    MultiDomainImageDataset,
    get_yoshiken_domain_path,
    split_dataset,
    make_loader,
)
from src.domain_models import build_domain_classifier_without_weights
from src.domain_visualize import (
    extract_cnn_domain_features,
    run_tsne,
    plot_tsne,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workdir",
        type=str,
        default="/media/share/Member/ueki",
    )

    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["km", "nihcc", "niosh"],
        choices=["km", "nihcc", "niosh"],
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
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to visualize. Splits are created internally from all images.",
    )

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    return parser.parse_args()


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_yoshiken_tsne_loader(args):
    transform = build_transform(args.img_size)

    selected_datasets = []

    for domain_label, domain_name in enumerate(args.domains):
        root = get_yoshiken_domain_path(
            workdir=args.workdir,
            domain_name=domain_name,
        )

        full_dataset = FlatDomainImageDataset(
            root=root,
            transform=transform,
            domain_label=domain_label,
        )

        if args.split == "all":
            selected_dataset = full_dataset
            split_name = "all"

        else:
            train_dataset, val_dataset, test_dataset = split_dataset(
                dataset=full_dataset,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=42 + domain_label,
            )

            if args.split == "train":
                selected_dataset = train_dataset
            elif args.split == "val":
                selected_dataset = val_dataset
            elif args.split == "test":
                selected_dataset = test_dataset
            else:
                raise ValueError(f"Unknown split: {args.split}")

            split_name = args.split

        print(
            f"[INFO] {domain_name}: "
            f"split={split_name}, "
            f"num_images={len(selected_dataset)}, "
            f"domain_label={domain_label}, "
            f"path={root}"
        )

        selected_datasets.append(selected_dataset)

    merged_dataset = MultiDomainImageDataset(selected_datasets)

    loader = make_loader(
        dataset=merged_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"[INFO] total images for t-SNE: {len(merged_dataset)}")

    return loader


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_domains = len(args.domains)

    print("[INFO] Yoshiken t-SNE")
    print("[INFO] domains:", args.domains)
    print("[INFO] model:", args.model)
    print("[INFO] checkpoint:", args.checkpoint)
    print("[INFO] split:", args.split)

    loader = build_yoshiken_tsne_loader(args)

    model = build_domain_classifier_without_weights(
        model_name=args.model,
        num_domains=num_domains,
    )

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    features, domain_labels, disease_labels = extract_cnn_domain_features(
        model=model,
        model_name=args.model,
        loader=loader,
        device=device,
    )

    print("[INFO] features shape:", features.shape)

    tsne_features = run_tsne(features)

    plot_tsne(
        tsne_features=tsne_features,
        labels=domain_labels,
        label_names=args.domains,
        title=f"Yoshiken t-SNE by domain ({args.model}, {args.split})",
        save_path=output_dir / "tsne_by_domain.png",
    )

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "domain_labels.npy", domain_labels)
    np.save(output_dir / "disease_labels.npy", disease_labels)
    np.save(output_dir / "tsne_features.npy", tsne_features)

    print("[INFO] Saved t-SNE figures to:", output_dir)


if __name__ == "__main__":
    main()