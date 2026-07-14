import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch

from src.domain_data import build_domain_loader
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
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
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
        "--output_dir",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_domains = len(args.domains)

    loader = build_domain_loader(
        workdir=args.workdir,
        domains=args.domains,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

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