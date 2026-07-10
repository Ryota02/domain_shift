import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn

from src.domain_data import build_train_val_test_loaders
from src.domain_models import build_domain_classifier
from src.domain_train import fit_domain_classifier
from src.domain_evaluate import evaluate_domain_classifier


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
        "--epochs",
        type=int,
        default=20,
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
        "--val_ratio",
        type=float,
        default=0.2,
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
        "--output_dir",
        type=str,
        default="outputs/cnn_domain_classifier",
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
    )

    return parser.parse_args()


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

    train_loader, val_loader, test_loader = build_train_val_test_loaders(
        workdir=args.workdir,
        domains=args.domains,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        split_train=args.split_train,
        split_test=args.split_test,
        seed=42,
    )

    model = build_domain_classifier(
        model_name=args.model,
        num_domains=num_domains,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found.")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
    )

    criterion = nn.CrossEntropyLoss()

    history, best_state_dict, best_val_acc = fit_domain_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        num_domains=num_domains,
    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_result = evaluate_domain_classifier(
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
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_acc": best_val_acc,
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

    checkpoint_path = output_dir / output_name.replace(".json", ".pth")

    torch.save(
        model.state_dict(),
        checkpoint_path,
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

    print("Test Confusion Matrix:")
    print(test_result["confusion_matrix"])
    print("[INFO] Saved result to:", output_path)
    print("[INFO] Saved checkpoint to:", checkpoint_path)


if __name__ == "__main__":
    main()