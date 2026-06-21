import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn

from src.config import load_config
from src.utils import set_seed, get_device
from src.data import build_datasets, build_loaders
from src.ada_model import SupervisedADAModel
from src.ada_train import fit_source_pretrain, fit_supervised_ada
from src.evaluate import evaluate_model, evaluate_domain_discriminator
from src.metrics import compute_binary_metrics, print_metrics, print_classification_report
from src.plots import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ADA config yaml file."
    )
    return parser.parse_args()


def build_optimizer(cfg, model, lr):
    optimizer_name = cfg["train"].get("optimizer", "sgd").lower()
    weight_decay = cfg["train"].get("weight_decay", 0.0)

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found.")

    if optimizer_name == "sgd":
        momentum = cfg["train"].get("momentum", 0.9)
        return torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(42)

    output_dir = cfg["output_dir"]
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(require_cuda=True)

    print("[INFO] experiment:", cfg["experiment_name"])
    print("[INFO] setting:", cfg["setting"])
    print("[INFO] device:", device)
    print("[INFO] torch:", torch.__version__)
    print("[INFO] torch cuda:", torch.version.cuda)

    datasets_dict = build_datasets(cfg)
    loaders = build_loaders(cfg, datasets_dict)

    print("[INFO] target classes:", datasets_dict["target_classes"])
    print("[INFO] target class_to_idx:", datasets_dict["target_class_to_idx"])

    model = SupervisedADAModel(
        num_classes=len(cfg["classes"]),
        feature_dim=cfg["model"].get("feature_dim", 1920),
        pretrained=cfg["model"].get("pretrained", True),
        grl_lambda=cfg["ada"].get("grl_lambda", 1.0),
    )

    model = model.to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    # =========================
    # Stage 1: Source pretraining
    # =========================

    optimizer_source = build_optimizer(
        cfg=cfg,
        model=model,
        lr=cfg["train"]["lr_source"],
    )

    start_time = time.time()

    source_history, source_best_val_acc, source_best_val_loss = fit_source_pretrain(
        model=model,
        source_train_loader=loaders["source_train"],
        source_val_loader=loaders["source_val"],
        criterion_cls=criterion_cls,
        optimizer=optimizer_source,
        device=device,
        epochs=cfg["train"]["epochs_source"],
    )

    torch.save(
        model.state_dict(),
        checkpoint_dir / "source_pretrained_model.pth"
    )

    # =========================
    # Stage 2: Supervised ADA
    # =========================

    optimizer_ada = build_optimizer(
        cfg=cfg,
        model=model,
        lr=cfg["train"]["lr_ada"],
    )

    ada_history, ada_best_val_acc, ada_best_val_loss = fit_supervised_ada(
        model=model,
        source_train_loader=loaders["source_train"],
        target_adapt_loader=loaders["target_adapt"],
        source_val_loader=loaders["source_val"],
        criterion_cls=criterion_cls,
        criterion_domain=criterion_domain,
        optimizer=optimizer_ada,
        device=device,
        epochs=cfg["train"]["epochs_ada"],
        lambda_target_cls=cfg["ada"].get("lambda_target_cls", 1.0),
        beta_domain=cfg["ada"].get("beta_domain", 0.1),
    )

    end_time = time.time()

    torch.save(
        model.state_dict(),
        checkpoint_dir / "best_model.pth"
    )

    # =========================
    # Target test evaluation
    # =========================

    test_result = evaluate_model(
        model=model,
        loader=loaders["target_test"],
        criterion=criterion_cls,
        device=device,
    )

    metrics = compute_binary_metrics(
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        y_proba=test_result["y_proba"],
        class_to_idx=datasets_dict["target_class_to_idx"],
        positive_class=cfg["positive_class"],
    )

    print("[TEST] Target Nigeria")
    print(f"Test Loss: {test_result['loss']:.4f}")
    print(f"Test Accuracy: {test_result['accuracy']:.4f}")

    print_metrics(metrics)

    print("[TEST] classification_report")
    print_classification_report(
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        class_names=datasets_dict["target_classes"],
    )

    domain_result = evaluate_domain_discriminator(
        model=model,
        source_loader=loaders["source_val"],
        target_loader=loaders["target_test"],
        device=device,
    )
    
    print("Domain Discriminator Evaluation")
    print(f"Domain Acc: {domain_result['domain_acc']:.4f}")
    print(f"Source Domain Acc: {domain_result['source_domain_acc']:.4f}")
    print(f"Target Domain Acc: {domain_result['target_domain_acc']:.4f}")

    result_log = {
        "experiment_name": cfg["experiment_name"],
        "setting": cfg["setting"],
        "sources": cfg["sources"],
        "target": cfg["target"],
        "classes": cfg["classes"],
        "positive_class": cfg["positive_class"],
        "model": cfg["model"],
        "train_config": cfg["train"],
        "ada_config": cfg["ada"],
        "source_best_val_accuracy": float(source_best_val_acc),
        "source_best_val_loss": float(source_best_val_loss),
        "ada_best_val_accuracy": float(ada_best_val_acc),
        "ada_best_val_loss": float(ada_best_val_loss),
        "target_test_loss": float(test_result["loss"]),
        "target_test_accuracy": float(test_result["accuracy"]),
        "domain_acc": float(domain_result["domain_acc"]),
        "source_domain_acc": float(domain_result["source_domain_acc"]),
        "target_domain_acc": float(domain_result["target_domain_acc"]),
        "accuracy": float(metrics["accuracy"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "ppv": float(metrics["ppv"]),
        "npv": float(metrics["npv"]),
        "f1": float(metrics["f1"]),
        "auc": float(metrics["auc"]),
        "training_time": float(end_time - start_time),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_to_idx": datasets_dict["target_class_to_idx"],
        "source_history": source_history,
        "ada_history": ada_history,
    }

    with open(result_dir / "metrics.json", "w") as f:
        json.dump(result_log, f, indent=2)

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_labels=datasets_dict["target_classes"],
        save_path=figure_dir / "confusion_matrix.png",
    )

    print("[INFO] Saved results to:", output_dir)


if __name__ == "__main__":
    main()