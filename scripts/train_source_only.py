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
from src.model import build_model
from src.train import fit_source_only
from src.evaluate import evaluate_model
from src.metrics import compute_binary_metrics, print_metrics, print_classification_report
from src.plots import plot_confusion_matrix, plot_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    set_seed(42)

    output_dir = cfg["output_dir"]
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"
    result_dir = output_dir / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(require_cuda=True)

    print("[INFO] experiment:", cfg["experiment_name"])
    print("[INFO] device:", device)
    print("[INFO] torch:", torch.__version__)
    print("[INFO] torch cuda:", torch.version.cuda)

    datasets_dict = build_datasets(cfg)
    loaders = build_loaders(cfg, datasets_dict)

    print("Target classes:", datasets_dict["target_classes"])
    print("Target class_to_idx:", datasets_dict["target_class_to_idx"])

    model = build_model(cfg)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(cfg["train"]["beta1"], cfg["train"]["beta2"])
    )

    start_time = time.time()

    history = fit_source_only(
        model=model,
        train_loader=loaders["source_train"],
        val_loader=loaders["source_val"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=cfg["train"]["epochs"],
    )

    end_time = time.time()

    torch.save(
        model.state_dict(),
        checkpoint_dir / "best_model.pth"
    )

    test_result = evaluate_model(
        model=model,
        loader=loaders["target_test"],
        criterion=criterion,
        device=device,
    )

    print(f"Test Loss: {test_result['loss']:.4f}")
    print(f"Test Accuracy: {test_result['accuracy']:.4f}")

    metrics = compute_binary_metrics(
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        y_proba=test_result["y_proba"],
        class_to_idx=datasets_dict["target_class_to_idx"],
        positive_class=cfg["positive_class"],
    )

    print_metrics(metrics)

    print("[TEST] classification_report")
    print_classification_report(
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        class_names=datasets_dict["target_classes"],
    )

    result_log = {
        "experiment_name": cfg["experiment_name"],
        "setting": cfg["setting"],
        "sources": cfg["sources"],
        "target": cfg["target"],
        "classes": cfg["classes"],
        "positive_class": cfg["positive_class"],
        "best_val_accuracy": float(best_val_acc),
        "test_loss": float(test_result["loss"]),
        "test_accuracy": float(test_result["accuracy"]),
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
    }

    with open(result_dir / "metrics.json", "w") as f:
        json.dump(result_log, f, indent=2)

    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_labels=datasets_dict["target_classes"],
        save_path=figure_dir / "confusion_matrix.png"
    )

    plot_history(
        history=history,
        save_dir=figure_dir,
    )

    print("[INFO] Saved results to:", output_dir)

if __name__ == "__main__":
    main()