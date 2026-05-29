import time
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.config import Config
from src.utils import set_seed, get_device
from src.data import build_datasets, build_loaders
from src.model import build_resnet18
from src.train import fit_source_only
from src.evaluate import evaluate_model
from src.metrics import compute_binary_metrics, print_metrics, print_classification_report
from src.plots import plot_confusion_matrix, plot_history


def main():
    set_seed(42)

    cfg = Config()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "results").mkdir(parents=True, exist_ok=True)

    device = get_device(require_cuda=True)

    print("[INFO] device:", device)
    print("[INFO] torch:", torch.__version__)
    print("[INFO] torch cuda:", torch.version.cuda)

    datasets_dict = build_datasets(cfg)
    loaders = build_loaders(cfg, datasets_dict)

    print("Target class_to_idx:", datasets_dict["target_test"].class_to_idx)

    model = build_resnet18(num_classes=len(cfg.classes), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )

    start_time = time.time()

    history = fit_source_only(
        model=model,
        train_loader=loaders["source_train"],
        val_loader=loaders["source_val"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=cfg.epochs,
    )

    end_time = time.time()

    torch.save(
        model.state_dict(),
        cfg.output_dir / "checkpoints" / "resnet18_source_only.pth",
    )

    test_result = evaluate_model(
        model=model,
        loader=loaders["target_test"],
        criterion=criterion,
        device=device,
    )

    print(f"Test Loss: {test_result['loss']:.4f}")
    print(f"Test Accuracy: {test_result['accuracy']:.4f}")

    class_to_idx = datasets_dict["target_test"].class_to_idx
    class_names = datasets_dict["target_test"].classes

    metrics = compute_binary_metrics(
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        y_proba=test_result["y_proba"],
        class_to_idx=class_to_idx,
        positive_class=cfg.positive_class,
    )

    print_metrics(metrics)

    print("[TEST] classification_report")
    print_classification_report(
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        class_names=class_names,
    )

    result_log = {
        "setting": "source_only",
        "source": ["Doha_clean", "China"],
        "target": "Nigeria",
        "test_loss": float(test_result["loss"]),
        "test_accuracy": float(test_result["accuracy"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "ppv": float(metrics["ppv"]),
        "npv": float(metrics["npv"]),
        "f1": float(metrics["f1"]),
        "auc": float(metrics["auc"]),
        "training_time": float(end_time - start_time),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_to_idx": class_to_idx,
    }

    with open(cfg.output_dir / "results" / "source_only_result.json", "w") as f:
        json.dump(result_log, f, indent=2)

    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_labels=class_names,
        save_path=cfg.output_dir / "figures" / "confusion_matrix.png",
    )

    plot_history(
        history=history,
        save_dir=cfg.output_dir / "figures",
    )


if __name__ == "__main__":
    main()