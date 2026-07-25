import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config import load_config
from src.data import build_loaders, build_datasets
from src.model import build_model
from src.daln_train import fit_daln
from src.metrics import compute_binary_metrics, print_metrics, print_classification_report
from src.plots import plot_confusion_matrix, plot_daln_training_curves, plot_roc_curve

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)

    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] DALN training")
    print("[INFO] config:", args.config)
    print("[INFO] output_dir:", output_dir)

    datasets_dict = build_datasets(cfg)
    loaders = build_loaders(cfg, datasets_dict)

    required_keys = ["source_train", "target_adapt", "target_test"]
    for key in required_keys:
        if key not in loaders:
            raise KeyError(
                f"loaders must contain '{key}'. "
                f"Current keys: {list(loaders.keys())}"
            )

    model = build_model(cfg)
    model = model.to(device)

    if not hasattr(model, "classifier_head"):
        raise AttributeError(
            "DALN requires model.classifier_head. "
            "Please update src/model.py."
        )
    start_time = time.time()
    result = fit_daln(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=device,
    )

    end_time = time.time()
    final = result["final_test_result"]

    metrics = compute_binary_metrics(
        y_true=final["y_true"],
        y_pred=final["y_pred"],
        y_proba=final["y_proba"],
        class_to_idx=datasets_dict["target_class_to_idx"],
        positive_class=cfg["positive_class"],
    )

    plot_paths = plot_daln_training_curves(
        history=result["history"],
        output_dir=figure_dir
    )

    roc_result = plot_roc_curve(
        y_true=final["y_true"],
        y_proba=final["y_proba"],
        positive_class=cfg["positive_class"],
        class_to_idx=datasets_dict["target_class_to_idx"],
        save_path=figure_dir / "roc_curve.png",
    )

    model_name = cfg["model"].get("backbone", "resnet50")
    output_name = f"daln_{model_name}.json"

   

    result_log = {
        "config": args.config,
        "sources": cfg["sources"],
        "target": cfg["target"],
        "model": cfg["model"],
        "train": cfg["train"],
        "grl": cfg.get("grl", {}),
        "selection_split": result["selection_split"],
        "best_val_acc": result["best_val_acc"],
        "accuracy": float(metrics["accuracy"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "ppv": float(metrics["ppv"]),
        "npv": float(metrics["npv"]),
        "f1": float(metrics["f1"]),
        "auc": float(metrics["auc"]),
        "training_time": float(end_time - start_time),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "history": result["history"],
    }

    with open(result_dir / "metrics.json", "w") as f:
        json.dump(result_log, f, indent=2)

    torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")

    print("[RESULT]")
    print("Selection split:", result["selection_split"])

    if result["best_val_acc"] is None:
        print("Best Val Acc: None")
    else:
        print(f"Best Val Acc: {result['best_val_acc']:.4f}")

    print(f"Final Test Acc: {final['accuracy']:.4f}")

    if "metrics" in final:
        metrics = final["metrics"]
        for key, value in metrics.items():
            print(f"{key}: {value}")

    print("Confusion Matrix:")
    plot_confusion_matrix(
        y_true=final["y_true"],
        y_pred=final["y_pred"],
        class_labels=datasets_dict["target_classes"],
        save_path=figure_dir / "confusion_matrix.png",
    )

    print("[INFO] Saved result to:", result_dir)
    print("[INFO] Saved checkpoint to:", checkpoint_dir)
    print("[INFO] Saved loss curve to:", plot_paths["loss_curve"])
    print("[INFO] Saved accuracy curve to:", plot_paths["accuracy_curve"])
    print("[INFO] Saved AUC curve to:", plot_paths["auc_curve"])
    print("[INFO] Saved Confusion Matrix curve to:", figure_dir/"confusion_matrix.png")

if __name__ == "__main__":
    main()