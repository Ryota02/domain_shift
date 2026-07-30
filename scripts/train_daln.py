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
from src.utils import (
    set_seed, 
    make_json_serializable,
    parse_args,
)
from src.metrics import (
    compute_binary_metrics, 
    compute_multiclass_metrics,
    print_metrics, 
    print_multiclass_metrics, 
    print_classification_report,
)
from src.plots import (
    plot_confusion_matrix, 
    plot_daln_training_curves, 
    plot_multiclass_roc_curve, 
    plot_roc_curve,
)

    
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

    task_type = cfg.get("task")

    print("[INFO] DALN training")
    print("[INFO] Task: ", task_type)
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
    num_dataset_classes = len(datasets_dict["target_classes"])
    
    num_model_classes = cfg["model"].get("num_classes",2,)
    
    if num_dataset_classes != num_model_classes:
        raise ValueError(
            "Number of dataset classes and "
            "model outputs do not match: "
            f"dataset={num_dataset_classes}, "
            f"model={num_model_classes}"
        )

    print("[INFO] Classes:", datasets_dict["target_classes"])
    print("[INFO] class_to_idx:",datasets_dict["target_class_to_idx"],)
    print("[INFO] Source samples:",len(datasets_dict["source_train"]),)
    print("[INFO] Target samples:",len(datasets_dict["target_test"]),)


    model = build_model(cfg)
    model = model.to(device)

    if not hasattr(model, "classifier_head"):
        raise AttributeError(
            "DALN requires model.classifier_head."
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
    torch.save(
    {
        "model_state_dict": result[
            "best_state_dict"
        ],
        "best_epoch": result["best_epoch"],
        "best_score": result["best_score"],
        "selection_split": result[
            "selection_split"
        ],
        "class_names": datasets_dict[
            "target_classes"
        ],
        "class_to_idx": datasets_dict[
            "target_class_to_idx"
        ],
        "config": cfg,
    },
    checkpoint_dir / "best_model.pth",
)

    if task_type == "binary":
        metrics = compute_binary_metrics(
            y_true=final["y_true"],
            y_pred=final["y_pred"],
            y_proba=final["y_proba"],
            class_to_idx=datasets_dict["target_class_to_idx"],
            positive_class=cfg["positive_class"],
        )
    
        roc_result = plot_roc_curve(
            y_true=final["y_true"],
            y_proba=final["y_proba"],
            positive_class=cfg["positive_class"],
            class_to_idx=datasets_dict["target_class_to_idx"],
            save_path=figure_dir / "roc_curve.png",
        )
        print_metrics(metrics)
        roc_log = {"auc": float(roc_result["auc"])}

    elif task_type == "multiclass": 
        metrics = compute_multiclass_metrics(
                y_true=final["y_true"],
                y_pred=final["y_pred"],
                y_proba=final["y_proba"],
                class_names=datasets_dict["target_classes"],
            )
        roc_log = plot_multiclass_roc_curve(
                y_true=final["y_true"],
                y_proba=final["y_proba"],
                class_names=datasets_dict["target_classes"],
                save_path=(figure_dir / "multiclass_roc.png"),
            )
        print_multiclass_metrics(metrics)
    else:
        raise ValueError(
            f"Unsupported task type: "
            f"{task_type}"
        )

    model_name = cfg["model"].get("backbone")

    print_classification_report(
            y_true=final["y_true"],
            y_pred=final["y_pred"],
            class_names=datasets_dict["target_classes"]
    )

    plot_paths = plot_daln_training_curves(
                history=result["history"],
                output_dir=figure_dir
            )

    plot_confusion_matrix(
        y_true=final["y_true"],
        y_pred=final["y_pred"],
        class_labels=datasets_dict["target_classes"],
        save_path=figure_dir / "confusion_matrix.png",
        normalize=False
    )

    plot_confusion_matrix(
        y_true=final["y_true"],
        y_pred=final["y_pred"],
        class_labels=datasets_dict["target_classes"],
        save_path=figure_dir / "confusion_matrix_normalized.png",
        normalize=True
    )
   

    result_log = {
        "config": args.config,
        "sources": cfg["sources"],
        "target": cfg["target"],
        "model": cfg["model"],
        "train": cfg["train"],
        "grl": cfg.get("grl", {}),
        "selection_split": result["selection_split"],
        "best_epoch": result["best_epoch"],
        "best_val_acc": result["best_val_acc"],
        "metrics": make_json_serializable(metrics),
        "roc": make_json_serializable(roc_log),
        "training_time": float(end_time - start_time),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "history": result["history"],
    }

    with open(result_dir / "metrics.json", "w") as f:
        json.dump(result_log, f, indent=2)


    print("[RESULT]")
    print("Selection split:", result["selection_split"])
    print("Best epoch: ", result["best_epoch"])

    if result["best_val_acc"] is None:
        print("Best Val Acc: None")
    else:
        print(f"Best Val Acc: {result['best_val_acc']:.4f}")

    print(f"Final Test Acc: {final['accuracy']:.4f}")

    print("[INFO] Saved result to:", result_dir)
    print("[INFO] Saved checkpoint to:", checkpoint_dir)
    print("[INFO] Saved loss curve to:", plot_paths["loss_curve"])
    print("[INFO] Saved accuracy curve to:", plot_paths["accuracy_curve"])
    print("[INFO] Saved AUC curve to:", plot_paths["auc_curve"])
    print("[INFO] Saved Confusion Matrix curve to:", figure_dir/"confusion_matrix.png")

if __name__ == "__main__":
    main()