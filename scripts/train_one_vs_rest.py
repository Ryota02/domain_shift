import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


from src.config import load_config
from src.one_vs_rest_data import (
    build_one_vs_rest_datasets,
    build_one_vs_rest_loaders,
    calculate_pos_weight,
)
from src.one_vs_rest_model import build_one_vs_rest_model
from src.one_vs_rest_plots import (
    plot_confusion,
    plot_roc_and_pr,
    plot_training_history,
)
from src.one_vs_rest_train import fit_one_vs_rest
from src.utils import (
    get_device,
    make_json_serializable,
    set_seed,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--target_class",
        type=str,
        choices=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax'],
        required=True,
        help="Choose Target class from ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if cfg.get("task", {}).get("type") != "one_vs_rest":
        raise ValueError("task.type must be 'one_vs_rest'.")

    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    device = get_device(
        require_cuda=bool(cfg["train"].get("require_cuda", True))
    )
    target_class = args.target_class
    
    output_dir = Path(cfg["output_dir"]) / target_class
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"

    for directory in [checkpoint_dir, result_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    datasets_dict = build_one_vs_rest_datasets(cfg, target_class)
    loaders = build_one_vs_rest_loaders(cfg, datasets_dict)
    model = build_one_vs_rest_model(cfg).to(device)

    use_pos_weight = bool(cfg.get("loss", {}).get("use_pos_weight", True))
    pos_weight = (
        calculate_pos_weight(datasets_dict)
        if use_pos_weight
        else 1.0
    )

    print("[INFO] Device:", device)
    print("[INFO] Backbone:", cfg["model"]["backbone"])
    print("[INFO] Target class:", datasets_dict["target_class"])
    print("[INFO] pos_weight:", pos_weight)

    start_time = time.time()

    result = fit_one_vs_rest(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=device,
        pos_weight=pos_weight,
    )

    elapsed_time = time.time() - start_time

    best_checkpoint = checkpoint_dir / "best_model.pth"
    last_checkpoint = checkpoint_dir / "last_model.pth"

    checkpoint_common = {
        "target_class": datasets_dict["target_class"],
        "class_names": datasets_dict["class_names"],
        "class_to_idx": datasets_dict["class_to_idx"],
        "original_classes": datasets_dict["original_classes"],
        "backbone": cfg["model"]["backbone"],
        "config": make_json_serializable(cfg),
    }

    torch.save(
        {
            **checkpoint_common,
            "model_state_dict": result["best_state_dict"],
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "threshold": result["threshold"],
        },
        best_checkpoint,
    )

    torch.save(
        {
            **checkpoint_common,
            "model_state_dict": result["last_state_dict"],
            "epoch": int(cfg["train"].get("epochs", 30)),
        },
        last_checkpoint,
    )

    test_result = result["test_result"]
    test_metrics = test_result["metrics"]

    plot_training_history(
        result["history"],
        figure_dir,
    )

    plot_roc_and_pr(
        test_result["y_true"],
        test_result["y_probability"],
        figure_dir,
    )

    plot_confusion(
        test_result["y_true"],
        test_result["y_probability"],
        result["threshold"],
        datasets_dict["class_names"],
        figure_dir / "confusion_matrix.png",
    )

    result_log = {
        "config_path": args.config,
        "target_class": datasets_dict["target_class"],
        "backbone": cfg["model"]["backbone"],
        "counts": datasets_dict["counts"],
        "pos_weight": result["pos_weight"],
        "selection_metric": cfg.get("evaluation", {}).get(
            "selection_metric",
            "pr_auc",
        ),
        "best_epoch": result["best_epoch"],
        "best_score": result["best_score"],
        "threshold": result["threshold"],
        "validation_metrics": result["validation_result"]["metrics"],
        "test_metrics": test_metrics,
        "training_time_seconds": elapsed_time,
        "history": result["history"],
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
    }

    result_path = result_dir / "metrics.json"

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_serializable(result_log),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[RESULT]")
    print("Target class:", datasets_dict["target_class"])
    print("Best epoch:", result["best_epoch"])
    print("Threshold:", f"{result['threshold']:.4f}")

    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("[INFO] Best checkpoint:", best_checkpoint)
    print("[INFO] Result:", result_path)
    print("[INFO] Figures:", figure_dir)


if __name__ == "__main__":
    main()
