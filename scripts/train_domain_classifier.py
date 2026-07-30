import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn

from src.config import load_config
from src.domain_data import build_domain_loader
from src.domain_models import build_domain_classifier
from src.domain_train import fit_domain_classifier
from src.domain_evaluate import evaluate_domain_classifier
from src.domain_visualize import plot_multiclass_roc_curve

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "resnet50", "vgg16"],
        help="Choose model from resnet18, resnet50, and vgg16",

    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    data_root = data_cfg["data_root"]
    domains = data_cfg["domains"]
    img_size = data_cfg.get("img_size", 224)
    num_workers = data_cfg.get("num_workers", 0)

    model_name = args.model
    freeze_backbone = False

    epochs = train_cfg.get("epochs", 20)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    output_dir = cfg["output_dir"] / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    num_domains = len(domains)

    print("[INFO] CNN domain classifier")
    print("[INFO] config:", args.config)
    print("[INFO] data_root:", data_root)
    print("[INFO] domains:", domains)
    print("[INFO] model:", model_name)
    print("[INFO] epochs:", epochs)
    print("[INFO] batch_size:", batch_size)

    train_loader = build_domain_loader(
        data_root=data_root,
        split="train",
        domains=domains,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    test_loader = build_domain_loader(
        data_root=data_root,
        split="test",
        domains=domains,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    val_root = Path(data_root) / "val"

    if val_root.exists():
        select_loader = build_domain_loader(
            data_root=data_root,
            split="val",
            domains=domains,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        select_name = "Val"
        selection_split = "val"
    else:
        select_loader = None
        select_name = None
        selection_split = "none"
        print("[WARN] val split not found. Best model is selected by test accuracy.")

    model = build_domain_classifier(
        model_name=model_name,
        num_domains=num_domains,
        freeze_backbone=freeze_backbone,
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
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    history, best_state_dict, best_select_acc = fit_domain_classifier(
        model=model,
        train_loader=train_loader,
        select_loader=select_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        num_domains=num_domains,
        domains=domains,
        select_name=select_name,
    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_test_result = evaluate_domain_classifier(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_domains=num_domains,
    )
    
    output_name = (
        f"domain_{model_name}_"
        f"{'_vs_'.join(domains)}.json"
    )
    
    output_path = output_dir / output_name
    checkpoint_path = output_dir / output_name.replace(".json", ".pth")
    
    roc_path = output_dir / output_name.replace(".json", "_roc.png")
    
    roc_auc_by_class = plot_multiclass_roc_curve(
        y_true=final_test_result["y_true"],
        y_score=final_test_result["y_score"],
        class_names=domains,
        save_path=roc_path,
    )
    
    result_log = {
        "config": args.config,
        "data_root": data_root,
        "domains": domains,
        "domain_label_map": {
            name: idx for idx, name in enumerate(domains)
        },
        "model": model_name,
        "freeze_backbone": freeze_backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "selection_split": selection_split,
        "best_select_acc": best_select_acc,
        "final_test_result": final_test_result,
        "roc_auc_by_class": roc_auc_by_class,
        "roc_curve_path": str(roc_path),
        "history": history,
    }
    with open(output_path, "w") as f:
        json.dump(result_log, f, indent=2)

    torch.save(model.state_dict(), checkpoint_path)


    print("[RESULT]")

    if best_select_acc is None:
        print("Best Select Acc: None")
        print("[INFO] No validation split was used. Final epoch model was evaluated on test set.")
    else:
        print(f"Best Select Acc: {best_select_acc:.4f}")
    
    print(f"Final Test Domain Acc: {final_test_result['domain_acc']:.4f}")

    for d, name in enumerate(domains):
        acc = final_test_result["domain_acc_by_class"][d]
        pred_ratio = final_test_result["pred_ratio"][d]

        if acc is not None:
            print(f"{name} Test Domain Acc: {acc:.4f}")
        else:
            print(f"{name} Test Domain Acc: None")

        print(f"Pred {name} Ratio: {pred_ratio:.4f}")

    print("Test Confusion Matrix:")
    print(final_test_result["confusion_matrix"])
    print("[INFO] Saved result to:", output_path)
    print("[INFO] Saved checkpoint to:", checkpoint_path)


if __name__ == "__main__":
    main()