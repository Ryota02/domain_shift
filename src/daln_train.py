import copy
from itertools import cycle

import torch
import torch.nn as nn

from src.daln import build_daln_discrepancy
from src.evaluate import evaluate_model


def adjust_learning_rate(optimizer, progress, alpha=10.0, beta=0.75):
    coeff = (1.0 + alpha * progress) ** (-beta)

    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["initial_lr"] * coeff


def build_daln_optimizer(model, cfg):
    train_cfg = cfg["train"]

    feature_lr = train_cfg.get("feature_lr", 1e-4)
    classifier_lr = train_cfg.get("classifier_lr", 1e-3)

    momentum = train_cfg.get("momentum", 0.9)
    weight_decay = train_cfg.get("weight_decay", 1e-3)

    if hasattr(model, "get_parameter_groups"):
        param_groups = model.get_parameter_groups(
            feature_lr=feature_lr,
            classifier_lr=classifier_lr,
        )
    else:
        param_groups = [
            {
                "params": model.parameters(),
                "lr": classifier_lr,
                "initial_lr": classifier_lr,
            }
        ]

    optimizer = torch.optim.SGD(
        param_groups,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    return optimizer


def train_daln_one_epoch(
    model,
    discrepancy,
    source_loader,
    target_loader,
    criterion,
    optimizer,
    device,
    lambda_nwd,
    epoch,
    epochs,
    use_lr_annealing=True,
):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_discrepancy_loss = 0.0
    total_correct = 0
    total_source = 0

    target_iter = cycle(target_loader)
    num_steps = len(source_loader)

    for step, (source_images, source_labels) in enumerate(source_loader):
        progress = ((epoch - 1) * num_steps + step) / float(epochs * num_steps)

        if use_lr_annealing:
            adjust_learning_rate(optimizer, progress)

        target_images, _ = next(target_iter)

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)

        all_images = torch.cat([source_images, target_images], dim=0)
        num_source = source_images.size(0)

        all_logits, all_features = model(all_images, return_features=True)
        source_logits = all_logits[:num_source]

        cls_loss = criterion(source_logits, source_labels)

        nwd = discrepancy(all_features)

        discrepancy_loss = -nwd
        transfer_loss = lambda_nwd * discrepancy_loss
        loss = cls_loss + transfer_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = source_logits.argmax(dim=1)

        total_loss += loss.item() * num_source
        total_cls_loss += cls_loss.item() * num_source
        total_discrepancy_loss += discrepancy_loss.item() * num_source
        total_correct += (preds == source_labels).sum().item()
        total_source += num_source

    return {
        "loss": total_loss / total_source,
        "cls_loss": total_cls_loss / total_source,
        "discrepancy_loss": total_discrepancy_loss / total_source,
        "source_acc": total_correct / total_source,
    }


def fit_daln(
    model,
    loaders,
    cfg,
    device,
):
    train_cfg = cfg["train"]

    epochs = train_cfg.get("epochs", 30)
    lambda_nwd = train_cfg.get("lambda_nwd", 1.0)
    use_lr_annealing = train_cfg.get("use_lr_annealing", True)

    criterion = nn.CrossEntropyLoss()

    optimizer = build_daln_optimizer(
        model=model,
        cfg=cfg,
    )

    discrepancy = build_daln_discrepancy(
        model=model,
        cfg=cfg,
    ).to(device)

    source_loader = loaders["source_train"]
    target_loader = loaders["target_adapt"]
    target_test_loader = loaders["target_test"]

    target_val_loader = loaders.get("target_val", None)

    history = {
        "train_loss": [],
        "cls_loss": [],
        "discrepancy_loss": [],
        "source_acc": [],
        "target_val_acc": [],
        "target_val_auc": [],
    }

    best_state_dict = None
    best_val_acc = None

    for epoch in range(1, epochs + 1):
        train_result = train_daln_one_epoch(
            model=model,
            discrepancy=discrepancy,
            source_loader=source_loader,
            target_loader=target_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lambda_nwd=lambda_nwd,
            epoch=epoch,
            epochs=epochs,
            use_lr_annealing=use_lr_annealing,
        )

        history["train_loss"].append(train_result["loss"])
        history["cls_loss"].append(train_result["cls_loss"])
        history["discrepancy_loss"].append(train_result["discrepancy_loss"])
        history["source_acc"].append(train_result["source_acc"])

        if target_val_loader is not None:
            val_result = evaluate_model(
                model=model,
                loader=target_val_loader,
                criterion=criterion,
                device=device,
            )

            val_acc = val_result["accuracy"]

            try:
                val_auc = val_result["metrics"]["roc_auc"]
            except Exception:
                val_auc = None

            history["target_val_acc"].append(val_acc)
            history["target_val_auc"].append(val_auc)

            auc_text = "None" if val_auc is None else f"{val_auc:.4f}"

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Loss={train_result['loss']:.4f} "
                f"ClsLoss={train_result['cls_loss']:.4f} "
                f"DiscLoss={train_result['discrepancy_loss']:.4f} "
                f"SourceAcc={train_result['source_acc']:.4f} "
                f"ValAcc={val_acc:.4f} "
                f"ValAUC={auc_text}"
            )

            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())

        else:
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Loss={train_result['loss']:.4f} "
                f"ClsLoss={train_result['cls_loss']:.4f} "
                f"DiscLoss={train_result['discrepancy_loss']:.4f} "
                f"SourceAcc={train_result['source_acc']:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        selection_split = "target_val"
    else:
        best_state_dict = copy.deepcopy(model.state_dict())
        selection_split = "last_epoch"

    final_test_result = evaluate_model(
        model=model,
        loader=target_test_loader,
        criterion=criterion,
        device=device,
    )

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "selection_split": selection_split,
        "final_test_result": final_test_result,
        "best_state_dict": best_state_dict,
    }