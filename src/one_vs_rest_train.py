import copy

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true, y_probability, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_probability = np.asarray(y_probability, dtype=np.float64)
    y_pred = (y_probability >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(y_true, y_probability, method="max_f1", fixed_threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_probability = np.asarray(y_probability, dtype=np.float64)

    if method == "fixed":
        return float(fixed_threshold)

    if method == "max_f1":
        precision, recall, thresholds = precision_recall_curve(
            y_true,
            y_probability,
        )

        if len(thresholds) == 0:
            return 0.5

        f1_values = (
            2.0 * precision[:-1] * recall[:-1]
            / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        )

        return float(thresholds[int(np.argmax(f1_values))])

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_probability)
        return float(thresholds[int(np.argmax(tpr - fpr))])

    raise ValueError(
        f"Unsupported threshold method: {method}. "
        "Use fixed, max_f1, or youden."
    )


def build_optimizer(model, cfg):
    train_cfg = cfg["train"]
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()

    common = {
        "lr": float(train_cfg.get("learning_rate", 1e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
    }

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **common)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=float(train_cfg.get("momentum", 0.9)),
            **common,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        predictions = (torch.sigmoid(logits) >= 0.5).long()

        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels.long()).sum().item()
        total_samples += images.size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_probability = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            logits = model(images)
            loss = criterion(logits, labels_device)
            probabilities = torch.sigmoid(logits)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            y_true.append(labels.cpu().numpy())
            y_probability.append(probabilities.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_probability = np.concatenate(y_probability)

    return {
        "loss": total_loss / max(total_samples, 1),
        "y_true": y_true,
        "y_probability": y_probability,
        "metrics": compute_binary_metrics(
            y_true,
            y_probability,
            threshold=threshold,
        ),
    }


def fit_one_vs_rest(model, loaders, cfg, device, pos_weight):
    train_cfg = cfg["train"]
    evaluation_cfg = cfg.get("evaluation", {})

    epochs = int(train_cfg.get("epochs", 30))
    selection_metric = evaluation_cfg.get("selection_metric", "pr_auc")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = build_optimizer(model, cfg)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("lr_factor", 0.1)),
        patience=int(train_cfg.get("lr_patience", 3)),
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_roc_auc": [],
        "val_pr_auc": [],
        "learning_rate": [],
    }

    best_epoch = 0
    best_score = -float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
        )

        val_result = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            threshold=0.5,
        )

        scheduler.step(val_result["loss"])
        val_metrics = val_result["metrics"]

        if selection_metric == "pr_auc":
            current_score = val_metrics["pr_auc"]
        elif selection_metric == "roc_auc":
            current_score = val_metrics["roc_auc"]
        elif selection_metric == "f1":
            current_score = val_metrics["f1"]
        else:
            raise ValueError(
                f"Unsupported selection metric: {selection_metric}"
            )

        if current_score > best_score:
            best_score = float(current_score)
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        history["train_loss"].append(float(train_result["loss"]))
        history["train_accuracy"].append(float(train_result["accuracy"]))
        history["val_loss"].append(float(val_result["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_roc_auc"].append(float(val_metrics["roc_auc"]))
        history["val_pr_auc"].append(float(val_metrics["pr_auc"]))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"TrainLoss={train_result['loss']:.4f} "
            f"ValLoss={val_result['loss']:.4f} "
            f"ValROC-AUC={val_metrics['roc_auc']:.4f} "
            f"ValPR-AUC={val_metrics['pr_auc']:.4f}"
        )

    last_state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state_dict)

    validation_result = evaluate(
        model,
        loaders["val"],
        criterion,
        device,
        threshold=0.5,
    )

    threshold = select_threshold(
        validation_result["y_true"],
        validation_result["y_probability"],
        method=evaluation_cfg.get("threshold_method", "max_f1"),
        fixed_threshold=float(evaluation_cfg.get("fixed_threshold", 0.5)),
    )

    validation_result["metrics"] = compute_binary_metrics(
        validation_result["y_true"],
        validation_result["y_probability"],
        threshold=threshold,
    )

    test_result = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        threshold=threshold,
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_state_dict": best_state_dict,
        "last_state_dict": last_state_dict,
        "threshold": threshold,
        "validation_result": validation_result,
        "test_result": test_result,
        "pos_weight": float(pos_weight),
    }
