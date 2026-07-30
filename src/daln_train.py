import copy
from itertools import cycle

import torch
import torch.nn as nn

from src.daln import build_daln_discrepancy
from src.evaluate import evaluate_model


def adjust_learning_rate(
    optimizer, 
    progress, 
    alpha=10.0, 
    beta=0.75
):
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

def extract_target_images(target_batch):
    if torch.is_tensor(target_batch):
        return target_batch

    if isinstance(target_batch, (tuple, list)):
        return target_batch[0]

    raise TypeError(
        f"Unsupported target batch type: "
        f"{type(target_batch)}"
    )

def train_source_only_one_epoch(
    model,
    source_loader,
    criterion,
    optimizer,
    device,
    epoch,
    epochs,
    iterations_per_epoch,
    use_lr_annealing=True,
):
    model.train()

    source_iter = cycle(source_loader)

    total_loss = 0.0
    total_correct = 0
    total_source = 0

    for step in range(iterations_per_epoch):
        progress = (
            (epoch - 1) * iterations_per_epoch
            + step
        ) / float(
            epochs * iterations_per_epoch
        )

        if use_lr_annealing:
            adjust_learning_rate(
                optimizer,
                progress,
            )

        images, labels = next(source_iter)

        images = images.to(device,non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            preds == labels
        ).sum().item()
        total_source += batch_size

    return {
        "loss": total_loss / total_source,
        "cls_loss": total_loss / total_source,
        "discrepancy_loss": 0.0,
        "source_acc": (
            total_correct / total_source
        ),
    }


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
    iterations_per_epoch, 
    use_lr_annealing=True,
):
    model.train()

    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_discrepancy_loss = 0.0
    total_correct = 0
    total_source = 0

    for step in range(iterations_per_epoch):
        progress = ((epoch - 1) * iterations_per_epoch + step) / float(epochs * iterations_per_epoch)

        if use_lr_annealing:
            adjust_learning_rate(optimizer, progress)

        source_images, source_labels = next(source_iter)
        target_batch = next(target_iter)
        target_images = extract_target_images(target_batch)

        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        all_images = torch.cat([source_images, target_images], dim=0)
        num_source = source_images.size(0)

        all_logits, all_features = model(all_images, return_features=True)
        
        source_logits = all_logits[:num_source]
        source_features = all_features[:num_source]
        target_features = all_features[num_source:]
        
        cls_loss = criterion(source_logits, source_labels)
        nwd = discrepancy(source_features, target_features)

        discrepancy_loss = -nwd
        transfer_loss = lambda_nwd * discrepancy_loss
        loss = cls_loss + transfer_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = source_logits.argmax(dim=1)
        batch_size = source_labels.size(0)

        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_discrepancy_loss += discrepancy_loss.item() * batch_size
        total_correct += (preds == source_labels).sum().item()
        total_source += batch_size

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

    evaluation_cfg = cfg.get("evaluation", {})

    method = cfg.get("method", "daln") # daln or source-only
    if method not in {"daln", "source_only"}: 
        raise ValueError("Unsupported method: ", method
                        )
    epochs = train_cfg.get("epochs", 30)
    lambda_nwd = train_cfg.get("lambda_nwd", 1.0)
    use_lr_annealing = train_cfg.get("use_lr_annealing", True)

    criterion = nn.CrossEntropyLoss()

    optimizer = build_daln_optimizer(
        model=model,
        cfg=cfg,
    )

    source_loader = loaders["source_train"]
    target_test_loader = loaders["target_test"]

    iterations_per_epoch = train_cfg.get("iterations_per_epoch", len(source_loader))

    target_loader = None
    discrepancy = None

    if method == "daln": 
        target_loader = loaders["target_adapt"]
        discrepancy = build_daln_discrepancy(model=model, cfg=cfg).to(device)
        
    selection_mode = evaluation_cfg.get("selection_mode", "last_epoch")

    if selection_mode not in {"target_test", "last_epoch"}: 
        raise ValueError(
            "selection_mode must be "
            "'target_test' or 'last_epoch'."
        )

    history = {
        "train_loss": [],
        "cls_loss": [],
        "discrepancy_loss": [],
        "source_acc": [],
        "target_val_acc": [],
        "target_val_auc": [],
        "target_test_acc": [],
    }

    best_state_dict = None
    best_score = None
    best_epoch = None
    best_val_acc = None

    for epoch in range(1, epochs + 1):
        if method == "daln": 
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
                iterations_per_epoch=iterations_per_epoch,
                use_lr_annealing=use_lr_annealing,
            )
        else: 
            train_result = train_source_only_one_epoch(
                model=model,
                source_loader=source_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                epochs=epochs,
                iterations_per_epoch=iterations_per_epoch,
                use_lr_annealing=use_lr_annealing,
            )

        history["train_loss"].append(train_result["loss"])
        history["cls_loss"].append(train_result["cls_loss"])
        history["discrepancy_loss"].append(train_result["discrepancy_loss"])
        history["source_acc"].append(train_result["source_acc"])

        if selection_mode == "target_test":
            test_result = evaluate_model(
                model=model, 
                loader=target_test_loader,
                criterion=criterion,
                device=device
            )
            test_acc = test_result["accuracy"]
            history["target_test_acc"].append(test_acc)
            current_score = test_acc
        else: 
            current_score = float(epoch)

        if (best_score is None or current_score > best_score): 
            best_score = float(current_score)
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Loss={train_result['loss']:.4f} "
            f"ClsLoss={train_result['cls_loss']:.4f} "
            f"DiscLoss={train_result['discrepancy_loss']:.4f} "
            f"SourceAcc={train_result['source_acc']:.4f} "
            f"TargetTestAcc={test_acc:.4f}"
        )

    if best_state_dict is None: 
        best_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = epochs
        
    model.load_state_dict(best_state_dict)

    final_test_result = evaluate_model(
        model=model,
        loader=target_test_loader,
        criterion=criterion,
        device=device,
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_acc": (
            best_score
            if selection_mode == "target_test"
            else None
        ),
        "selection_split": selection_mode,
        "final_test_result": final_test_result,
        "best_state_dict": best_state_dict,
    }