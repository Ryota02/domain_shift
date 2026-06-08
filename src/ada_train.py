import copy
import torch
from tqdm import tqdm
from itertools import cycle


def train_source_pretrain_one_epoch(
    model,
    source_loader,
    criterion_cls,
    optimizer,
    device,
    epoch,
    epochs,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
        source_loader,
        desc=f"Source Pretrain {epoch+1}/{epochs}"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model.forward_class(images)
        loss = criterion_cls(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def validate_classification(
    model,
    loader,
    criterion_cls,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model.forward_class(images)
            loss = criterion_cls(logits, labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def fit_source_pretrain(
    model,
    source_train_loader,
    source_val_loader,
    criterion_cls,
    optimizer,
    device,
    epochs,
):
    history = {
        "source_train_loss": [],
        "source_train_accuracy": [],
        "source_val_loss": [],
        "source_val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(epochs):
        train_result = train_source_pretrain_one_epoch(
            model=model,
            source_loader=source_train_loader,
            criterion_cls=criterion_cls,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
        )

        val_result = validate_classification(
            model=model,
            loader=source_val_loader,
            criterion_cls=criterion_cls,
            device=device,
        )

        history["source_train_loss"].append(train_result["loss"])
        history["source_train_accuracy"].append(train_result["accuracy"])
        history["source_val_loss"].append(val_result["loss"])
        history["source_val_accuracy"].append(val_result["accuracy"])

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_val_acc = val_result["accuracy"]
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"[Source Pretrain] Epoch {epoch+1}/{epochs} "
            f"Train Loss: {train_result['loss']:.4f}, "
            f"Train Acc: {train_result['accuracy']:.4f}, "
            f"Val Loss: {val_result['loss']:.4f}, "
            f"Val Acc: {val_result['accuracy']:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history, best_val_acc, best_val_loss


def train_ada_one_epoch(
    model,
    source_loader,
    target_loader,
    criterion_cls,
    criterion_domain,
    optimizer,
    device,
    epoch,
    epochs,
    lambda_target_cls=1.0,
    beta_domain=0.1,
):
    model.train()

    total_loss = 0.0
    total_src_cls_loss = 0.0
    total_tgt_cls_loss = 0.0
    total_domain_loss = 0.0

    src_correct = 0
    tgt_correct = 0
    domain_correct = 0

    src_total = 0
    tgt_total = 0
    domain_total = 0

    target_iter = cycle(target_loader)

    for source_images, source_labels in tqdm(
        source_loader,
        desc=f"ADA {epoch+1}/{epochs}"
    ):
        target_images, target_labels = next(target_iter)

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)

        target_images = target_images.to(device)
        target_labels = target_labels.to(device)

        optimizer.zero_grad()

        source_logits, source_features = model.forward_class(source_images)
        target_logits, target_features = model.forward_class(target_images)

        source_cls_loss = criterion_cls(source_logits, source_labels)
        target_cls_loss = criterion_cls(target_logits, target_labels)

        source_domain_labels = torch.zeros(
            source_images.size(0),
            dtype=torch.long,
            device=device
        )

        target_domain_labels = torch.ones(
            target_images.size(0),
            dtype=torch.long,
            device=device
        )

        source_domain_logits = model.forward_domain(source_features)
        target_domain_logits = model.forward_domain(target_features)

        domain_logits = torch.cat(
            [source_domain_logits, target_domain_logits],
            dim=0
        )

        domain_labels = torch.cat(
            [source_domain_labels, target_domain_labels],
            dim=0
        )

        domain_loss = criterion_domain(domain_logits, domain_labels)

        loss = (
            source_cls_loss
            + lambda_target_cls * target_cls_loss
            + beta_domain * domain_loss
        )

        loss.backward()
        optimizer.step()

        source_preds = source_logits.argmax(dim=1)
        target_preds = target_logits.argmax(dim=1)
        domain_preds = domain_logits.argmax(dim=1)

        total_loss += loss.item() * source_images.size(0)
        total_src_cls_loss += source_cls_loss.item() * source_images.size(0)
        total_tgt_cls_loss += target_cls_loss.item() * target_images.size(0)
        total_domain_loss += domain_loss.item() * domain_labels.size(0)

        src_correct += (source_preds == source_labels).sum().item()
        tgt_correct += (target_preds == target_labels).sum().item()
        domain_correct += (domain_preds == domain_labels).sum().item()

        src_total += source_labels.size(0)
        tgt_total += target_labels.size(0)
        domain_total += domain_labels.size(0)

    return {
        "loss": total_loss / src_total,
        "source_cls_loss": total_src_cls_loss / src_total,
        "target_cls_loss": total_tgt_cls_loss / tgt_total,
        "domain_loss": total_domain_loss / domain_total,
        "source_accuracy": src_correct / src_total,
        "target_adapt_accuracy": tgt_correct / tgt_total,
        "domain_accuracy": domain_correct / domain_total,
    }


def fit_supervised_ada(
    model,
    source_train_loader,
    target_adapt_loader,
    source_val_loader,
    criterion_cls,
    criterion_domain,
    optimizer,
    device,
    epochs,
    lambda_target_cls=1.0,
    beta_domain=0.1,
):
    history = {
        "ada_loss": [],
        "source_cls_loss": [],
        "target_cls_loss": [],
        "domain_loss": [],
        "source_accuracy": [],
        "target_adapt_accuracy": [],
        "domain_accuracy": [],
        "source_val_loss": [],
        "source_val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(epochs):
        train_result = train_ada_one_epoch(
            model=model,
            source_loader=source_train_loader,
            target_loader=target_adapt_loader,
            criterion_cls=criterion_cls,
            criterion_domain=criterion_domain,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
            lambda_target_cls=lambda_target_cls,
            beta_domain=beta_domain,
        )

        val_result = validate_classification(
            model=model,
            loader=source_val_loader,
            criterion_cls=criterion_cls,
            device=device,
        )

        history["ada_loss"].append(train_result["loss"])
        history["source_cls_loss"].append(train_result["source_cls_loss"])
        history["target_cls_loss"].append(train_result["target_cls_loss"])
        history["domain_loss"].append(train_result["domain_loss"])
        history["source_accuracy"].append(train_result["source_accuracy"])
        history["target_adapt_accuracy"].append(train_result["target_adapt_accuracy"])
        history["domain_accuracy"].append(train_result["domain_accuracy"])
        history["source_val_loss"].append(val_result["loss"])
        history["source_val_accuracy"].append(val_result["accuracy"])

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_val_acc = val_result["accuracy"]
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"[ADA] Epoch {epoch+1}/{epochs} "
            f"Loss: {train_result['loss']:.4f}, "
            f"SrcCls: {train_result['source_cls_loss']:.4f}, "
            f"TgtCls: {train_result['target_cls_loss']:.4f}, "
            f"Domain: {train_result['domain_loss']:.4f}, "
            f"SrcAcc: {train_result['source_accuracy']:.4f}, "
            f"TgtAdaptAcc: {train_result['target_adapt_accuracy']:.4f}, "
            f"DomainAcc: {train_result['domain_accuracy']:.4f}, "
            f"ValAcc: {val_result['accuracy']:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history, best_val_acc, best_val_loss