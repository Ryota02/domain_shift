import copy
import torch

from src.domain_evaluate import evaluate_domain_classifier


def train_domain_classifier_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, domain_labels, _ in loader:
        images = images.to(device)
        domain_labels = domain_labels.to(device)

        logits = model(images)
        loss = criterion(logits, domain_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * domain_labels.size(0)
        correct += (preds == domain_labels).sum().item()
        total += domain_labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def fit_domain_classifier(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    num_domains,
):
    history = {
        "train_loss": [],
        "train_domain_acc": [],
        "val_loss": [],
        "val_domain_acc": [],
    }

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_result = train_domain_classifier_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_result = evaluate_domain_classifier(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_domains=num_domains,
        )

        history["train_loss"].append(train_result["loss"])
        history["train_domain_acc"].append(train_result["accuracy"])
        history["val_loss"].append(val_result["loss"])
        history["val_domain_acc"].append(val_result["domain_acc"])

        by_class_text = " ".join([
            f"Domain{d}Acc={val_result['domain_acc_by_class'][d]:.4f}"
            for d in range(num_domains)
            if val_result["domain_acc_by_class"][d] is not None
        ])

        pred_ratio_text = " ".join([
            f"PredDomain{d}={val_result['pred_ratio'][d]:.4f}"
            for d in range(num_domains)
        ])

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"TrainLoss={train_result['loss']:.4f} "
            f"TrainAcc={train_result['accuracy']:.4f} "
            f"ValLoss={val_result['loss']:.4f} "
            f"ValAcc={val_result['domain_acc']:.4f} "
            f"{by_class_text} "
            f"{pred_ratio_text}"
        )

        if val_result["domain_acc"] > best_val_acc:
            best_val_acc = val_result["domain_acc"]
            best_state_dict = copy.deepcopy(model.state_dict())

    return history, best_state_dict, best_val_acc