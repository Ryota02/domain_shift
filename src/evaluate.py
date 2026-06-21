import numpy as np
import torch


def evaluate_model(model, loader, criterion, device):
    model.eval()

    y_true = []
    y_pred = []
    y_proba = []

    test_loss = 0.0
    correct = 0
    total = 0

    misclassified = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if hasattr(model, "forward_class"):
                logits, _ = model.forward_class(images)
            else:
                logits = model(images)

            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            test_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_proba.append(probs.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append({
                        "image": images[i].cpu(),
                        "true": labels[i].cpu().item(),
                        "pred": preds[i].cpu().item(),
                    })

    return {
        "loss": test_loss / total,
        "accuracy": correct / total,
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "y_proba": np.concatenate(y_proba),
        "misclassified": misclassified,
    }


def evaluate_domain_discriminator(
    model,
    source_loader,
    target_loader,
    device,
):
    """
    Domain Discriminator の精度を評価する。

    source domain label:
        0

    target domain label:
        1

    注意:
        この関数は SupervisedADAModel を想定している。
        つまり，model.forward_class() と model.forward_domain()
        が定義されている必要がある。
    """

    if not hasattr(model, "forward_class"):
        raise ValueError(
            "evaluate_domain_discriminator requires model.forward_class(). "
            "This function is intended for ADA models."
        )

    if not hasattr(model, "forward_domain"):
        raise ValueError(
            "evaluate_domain_discriminator requires model.forward_domain(). "
            "This function is intended for ADA models."
        )

    model.eval()

    correct = 0
    total = 0

    source_correct = 0
    source_total = 0

    target_correct = 0
    target_total = 0

    y_true_domain = []
    y_pred_domain = []
    y_proba_domain = []

    with torch.no_grad():
        # -------------------------
        # Source domain evaluation
        # -------------------------
        for images, _ in source_loader:
            images = images.to(device)

            _, features = model.forward_class(images)
            domain_logits = model.forward_domain(features)

            domain_labels = torch.zeros(
                images.size(0),
                dtype=torch.long,
                device=device,
            )

            domain_probs = torch.softmax(domain_logits, dim=1)
            domain_preds = domain_probs.argmax(dim=1)

            source_correct += (domain_preds == domain_labels).sum().item()
            source_total += domain_labels.size(0)

            correct += (domain_preds == domain_labels).sum().item()
            total += domain_labels.size(0)

            y_true_domain.append(domain_labels.cpu().numpy())
            y_pred_domain.append(domain_preds.cpu().numpy())
            y_proba_domain.append(domain_probs.cpu().numpy())

        # -------------------------
        # Target domain evaluation
        # -------------------------
        for images, _ in target_loader:
            images = images.to(device)

            _, features = model.forward_class(images)
            domain_logits = model.forward_domain(features)

            domain_labels = torch.ones(
                images.size(0),
                dtype=torch.long,
                device=device,
            )

            domain_probs = torch.softmax(domain_logits, dim=1)
            domain_preds = domain_probs.argmax(dim=1)

            target_correct += (domain_preds == domain_labels).sum().item()
            target_total += domain_labels.size(0)

            correct += (domain_preds == domain_labels).sum().item()
            total += domain_labels.size(0)

            y_true_domain.append(domain_labels.cpu().numpy())
            y_pred_domain.append(domain_preds.cpu().numpy())
            y_proba_domain.append(domain_probs.cpu().numpy())

    return {
        "domain_acc": correct / total,
        "source_domain_acc": source_correct / source_total,
        "target_domain_acc": target_correct / target_total,
        "y_true_domain": np.concatenate(y_true_domain),
        "y_pred_domain": np.concatenate(y_pred_domain),
        "y_proba_domain": np.concatenate(y_proba_domain),
    }