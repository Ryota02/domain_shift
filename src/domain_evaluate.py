import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_domain_classifier(
    model,
    loader,
    criterion,
    device,
    num_domains,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0 for _ in range(num_domains)]
    class_total = [0 for _ in range(num_domains)]
    pred_count = [0 for _ in range(num_domains)]

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for images, domain_labels, _ in loader:
            images = images.to(device)
            domain_labels = domain_labels.to(device)

            logits = model(images)
            loss = criterion(logits, domain_labels)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item() * domain_labels.size(0)
            correct += (preds == domain_labels).sum().item()
            total += domain_labels.size(0)

            y_true.extend(domain_labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs.cpu().tolist())

            for d in range(num_domains):
                mask = domain_labels == d
                class_correct[d] += (preds[mask] == d).sum().item()
                class_total[d] += mask.sum().item()
                pred_count[d] += (preds == d).sum().item()

    domain_acc_by_class = {}

    for d in range(num_domains):
        if class_total[d] == 0:
            domain_acc_by_class[d] = None
        else:
            domain_acc_by_class[d] = class_correct[d] / class_total[d]

    valid_accs = [
        acc for acc in domain_acc_by_class.values()
        if acc is not None
    ]

    macro_domain_acc = sum(valid_accs) / len(valid_accs)

    pred_ratio = {
        d: pred_count[d] / total
        for d in range(num_domains)
    }

    return {
        "loss": total_loss / total,
        "domain_acc": correct / total,
        "macro_domain_acc": macro_domain_acc,
        "domain_acc_by_class": domain_acc_by_class,
        "pred_ratio": pred_ratio,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }