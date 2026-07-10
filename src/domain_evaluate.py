import torch
from sklearn.metrics import confusion_matrix, classification_report


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

    with torch.no_grad():
        for images, domain_labels, _ in loader:
            images = images.to(device)
            domain_labels = domain_labels.to(device)

            logits = model(images)
            loss = criterion(logits, domain_labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * domain_labels.size(0)
            correct += (preds == domain_labels).sum().item()
            total += domain_labels.size(0)

            y_true.extend(domain_labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

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

    pred_ratio = {
        d: pred_count[d] / total
        for d in range(num_domains)
    }

    cm = confusion_matrix(y_true, y_pred).tolist()

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return {
        "loss": total_loss / total,
        "domain_acc": correct / total,
        "domain_acc_by_class": domain_acc_by_class,
        "pred_ratio": pred_ratio,
        "confusion_matrix": cm,
        "classification_report": report,
    }