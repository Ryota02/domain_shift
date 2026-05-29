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