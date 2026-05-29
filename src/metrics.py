import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_pred, y_proba, class_to_idx, positive_class="PNEUMONIA"):
    pos_label = class_to_idx[positive_class]

    neg_candidates = [idx for cls, idx in class_to_idx.items() if idx != pos_label]
    assert len(neg_candidates) == 1, f"Binary classification expected, got {class_to_idx}"
    neg_label = neg_candidates[0]

    y_score = y_proba[:, pos_label]

    metrics = {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=neg_label, zero_division=0),
        "ppv": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "npv": precision_score(y_true, y_pred, pos_label=neg_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
    }

    return metrics


def print_metrics(metrics):
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"感度: {metrics['sensitivity']:.4f}")
    print(f"特異度: {metrics['specificity']:.4f}")
    print(f"PPV: {metrics['ppv']:.4f}")
    print(f"NPV: {metrics['npv']:.4f}")
    print(f"F-Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")


def print_classification_report(y_true, y_pred, class_names):
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    ))