import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_binary_metrics(
    y_true,
    y_pred,
    y_proba,
    class_to_idx,
    positive_class="PNEUMONIA",
):
    pos_label = class_to_idx[positive_class]

    neg_candidates = [
        idx
        for cls, idx in class_to_idx.items()
        if idx != pos_label
    ]
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

def compute_multiclass_metrics(
    y_true,
    y_pred,
    y_proba,
    class_names,
):
    labels = np.arange(len(class_names))

    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    weighted_p, weighted_r, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )
    class_p, class_r, class_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )

    y_true_bin = label_binarize(
        y_true,
        classes=labels,
    )

    per_class = {}
    auc_values = []

    for class_id, class_name in enumerate(class_names):
        binary_true = y_true_bin[:, class_id]

        if np.unique(binary_true).size < 2:
            class_auc = None
        else:
            class_auc = float(
                roc_auc_score(
                    binary_true,
                    y_proba[:, class_id],
                )
            )
            auc_values.append(class_auc)

        per_class[class_name] = {
            "precision": float(class_p[class_id]),
            "recall": float(class_r[class_id]),
            "f1": float(class_f1[class_id]),
            "support": int(support[class_id]),
            "auc_ovr": class_auc,
        }

    return {
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "macro_auc_ovr": (
            float(np.mean(auc_values))
            if auc_values else None
        ),
        "per_class": per_class,
    }


def print_multiclass_metrics(metrics):
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        "Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(
        f"Weighted F1: "
        f"{metrics['weighted_f1']:.4f}"
    )

    if metrics["macro_auc_ovr"] is not None:
        print(
            f"Macro AUC OvR: "
            f"{metrics['macro_auc_ovr']:.4f}"
        )

def print_metrics(metrics):
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"PPV: {metrics['ppv']:.4f}")
    print(f"NPV: {metrics['npv']:.4f}")
    print(f"F-Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")


def print_classification_report(
    y_true,
    y_pred,
    class_names,
):
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )