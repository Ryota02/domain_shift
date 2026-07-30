from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns


def _ensure_parent_dir(save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path

def _epochs_from_values(values):
    return list(range(1, len(values) + 1))

def _get_epochs_from_history(history, key):
    """
    historyがdict of lists形式であることを想定する。

    例:
    history = {
        "ada_loss": [0.9, 0.8, 0.7],
        "source_cls_loss": [0.5, 0.4, 0.3],
    }
    """

    if key not in history:
        raise KeyError(f"{key} not found in history.")

    return list(range(1, len(history[key]) + 1))


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_labels,
    save_path,
    normalize=False,
):
    save_path = _ensure_parent_dir(save_path)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(
                cm,
                dtype=float,
            ),
            where=row_sums != 0,
        )
        fmt = ".2f"
        title = "Normalized confusion matrix"
    else:
        fmt = "d"
        title = "Confusion matrix"

    figure_size = max(6, len(class_labels))
    plt.figure(figsize=(figure_size, figure_size))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_source_pretrain_history(history, save_path):
    """
    Source pretraining用のloss / accuracyを保存する。

    あなたのfit_source_pretrain()のhistory形式に対応:
    history = {
        "source_train_loss": [],
        "source_train_accuracy": [],
        "source_val_loss": [],
        "source_val_accuracy": [],
    }
    """

    save_path = _ensure_parent_dir(save_path)

    if len(history.get("source_train_loss", [])) == 0:
        print("[WARN] Empty source pretrain history. Skip plotting.")
        return

    epochs = _get_epochs_from_history(history, "source_train_loss")

    # -------------------------
    # Loss
    # -------------------------
    plt.figure(figsize=(8, 6))

    if "source_train_loss" in history:
        plt.plot(
            epochs,
            history["source_train_loss"],
            label="Source train loss",
        )

    if "source_val_loss" in history:
        plt.plot(
            epochs,
            history["source_val_loss"],
            label="Source validation loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Source pretraining loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # -------------------------
    # Accuracy
    # -------------------------
    acc_save_path = save_path.with_name(
        save_path.stem + "_accuracy" + save_path.suffix
    )

    plt.figure(figsize=(8, 6))

    if "source_train_accuracy" in history:
        plt.plot(
            epochs,
            history["source_train_accuracy"],
            label="Source train accuracy",
        )

    if "source_val_accuracy" in history:
        plt.plot(
            epochs,
            history["source_val_accuracy"],
            label="Source validation accuracy",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Source pretraining accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_save_path, dpi=300)
    plt.close()


def plot_ada_losses(history, save_path):
    """
    ADA学習中のlossを1枚にまとめて保存する。

    あなたのfit_supervised_ada()のhistory形式に対応:
    history = {
        "ada_loss": [],
        "source_cls_loss": [],
        "target_cls_loss": [],
        "domain_loss": [],
        "source_val_loss": [],
    }
    """

    save_path = _ensure_parent_dir(save_path)

    if len(history.get("ada_loss", [])) == 0:
        print("[WARN] Empty ADA history. Skip plot_ada_losses.")
        return

    epochs = _get_epochs_from_history(history, "ada_loss")

    plt.figure(figsize=(8, 6))

    if "ada_loss" in history:
        plt.plot(
            epochs,
            history["ada_loss"],
            label=r"$L_{total}$",
        )

    if "source_cls_loss" in history:
        plt.plot(
            epochs,
            history["source_cls_loss"],
            label=r"$L_{cls}^{s}$",
        )

    if "target_cls_loss" in history:
        plt.plot(
            epochs,
            history["target_cls_loss"],
            label=r"$L_{cls}^{t}$",
        )

    if "domain_loss" in history:
        plt.plot(
            epochs,
            history["domain_loss"],
            label=r"$L_{domain}$ / $L_{adv}$",
        )

    if "source_val_loss" in history:
        plt.plot(
            epochs,
            history["source_val_loss"],
            label="Source validation loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ADA training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ada_accuracies(history, save_path):
    """
    ADA学習中のaccuracyを1枚にまとめて保存する。
    """

    save_path = _ensure_parent_dir(save_path)

    if len(history.get("source_accuracy", [])) == 0:
        print("[WARN] Empty ADA history. Skip plot_ada_accuracies.")
        return

    epochs = _get_epochs_from_history(history, "source_accuracy")

    plt.figure(figsize=(8, 6))

    if "source_accuracy" in history:
        plt.plot(
            epochs,
            history["source_accuracy"],
            label="Source accuracy",
        )

    if "target_adapt_accuracy" in history:
        plt.plot(
            epochs,
            history["target_adapt_accuracy"],
            label="Target adapt accuracy",
        )

    if "source_val_accuracy" in history:
        plt.plot(
            epochs,
            history["source_val_accuracy"],
            label="Source validation accuracy",
        )

    if "domain_accuracy" in history:
        plt.plot(
            epochs,
            history["domain_accuracy"],
            label="Domain discriminator accuracy",
        )

        plt.axhline(
            y=0.5,
            linestyle="--",
            linewidth=1,
            label="Random domain accuracy",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ADA training accuracies")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_domain_accuracy(history, save_path):
    """
    Domain Discriminator accuracyだけを保存する。

    Domain accuracyが0.5に近いほど、
    Source/Targetを識別しにくい特徴になっている可能性がある。
    """

    save_path = _ensure_parent_dir(save_path)

    if "domain_accuracy" not in history:
        print("[WARN] 'domain_accuracy' not found. Skip plot_domain_accuracy.")
        return

    if len(history["domain_accuracy"]) == 0:
        print("[WARN] Empty domain_accuracy. Skip plot_domain_accuracy.")
        return

    epochs = _get_epochs_from_history(history, "domain_accuracy")

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs,
        history["domain_accuracy"],
        label="Domain discriminator accuracy",
    )

    plt.axhline(
        y=0.5,
        linestyle="--",
        linewidth=1,
        label="Random guess",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Domain discriminator accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ada_loss_components_separately(history, output_dir):
    """
    ADAの各lossを個別に保存する。

    出力:
      - ada_loss.png
      - source_cls_loss.png
      - target_cls_loss.png
      - domain_loss.png
      - source_val_loss.png
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(history.get("ada_loss", [])) == 0:
        print("[WARN] Empty ADA history. Skip separate loss plots.")
        return

    epochs = _get_epochs_from_history(history, "ada_loss")

    targets = {
        "ada_loss": r"$L_{total}$",
        "source_cls_loss": r"$L_{cls}^{s}$",
        "target_cls_loss": r"$L_{cls}^{t}$",
        "domain_loss": r"$L_{domain}$ / $L_{adv}$",
        "source_val_loss": "Source validation loss",
    }

    for key, label in targets.items():
        if key not in history:
            continue

        plt.figure(figsize=(8, 6))
        plt.plot(
            epochs,
            history[key],
            label=label,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}.png", dpi=300)
        plt.close()


def plot_ada_accuracy_components_separately(history, output_dir):
    """
    ADAの各accuracyを個別に保存する。

    出力:
      - source_accuracy.png
      - target_adapt_accuracy.png
      - source_val_accuracy.png
      - domain_accuracy.png
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(history.get("source_accuracy", [])) == 0:
        print("[WARN] Empty ADA history. Skip separate accuracy plots.")
        return

    epochs = _get_epochs_from_history(history, "source_accuracy")

    targets = {
        "source_accuracy": "Source accuracy",
        "target_adapt_accuracy": "Target adapt accuracy",
        "source_val_accuracy": "Source validation accuracy",
        "domain_accuracy": "Domain discriminator accuracy",
    }

    for key, label in targets.items():
        if key not in history:
            continue

        plt.figure(figsize=(8, 6))
        plt.plot(
            epochs,
            history[key],
            label=label,
        )

        if key == "domain_accuracy":
            plt.axhline(
                y=0.5,
                linestyle="--",
                linewidth=1,
                label="Random guess",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(label)
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}.png", dpi=300)
        plt.close()


def plot_all_ada_history(history, output_dir):
    """
    ADA historyから主要なグラフをまとめて保存する。
    train_ada.py側ではこの関数を呼ぶだけでもよい。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ada_losses(
        history=history,
        save_path=output_dir / "ada_losses.png",
    )

    plot_ada_accuracies(
        history=history,
        save_path=output_dir / "ada_accuracies.png",
    )

    plot_domain_accuracy(
        history=history,
        save_path=output_dir / "domain_accuracy.png",
    )

    plot_ada_loss_components_separately(
        history=history,
        output_dir=output_dir / "ada_loss_components",
    )

    plot_ada_accuracy_components_separately(
        history=history,
        output_dir=output_dir / "ada_accuracy_components",
    )

def plot_daln_loss_curves(history, save_path):
    """
    DALNのloss curveを保存する。

    保存対象:
      - train_loss
      - cls_loss
      - discrepancy_loss
    """

    save_path = _ensure_parent_dir(save_path)

    plt.figure(figsize=(8, 5))

    if len(history.get("train_loss", [])) > 0:
        values = history["train_loss"]
        plt.plot(_epochs_from_values(values), values, label="Total loss")

    if len(history.get("cls_loss", [])) > 0:
        values = history["cls_loss"]
        plt.plot(_epochs_from_values(values), values, label="Classification loss")

    if len(history.get("discrepancy_loss", [])) > 0:
        values = history["discrepancy_loss"]
        plt.plot(_epochs_from_values(values), values, label="Discrepancy loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DALN training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_daln_accuracy_curves(history, save_path):
    """
    DALNのaccuracy curveを保存する。

    保存対象:
      - source_acc
      - target_val_acc がある場合
    """

    save_path = _ensure_parent_dir(save_path)

    plt.figure(figsize=(8, 5))

    if len(history.get("source_acc", [])) > 0:
        values = history["source_acc"]
        plt.plot(_epochs_from_values(values), values, label="Source train accuracy")

    if len(history.get("target_val_acc", [])) > 0:
        values = history["target_val_acc"]
        plt.plot(_epochs_from_values(values), values, label="Target validation accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("DALN accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_daln_auc_curve(history, save_path):
    """
    DALNのvalidation AUC curveを保存する。
    target_val_auc がない場合は何も保存しない。
    """

    values = history.get("target_val_auc", [])

    values = [v for v in values if v is not None]

    if len(values) == 0:
        return None

    save_path = _ensure_parent_dir(save_path)

    plt.figure(figsize=(8, 5))
    plt.plot(_epochs_from_values(values), values, label="Target validation AUC")

    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("DALN validation AUC")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return str(save_path)


def plot_daln_training_curves(history, output_dir):
    """
    DALNのtraining curveをまとめて保存する。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_path = output_dir / "daln_loss_curves.png"
    acc_path = output_dir / "daln_accuracy_curves.png"
    auc_path = output_dir / "daln_auc_curve.png"

    plot_daln_loss_curves(history, loss_path)
    plot_daln_accuracy_curves(history, acc_path)
    saved_auc_path = plot_daln_auc_curve(history, auc_path)

    saved_paths = {
        "loss_curve": str(loss_path),
        "accuracy_curve": str(acc_path),
        "auc_curve": saved_auc_path,
    }

    return saved_paths

def plot_roc_curve(y_true, y_proba, positive_class, class_to_idx, save_path):
    """
    2値分類のROCカーブを描画して保存する。

    y_true: 正解ラベル配列 (class index)
    y_proba: softmax確率配列 shape=(N, num_classes)
    positive_class: 陽性クラス名 (cfg["positive_class"])
    class_to_idx: {クラス名: index} の辞書
    save_path: 保存先パス (例: figure_dir / "roc_curve.png")
    """
    pos_idx = class_to_idx[positive_class]

    # 正解ラベルを陽性=1/陰性=0の2値に変換
    y_true_binary = (y_true == pos_idx).astype(int)
    # 陽性クラスの予測確率
    y_score = y_proba[:, pos_idx]

    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
              label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": roc_auc}

def plot_multiclass_roc_curve(
    y_true,
    y_proba,
    class_names,
    save_path,
):
    save_path = _ensure_parent_dir(save_path)
    num_classes = len(class_names)

    y_true_binary = label_binarize(y_true,classes=np.arange(num_classes))

    auc_values = {}

    plt.figure(figsize=(11, 8))

    for class_id, class_name in enumerate(class_names):
        binary_true = y_true_binary[:,class_id]

        if np.unique(binary_true).size < 2:
            auc_values[class_name] = None
            continue

        fpr, tpr, _ = roc_curve(binary_true,y_proba[:, class_id])
        class_auc = auc(fpr, tpr)
        auc_values[class_name] = float(class_auc)

        plt.plot(
            fpr,
            tpr,
            linewidth=1.5,
            label=(f"{class_name} (AUC={class_auc:.3f})"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "auc_per_class": auc_values,
    }