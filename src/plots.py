from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def _ensure_parent_dir(save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


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
    """
    Confusion matrixを保存する。
    """

    save_path = _ensure_parent_dir(save_path)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized confusion matrix"
    else:
        fmt = "d"
        title = "Confusion matrix"

    plt.figure(figsize=(6, 5))

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