from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_multiclass_roc_curve(
    y_true,
    y_score,
    class_names,
    save_path,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    num_classes = len(class_names)
    classes = list(range(num_classes))

    y_true_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(7, 6))

    auc_dict = {}

    for class_idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(
            y_true_bin[:, class_idx],
            y_score[:, class_idx],
        )

        roc_auc = auc(fpr, tpr)
        auc_dict[class_name] = float(roc_auc)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_name} AUC = {roc_auc:.3f}",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Domain classification ROC curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return auc_dict


def extract_resnet_features(model, images):
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    return x


def extract_vgg_features(model, images):
    x = model.features(images)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    for layer in model.classifier[:-1]:
        x = layer(x)

    return x


def extract_cnn_domain_features(
    model,
    model_name,
    loader,
    device,
):
    model.eval()

    all_features = []
    all_domain_labels = []
    all_disease_labels = []

    with torch.no_grad():
        for images, domain_labels, disease_labels in loader:
            images = images.to(device)

            if model_name.startswith("resnet"):
                features = extract_resnet_features(model, images)
            elif model_name == "vgg16":
                features = extract_vgg_features(model, images)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            all_features.append(features.cpu().numpy())
            all_domain_labels.append(domain_labels.numpy())
            all_disease_labels.append(disease_labels.numpy())

    features = np.concatenate(all_features, axis=0)
    domain_labels = np.concatenate(all_domain_labels, axis=0)
    disease_labels = np.concatenate(all_disease_labels, axis=0)

    return features, domain_labels, disease_labels


def run_tsne(features):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    return tsne.fit_transform(features)


def plot_tsne(
    tsne_features,
    labels,
    label_names,
    title,
    save_path,
):
    plt.figure(figsize=(8, 7))

    for label_id, label_name in enumerate(label_names):
        mask = labels == label_id

        plt.scatter(
            tsne_features[mask, 0],
            tsne_features[mask, 1],
            s=10,
            alpha=0.7,
            label=label_name,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()