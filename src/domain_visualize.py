import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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