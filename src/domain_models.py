import torch.nn as nn
from torchvision import models


def build_domain_classifier(
    model_name,
    num_domains,
    freeze_backbone=False,
):
    """
    画像を入力し，domain labelを予測するCNNを作る。
    """

    if model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        in_features = model.fc.in_features

        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False

        model.fc = nn.Linear(in_features, num_domains)
        return model

    if model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        in_features = model.fc.in_features

        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False

        model.fc = nn.Linear(in_features, num_domains)
        return model

    if model_name == "vgg16":
        model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )

        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_domains)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def build_domain_classifier_without_weights(model_name, num_domains):
    """
    checkpoint読み込み用。
    ImageNet weightsを読み込まず，構造だけ作る。
    """

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_domains)
        return model

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_domains)
        return model

    if model_name == "vgg16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_domains)
        return model

    raise ValueError(f"Unsupported model: {model_name}")