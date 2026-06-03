import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, densenet201, DenseNet201_Weights


def build_model(cfg):
    model_name = cfg["model"]["name"]
    pretrained = cfg["model"]["pretrained"]
    num_classes = len(cfg["classes"])
    freeze_features = cfg["model"]["freeze_features"]

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)

        if freeze_features: 
            for param in model.parameters():
                param.requires_grad = False
        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model

    elif model_name == "densenet201":
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet201(weights=weights)

        if freeze_features: 
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")

            