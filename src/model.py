import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes: int, pretrained: bool = True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model