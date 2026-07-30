import torch
import torch.nn as nn
from torchvision import models


class CXRClassifier(nn.Module):
    def __init__(
        self,
        backbone_name="densenet201",
        num_classes=2,
        pretrained=True,
        freeze_features=False,
    ):
        super().__init__()

        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)

            feature_dim = backbone.fc.in_features

            self.feature_extractor = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                backbone.avgpool,
                nn.Flatten(),
            )

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)

            feature_dim = backbone.fc.in_features

            self.feature_extractor = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                backbone.avgpool,
                nn.Flatten(),
            )

        elif backbone_name == "resnet101":
            weights = (
                models.ResNet101_Weights.IMAGENET1K_V1
                if pretrained else None
            )
            backbone = models.resnet101(weights=weights)
            feature_dim = backbone.fc.in_features

            self.feature_extractor = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                backbone.avgpool,
                nn.Flatten(),
            )

        elif backbone_name == "densenet121":
            weights = (
                models.DenseNet121_Weights.IMAGENET1K_V1
                if pretrained else None
            )
            backbone = models.densenet121(weights=weights)
            feature_dim = backbone.classifier.in_features

            self.feature_extractor = nn.Sequential(
                backbone.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        elif backbone_name == "densenet201":
            weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.densenet201(weights=weights)

            feature_dim = backbone.classifier.in_features

            self.feature_extractor = nn.Sequential(
                backbone.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.classifier_head = nn.Linear(feature_dim, num_classes)

        if freeze_features:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward_features(self, x):
        return self.feature_extractor(x)

    def classify_from_features(self, features):
        return self.classifier_head(features)

    def forward_class(self, x):
        features = self.forward_features(x)
        logits = self.classify_from_features(features)
        return logits, features

    def forward(self, x, return_features=False):
        logits, features = self.forward_class(x)

        if return_features:
            return logits, features

        return logits

    def get_parameter_groups(
        self,
        feature_lr,
        classifier_lr,
    ):
        return [
            {
                "params": (
                    self.feature_extractor.parameters()
                ),
                "lr": feature_lr,
                "initial_lr": feature_lr,
            },
            {
                "params": (
                    self.classifier_head.parameters()
                ),
                "lr": classifier_lr,
                "initial_lr": classifier_lr,
            },
        ]


def build_model(cfg):
    model_cfg = cfg["model"]

    return CXRClassifier(
        backbone_name=model_cfg.get("backbone", "densenet201"),
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        freeze_features=model_cfg.get("freeze_features", False),
    )