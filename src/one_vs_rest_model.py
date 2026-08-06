import torch.nn as nn

from torchvision.models import (
    DenseNet201_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    densenet201,
    resnet18,
    resnet50,
    swin_t,
    vit_b_16,
)


class OneVsRestClassifier(nn.Module):
    """CNN／ViT共通の1クラス対その他分類モデル．"""

    def __init__(self, backbone_name, pretrained=True, dropout=0.0):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name == "densenet201":
            weights = DenseNet201_Weights.DEFAULT if pretrained else None
            self.backbone = densenet201(weights=weights)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif backbone_name == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = vit_b_16(weights=weights)
            feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()

        elif backbone_name == "swin_t":
            weights = Swin_T_Weights.DEFAULT if pretrained else None
            self.backbone = swin_t(weights=weights)
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                "Use resnet18, resnet50, densenet201, vit_b_16, or swin_t."
            )

        self.dropout = nn.Dropout(float(dropout))
        self.classifier_head = nn.Linear(feature_dim, 1)

    def forward_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        features = self.forward_features(images)
        return self.classifier_head(self.dropout(features)).squeeze(1)


def build_one_vs_rest_model(cfg):
    model_cfg = cfg["model"]

    return OneVsRestClassifier(
        backbone_name=model_cfg["backbone"],
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
