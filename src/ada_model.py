import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import densenet201, DenseNet201_Weights


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DenseNet201FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = densenet201(weights=weights)

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=512, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, features):
        return self.net(features)


class SupervisedADAModel(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_dim=1920,
        pretrained=True,
        grl_lambda=1.0,
    ):
        super().__init__()

        self.feature_extractor = DenseNet201FeatureExtractor(
            pretrained=pretrained
        )

        self.classifier = ClassifierHead(
            feature_dim=feature_dim,
            num_classes=num_classes
        )

        self.grl = GradientReversalLayer(lambda_=grl_lambda)

        self.domain_discriminator = DomainDiscriminator(
            feature_dim=feature_dim
        )
        
    def forward(self, x):
        logits, _ = self.forward_class(x)
        return logits

    def forward_class(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def forward_domain(self, features):
        reversed_features = self.grl(features)
        domain_logits = self.domain_discriminator(reversed_features)
        return domain_logits