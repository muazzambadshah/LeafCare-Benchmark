"""LeafCare PhytoEdgeNet with explicit variants for future controlled ablations.

The default full variant preserves the original module names and checkpoint layout.
Variant results require fresh training and are not included in the revised paper.
"""
import torch
from torch import nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return features * self.gate(features)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, features):
        summary = torch.cat(
            (features.mean(1, keepdim=True), features.amax(1, keepdim=True)), dim=1
        )
        return features * self.conv(summary).sigmoid()


class PhytoFeatures(nn.Module):
    def __init__(self, pretrained, variant="full"):
        super().__init__()
        if variant not in {"head_only", "attention", "fusion", "full"}:
            raise ValueError(f"Unknown ablation variant: {variant}")
        self.variant = variant
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights).features
        self.channel = ChannelAttention(576) if variant in {"attention", "full"} else nn.Identity()
        self.spatial = SpatialAttention() if variant in {"attention", "full"} else nn.Identity()
        self.fusion = (
            nn.Sequential(
                nn.Conv2d(624, 576, 1, bias=False),
                nn.BatchNorm2d(576),
                nn.SiLU(),
            )
            if variant in {"fusion", "full"} else nn.Identity()
        )

    def forward(self, image):
        early = None
        features = image
        for index, layer in enumerate(self.backbone):
            features = layer(features)
            if index == 8:
                early = features
        features = self.spatial(self.channel(features))
        if self.variant in {"fusion", "full"}:
            early = nn.functional.adaptive_avg_pool2d(early, features.shape[-2:])
            features = self.fusion(torch.cat((features, early), dim=1))
        return features


class PhytoEdgeNet(nn.Module):
    def __init__(self, classes=147, pretrained=True, variant="full"):
        super().__init__()
        self.variant = variant
        self.features = PhytoFeatures(pretrained, variant)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1152, classes))
        self.backbone_frozen = False

    @property
    def cam_layer(self):
        if self.variant in {"fusion", "full"}:
            return self.features.fusion[-1]
        if self.variant == "attention":
            return self.features.spatial
        return self.features.backbone[-1]

    def set_backbone_frozen(self, frozen):
        self.backbone_frozen = frozen
        self.features.backbone.requires_grad_(not frozen)
        self.features.backbone.train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.features.backbone.eval()
        return self

    def forward(self, image):
        features = self.features(image)
        pooled = torch.cat(
            (self.avgpool(features).flatten(1), self.maxpool(features).flatten(1)), dim=1
        )
        return self.classifier(pooled)


def phytoedge_net(weights=None):
    return PhytoEdgeNet(pretrained=weights is not None, variant="full")


def phytoedge_head_only(weights=None):
    return PhytoEdgeNet(pretrained=weights is not None, variant="head_only")


def phytoedge_attention(weights=None):
    return PhytoEdgeNet(pretrained=weights is not None, variant="attention")
