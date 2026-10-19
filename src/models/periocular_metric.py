from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class PeriocularMetricEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained_imagenet: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_imagenet else None
        backbone = resnet18(weights=weights)
        input_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embedding = self.projection(features)
        return F.normalize(embedding, dim=1)
