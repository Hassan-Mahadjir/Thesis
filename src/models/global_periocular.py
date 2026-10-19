from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from facenet_pytorch import InceptionResnetV1
from torchvision.models import ResNet18_Weights, resnet18


VALID_VARIANTS = {
    "global_pair",
    "local_pair",
    "global_local_pair",
}


class GlobalPeriocularVerifier(nn.Module):
    """
    Diagnostic ablation model.

    Variants
    --------
    global_pair:
        Pair head sees only:
            |G1-G2|, G1*G2

        This answers:
            "Does replacing cosine similarity with the learned pair head
             already hurt the strong global representation?"

    local_pair:
        Pair head sees only:
            |L1-L2|, L1*L2

        This answers:
            "Does the periocular representation contain useful standalone
             twin-discriminative information?"

    global_local_pair:
        Original proposed architecture:
            |G1-G2|, G1*G2, |L1-L2|, L1*L2

        This is the architecture already evaluated.
    """

    def __init__(
        self,
        variant: str = "global_local_pair",
        global_pretrained: str = "vggface2",
        local_embedding_dim: int = 256,
        local_pretrained_imagenet: bool = True,
        hidden_1: int = 512,
        hidden_2: int = 128,
        dropout: float = 0.30,
        freeze_global: bool = True,
    ):
        super().__init__()

        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Expected one of {sorted(VALID_VARIANTS)}."
            )

        self.variant = variant
        self.freeze_global = bool(freeze_global)
        self.local_embedding_dim = int(local_embedding_dim)

        self.global_backbone = InceptionResnetV1(
            pretrained=global_pretrained,
            classify=False,
        )

        if self.freeze_global:
            for parameter in self.global_backbone.parameters():
                parameter.requires_grad = False

        local_weights = (
            ResNet18_Weights.DEFAULT
            if local_pretrained_imagenet
            else None
        )

        self.local_backbone = resnet18(weights=local_weights)
        local_input_dim = self.local_backbone.fc.in_features
        self.local_backbone.fc = nn.Identity()

        self.local_projection = nn.Sequential(
            nn.Linear(local_input_dim, local_embedding_dim),
            nn.BatchNorm1d(local_embedding_dim),
            nn.PReLU(local_embedding_dim),
        )

        if variant == "global_pair":
            pair_dim = 2 * 512
        elif variant == "local_pair":
            pair_dim = 2 * local_embedding_dim
        else:
            pair_dim = 2 * 512 + 2 * local_embedding_dim

        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.PReLU(hidden_1),
            nn.Dropout(dropout),

            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.PReLU(hidden_2),
            nn.Dropout(dropout),

            nn.Linear(hidden_2, 1),
        )

        # Make unused local parameters non-trainable in the global-only test.
        if self.variant == "global_pair":
            for parameter in self.local_backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.local_projection.parameters():
                parameter.requires_grad = False

    @property
    def uses_global(self) -> bool:
        return self.variant in {"global_pair", "global_local_pair"}

    @property
    def uses_local(self) -> bool:
        return self.variant in {"local_pair", "global_local_pair"}

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_global:
            self.global_backbone.eval()

        if not self.uses_local:
            self.local_backbone.eval()
            self.local_projection.eval()

        return self

    def encode_global(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_global:
            with torch.no_grad():
                embedding = self.global_backbone(images)
        else:
            embedding = self.global_backbone(images)

        return F.normalize(embedding, dim=1)

    def encode_local(self, images: torch.Tensor) -> torch.Tensor:
        if not self.uses_local:
            # This branch is not used by global_pair. The method remains
            # available for compatibility with the shared extraction code.
            with torch.no_grad():
                embedding = self.local_backbone(images)
                embedding = self.local_projection(embedding)
        else:
            embedding = self.local_backbone(images)
            embedding = self.local_projection(embedding)

        return F.normalize(embedding, dim=1)

    def encode(
        self,
        global_images: torch.Tensor,
        local_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_embedding = self.encode_global(global_images)
        local_embedding = self.encode_local(local_images)
        return global_embedding, local_embedding

    def pair_features(
        self,
        g1: torch.Tensor,
        g2: torch.Tensor,
        l1: torch.Tensor,
        l2: torch.Tensor,
    ) -> torch.Tensor:
        global_features = torch.cat(
            [torch.abs(g1 - g2), g1 * g2],
            dim=1,
        )
        local_features = torch.cat(
            [torch.abs(l1 - l2), l1 * l2],
            dim=1,
        )

        if self.variant == "global_pair":
            return global_features

        if self.variant == "local_pair":
            return local_features

        return torch.cat(
            [global_features, local_features],
            dim=1,
        )

    def score_embeddings(
        self,
        g1: torch.Tensor,
        g2: torch.Tensor,
        l1: torch.Tensor,
        l2: torch.Tensor,
    ) -> torch.Tensor:
        features = self.pair_features(g1, g2, l1, l2)
        return self.pair_head(features).squeeze(1)

    def forward(
        self,
        global_a: torch.Tensor,
        global_b: torch.Tensor,
        local_a: torch.Tensor,
        local_b: torch.Tensor,
    ) -> torch.Tensor:
        if self.uses_global:
            global_combined = torch.cat(
                [global_a, global_b],
                dim=0,
            )
            global_embeddings = self.encode_global(global_combined)
            g1, g2 = torch.chunk(global_embeddings, 2, dim=0)
        else:
            batch = global_a.shape[0]
            device = global_a.device
            g1 = torch.zeros((batch, 512), device=device)
            g2 = torch.zeros((batch, 512), device=device)

        if self.uses_local:
            local_combined = torch.cat(
                [local_a, local_b],
                dim=0,
            )
            local_embeddings = self.encode_local(local_combined)
            l1, l2 = torch.chunk(local_embeddings, 2, dim=0)
        else:
            batch = local_a.shape[0]
            device = local_a.device
            l1 = torch.zeros(
                (batch, self.local_embedding_dim),
                device=device,
            )
            l2 = torch.zeros(
                (batch, self.local_embedding_dim),
                device=device,
            )

        return self.score_embeddings(g1, g2, l1, l2)
