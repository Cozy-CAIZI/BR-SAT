from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models import efficientnet_b0


DIAGNOSIS_NAMES = ("negative", "weak_positive", "positive")
REACTION_NAMES = ("no_agglutination", "weak_agglutination", "strong_agglutination")
RBPT_MASK_NAMES = (
    "rbt_reaction_zone",
    "rbt_agglutinate_weak",
    "rbt_agglutinate_strong",
    "artifact_interference",
)
SAT_MASK_NAMES = (
    "sat_bottom_zone",
    "sat_agglutinate_weak",
    "sat_agglutinate_strong",
    "artifact_interference",
)
NUM_MODALITIES = 5


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int = 256
    transformer_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.20
    num_centers: int = 1
    pretrained_backbone: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(inputs, self.alpha)


class EfficientNetFeatureEncoder(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 truncated at stride 16."""

    output_channels = 112

    def __init__(self, pretrained: bool):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = nn.Sequential(*list(backbone.features.children())[:6])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class MultiLabelSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = max(64, in_channels // 2)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.layers(features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class AIRBPTSATMultimodalModel(nn.Module):
    """Five-image patient-level classifier with expert-mask auxiliary heads.

    RBPT and the four SAT dilution images are encoded separately. SAT weights are
    shared across dilutions. Missing modalities are hidden from attention rather
    than silently interpreted as negative images.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        channels = EfficientNetFeatureEncoder.output_channels
        self.rbpt_encoder = EfficientNetFeatureEncoder(config.pretrained_backbone)
        self.sat_encoder = EfficientNetFeatureEncoder(config.pretrained_backbone)
        self.rbpt_segmentation = MultiLabelSegmentationHead(channels, len(RBPT_MASK_NAMES))
        self.sat_segmentation = MultiLabelSegmentationHead(channels, len(SAT_MASK_NAMES))
        self.rbpt_projection = nn.Linear(channels, config.embed_dim)
        self.sat_projection = nn.Linear(channels, config.embed_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(1, NUM_MODALITIES, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.embed_dim),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, len(DIAGNOSIS_NAMES)),
        )
        # These heads operate before cross-modal fusion. They intentionally make
        # no RBPT-SAT agreement assumption and no monotonic SAT-dilution assumption.
        self.rbpt_reaction_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, len(REACTION_NAMES)),
        )
        self.sat_reaction_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, len(REACTION_NAMES)),
        )
        self.domain_head = nn.Sequential(
            GradientReversal(),
            nn.Linear(config.embed_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(128, max(1, config.num_centers)),
        )

    @staticmethod
    def _pool(features: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(features, 1).flatten(1)

    def set_domain_alpha(self, alpha: float) -> None:
        self.domain_head[0].alpha = float(alpha)

    def forward(
        self,
        rbpt_images: torch.Tensor,
        sat_images: torch.Tensor,
        modality_present: torch.Tensor | None = None,
        *,
        return_masks: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        if sat_images.ndim != 5 or sat_images.shape[1] != 4 or sat_images.shape[2] != 3:
            raise ValueError("sat_images must have shape (batch, 4, 3, height, width)")
        if rbpt_images.ndim != 4 or rbpt_images.shape[1] != 3:
            raise ValueError("rbpt_images must have shape (batch, 3, height, width)")
        batch, _, _, height, width = sat_images.shape
        if modality_present is None:
            modality_present = torch.ones(batch, NUM_MODALITIES, dtype=torch.bool, device=rbpt_images.device)
        modality_present = modality_present.bool()
        if modality_present.shape != (batch, NUM_MODALITIES):
            raise ValueError(f"modality_present must have shape ({batch}, {NUM_MODALITIES})")

        if rbpt_images.is_cuda:
            rbpt_images = rbpt_images.contiguous(memory_format=torch.channels_last)
        rbpt_features = self.rbpt_encoder(rbpt_images)
        flat_sat = sat_images.flatten(0, 1)
        if flat_sat.is_cuda:
            flat_sat = flat_sat.contiguous(memory_format=torch.channels_last)
        sat_features = self.sat_encoder(flat_sat)
        rbpt_token = self.rbpt_projection(self._pool(rbpt_features)).unsqueeze(1)
        sat_tokens = self.sat_projection(self._pool(sat_features)).reshape(batch, 4, -1)
        rbpt_reaction_logits = self.rbpt_reaction_head(
            rbpt_token.squeeze(1) + self.modality_embedding[:, 0]
        )
        sat_reaction_logits = self.sat_reaction_head(
            sat_tokens + self.modality_embedding[:, 1:]
        )
        modality_tokens = torch.cat((rbpt_token, sat_tokens), dim=1) + self.modality_embedding
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat((cls, modality_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=modality_present.device),
                ~modality_present,
            ),
            dim=1,
        )
        fused = self.fusion(sequence, src_key_padding_mask=padding_mask)[:, 0]
        diagnosis_logits = self.classifier(fused)
        domain_logits = self.domain_head(fused)

        rbpt_mask_logits = None
        sat_mask_logits = None
        if return_masks:
            rbpt_mask_logits = self.rbpt_segmentation(rbpt_features, (height, width))
            sat_mask_logits = self.sat_segmentation(sat_features, (height, width)).reshape(
                batch, 4, len(SAT_MASK_NAMES), height, width
            )
        return {
            "diagnosis": diagnosis_logits,
            "rbpt_reaction": rbpt_reaction_logits,
            "sat_reactions": sat_reaction_logits,
            "domain": domain_logits,
            "rbpt_masks": rbpt_mask_logits,
            "sat_masks": sat_mask_logits,
            "embedding": fused,
        }


def backbone_parameters(model: AIRBPTSATMultimodalModel):
    yield from model.rbpt_encoder.parameters()
    yield from model.sat_encoder.parameters()


def head_parameters(model: AIRBPTSATMultimodalModel):
    backbone_ids = {id(parameter) for parameter in backbone_parameters(model)}
    for parameter in model.parameters():
        if id(parameter) not in backbone_ids:
            yield parameter


if __name__ == "__main__":
    network = AIRBPTSATMultimodalModel(ModelConfig(pretrained_backbone=False, num_centers=4))
    outputs = network(
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 4, 3, 224, 224),
        torch.ones(2, 5, dtype=torch.bool),
    )
    print({name: None if value is None else tuple(value.shape) for name, value in outputs.items()})
