from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from multimodal_model_nvidia import DIAGNOSIS_NAMES, NUM_MODALITIES, REACTION_NAMES
from multimodal_model_nvidia import GradientReversal


@dataclass(frozen=True)
class DualViewModelConfig:
    embed_dim: int = 256
    transformer_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.25
    num_centers: int = 1
    pretrained_backbone: bool = True
    reaction_evidence_scale: float = 0.35
    architecture_version: str = "dual_view_highres_roi_evidence_r4"

    def to_dict(self) -> dict:
        return asdict(self)


class EfficientNetFeatureEncoder(nn.Module):
    output_channels = 112

    def __init__(self, pretrained: bool):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = nn.Sequential(*list(backbone.features.children())[:6])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class AIRBPTSATDualViewModel(nn.Module):
    """Five-image classifier with high-resolution assay ROI evidence fusion.

    RBPT and each SAT dilution remain independent observations. The model learns
    reaction evidence from fixed, fold-safe assay-region crops and never assumes
    that RBPT/SAT agree or that SAT dilution reactions are monotonic.
    """

    def __init__(self, config: DualViewModelConfig):
        super().__init__()
        self.config = config
        channels = EfficientNetFeatureEncoder.output_channels
        half = config.embed_dim // 2
        if config.embed_dim % 2:
            raise ValueError("embed_dim must be even")
        self.rbpt_encoder = EfficientNetFeatureEncoder(config.pretrained_backbone)
        self.sat_encoder = EfficientNetFeatureEncoder(config.pretrained_backbone)
        self.rbpt_global_projection = nn.Linear(channels, half)
        self.rbpt_roi_projection = nn.Linear(channels, half)
        self.sat_global_projection = nn.Linear(channels, half)
        self.sat_roi_projection = nn.Linear(channels, half)
        self.rbpt_reaction_projection = nn.Linear(channels, config.embed_dim)
        self.sat_reaction_projection = nn.Linear(channels, config.embed_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(1, NUM_MODALITIES, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.embed_dim),
            enable_nested_tensor=False,
        )
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
        self.reaction_token_projection = nn.Sequential(
            nn.Linear(len(REACTION_NAMES), config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.Tanh(),
        )
        # RBPT(3) + four SAT dilution distributions(12) + SAT mean(3) + SAT max(3).
        self.assay_summary_projection = nn.Sequential(
            nn.Linear(21, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, len(DIAGNOSIS_NAMES)),
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

    def _encode_pair(
        self,
        encoder: nn.Module,
        global_images: torch.Tensor,
        roi_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Global and ROI views intentionally support different resolutions.
        if global_images.is_cuda:
            global_images = global_images.contiguous(memory_format=torch.channels_last)
            roi_images = roi_images.contiguous(memory_format=torch.channels_last)
        global_pool = self._pool(encoder(global_images))
        roi_pool = self._pool(encoder(roi_images))
        return global_pool, roi_pool

    def forward(
        self,
        rbpt_global: torch.Tensor,
        rbpt_roi: torch.Tensor,
        sat_global: torch.Tensor,
        sat_roi: torch.Tensor,
        modality_present: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if rbpt_global.ndim != 4 or rbpt_roi.ndim != 4:
            raise ValueError("RBPT full and ROI tensors must both have shape (B,3,H,W)")
        if rbpt_global.shape[:2] != rbpt_roi.shape[:2] or rbpt_global.shape[1] != 3:
            raise ValueError("RBPT full and ROI tensors must share batch/channel dimensions")
        if sat_global.ndim != 5 or sat_roi.ndim != 5:
            raise ValueError("SAT full and ROI tensors must both have shape (B,4,3,H,W)")
        if sat_global.shape[:3] != sat_roi.shape[:3] or sat_global.shape[1:3] != (4, 3):
            raise ValueError("SAT full and ROI tensors must share batch/modality/channel dimensions")
        batch = rbpt_global.shape[0]
        if sat_global.shape[0] != batch:
            raise ValueError("RBPT and SAT batch sizes differ")
        if modality_present is None:
            modality_present = torch.ones(batch, NUM_MODALITIES, dtype=torch.bool, device=rbpt_global.device)
        modality_present = modality_present.bool()
        if modality_present.shape != (batch, NUM_MODALITIES):
            raise ValueError(f"modality_present must have shape ({batch}, {NUM_MODALITIES})")

        rbpt_global_pool, rbpt_roi_pool = self._encode_pair(
            self.rbpt_encoder, rbpt_global, rbpt_roi
        )
        flat_sat_global = sat_global.flatten(0, 1)
        flat_sat_roi = sat_roi.flatten(0, 1)
        sat_global_pool, sat_roi_pool = self._encode_pair(
            self.sat_encoder, flat_sat_global, flat_sat_roi
        )
        rbpt_token = torch.cat(
            (
                self.rbpt_global_projection(rbpt_global_pool),
                self.rbpt_roi_projection(rbpt_roi_pool),
            ),
            dim=1,
        ).unsqueeze(1)
        sat_tokens = torch.cat(
            (
                self.sat_global_projection(sat_global_pool),
                self.sat_roi_projection(sat_roi_pool),
            ),
            dim=1,
        ).reshape(batch, 4, -1)
        rbpt_reaction_logits = self.rbpt_reaction_head(
            self.rbpt_reaction_projection(rbpt_roi_pool) + self.modality_embedding[:, 0]
        )
        sat_reaction_logits = self.sat_reaction_head(
            self.sat_reaction_projection(sat_roi_pool).reshape(batch, 4, -1)
            + self.modality_embedding[:, 1:]
        )
        rbpt_reaction_probabilities = rbpt_reaction_logits.softmax(1)
        sat_reaction_probabilities = sat_reaction_logits.softmax(2)

        modality_tokens = torch.cat((rbpt_token, sat_tokens), dim=1) + self.modality_embedding
        reaction_tokens = torch.cat(
            (
                rbpt_reaction_probabilities.unsqueeze(1),
                sat_reaction_probabilities,
            ),
            dim=1,
        )
        modality_tokens = modality_tokens + float(self.config.reaction_evidence_scale) * self.reaction_token_projection(reaction_tokens)
        sequence = torch.cat((self.cls_token.expand(batch, -1, -1), modality_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=modality_present.device),
                ~modality_present,
            ),
            dim=1,
        )
        fused = self.fusion(sequence, src_key_padding_mask=padding_mask)[:, 0]
        sat_mean = sat_reaction_probabilities.mean(dim=1)
        sat_max = sat_reaction_probabilities.max(dim=1).values
        assay_summary = torch.cat(
            (
                rbpt_reaction_probabilities,
                sat_reaction_probabilities.flatten(1),
                sat_mean,
                sat_max,
            ),
            dim=1,
        )
        fused = fused + self.assay_summary_projection(assay_summary)
        return {
            "diagnosis": self.classifier(fused),
            "rbpt_reaction": rbpt_reaction_logits,
            "sat_reactions": sat_reaction_logits,
            "domain": self.domain_head(fused),
            "embedding": fused,
            "assay_summary": assay_summary,
        }


def backbone_parameters(model: AIRBPTSATDualViewModel):
    yield from model.rbpt_encoder.parameters()
    yield from model.sat_encoder.parameters()


def head_parameters(model: AIRBPTSATDualViewModel):
    backbone_ids = {id(parameter) for parameter in backbone_parameters(model)}
    for parameter in model.parameters():
        if id(parameter) not in backbone_ids:
            yield parameter
