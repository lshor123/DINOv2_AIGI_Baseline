import os
from typing import Any

import torch
from torch import nn

from networks.dynamic_head import DynamicLinearHead, ResidualMLPHead


class DINOv2Baseline(nn.Module):
    """DINOv2 ViT-L/14 followed by a two-class linear classifier."""

    def __init__(
        self,
        backbone: str = "dinov2_vitl14",
        repo_path: str = "",
        weights_path: str = "",
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        classifier_type: str = "linear",
        dynamic_rank: int = 16,
        mlp_width: int = 32,
        head_fp32: bool = False,
    ):
        super().__init__()
        if backbone != "dinov2_vitl14":
            raise ValueError(
                f"This baseline is fixed to dinov2_vitl14, but received {backbone!r}."
            )
        if not repo_path or not os.path.isdir(repo_path):
            raise FileNotFoundError(
                f"DINOv2 repository not found: {repo_path}. "
                "Run scripts/prepare_server.sh first or pass --dino_repo."
            )
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"DINOv2 weights not found: {weights_path}. "
                "Run scripts/prepare_server.sh first or pass --dino_weights."
            )

        self.backbone_name = backbone
        self.backbone = torch.hub.load(
            repo_path,
            backbone,
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = self._unwrap_state_dict(state_dict)
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "DINOv2 weight mismatch. "
                f"Missing keys: {incompatible.missing_keys}; "
                f"unexpected keys: {incompatible.unexpected_keys}"
            )

        feature_dim = getattr(self.backbone, "embed_dim", None)
        if feature_dim is None:
            feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim != 1024:
            raise RuntimeError(
                f"Expected DINOv2 ViT-L/14 feature dimension 1024, got {feature_dim}."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2),
        )
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

        self.head_fp32 = head_fp32 or classifier_type != "linear"
        if classifier_type != "linear" and dropout != 0.0:
            raise ValueError("Residual-head experiments require dropout=0 for local consistency")
        if classifier_type == "dynamic":
            self.classifier = DynamicLinearHead(self.classifier, feature_dim, dynamic_rank)
        elif classifier_type == "mlp":
            self.classifier = ResidualMLPHead(self.classifier, feature_dim, mlp_width)
        elif classifier_type != "linear":
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _unwrap_state_dict(checkpoint: Any):
        if not isinstance(checkpoint, dict):
            return checkpoint
        for key in ("teacher", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

        cleaned = {}
        for key, value in checkpoint.items():
            for prefix in ("module.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            cleaned[key] = value
        return cleaned

    def forward_features(self, image):
        features = self.backbone(image)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return features

    def forward(self, image):
        return self.forward_head(self.forward_features(image))

    def forward_head(self, features):
        if self.head_fp32:
            with torch.autocast(device_type=features.device.type, enabled=False):
                return self.classifier(features.float())
        return self.classifier(features)

    def load_g5_initialization(self, checkpoint_path):
        """Warm start model weights only, with strict G5 key/shape validation.

        Optimizer/scheduler/epoch counters are deliberately reset. A new run
        is additional training from G5, not resuming G5's optimizer state.
        """
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected = {k for k in self.state_dict() if k.startswith("backbone.")}
        expected.update(("classifier.1.weight", "classifier.1.bias"))
        if set(state) != expected:
            raise RuntimeError(
                f"Expected a G5 linear checkpoint; missing={sorted(expected-set(state))}, "
                f"unexpected={sorted(set(state)-expected)}"
            )
        incompatible = self.load_state_dict(state, strict=False)
        allowed_missing = set(self.state_dict()) - expected
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected G5 initialization mismatch: {incompatible}")
