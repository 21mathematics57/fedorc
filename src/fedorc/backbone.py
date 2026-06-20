"""Backbone loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn


def load_backbone(backbone_path: Path, device: torch.device) -> Tuple[nn.Module, int]:
    """Load a saved backbone tuple: ``(backbone, _, feature_size)``."""
    backbone, _, feature_size = torch.load(
        backbone_path,
        map_location=device,
        weights_only=False,
    )
    backbone = backbone.to(device)
    backbone.eval()
    return backbone, feature_size
