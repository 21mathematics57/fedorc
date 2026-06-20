"""Evaluation utilities."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .buffers import RandomBuffer
from .trainer import orthogonalize_compensation_features


def evaluate_dual_buffer(
    data_loader: DataLoader,
    backbone: nn.Module,
    buffer: RandomBuffer,
    w_base: torch.Tensor,
    w_comp: torch.Tensor,
    comp_lam: float,
    device: torch.device,
    desc: str = "Eval",
) -> Tuple[float, float]:
    backbone.eval()
    buffer.eval()

    dtype = w_base.dtype
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            features = backbone(images)
            phi_base = torch.relu(buffer(features)).to(dtype=dtype)

            comp_features = torch.dropout(features, p=0.2, train=False)
            phi_comp = torch.tanh(buffer(comp_features)).to(dtype=dtype)
            phi_comp = orthogonalize_compensation_features(phi_comp, phi_base)

            logits_main = phi_base @ w_base
            logits_comp = phi_comp @ w_comp
            logits = logits_main + comp_lam * logits_comp

            loss = criterion(logits.float(), labels)
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total
