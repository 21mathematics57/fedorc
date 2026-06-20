"""Local analytic training routines."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .buffers import RandomBuffer


def orthogonalize_compensation_features(
    phi_comp: torch.Tensor,
    phi_base: torch.Tensor,
) -> torch.Tensor:
    q_feat, _ = torch.linalg.qr(phi_base.T, mode="reduced")
    return phi_comp - phi_comp @ q_feat @ q_feat.T


def local_analytic_train_acil(
    data_loader: DataLoader,
    backbone: nn.Module,
    buffer: RandomBuffer,
    num_classes_global: int,
    device: torch.device,
    dtype: torch.dtype,
    main_lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train the base stream analytic classifier for one client."""
    backbone.eval()
    buffer.eval()

    feature_dim = buffer.out_features
    c_raw = torch.zeros((feature_dim, feature_dim), dtype=dtype, device=device)
    b_stat = torch.zeros((feature_dim, num_classes_global), dtype=dtype, device=device)

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Local analytic (base ReLU)"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            features = backbone(images)
            phi = torch.relu(buffer(features)).to(dtype=dtype)
            one_hot = torch.nn.functional.one_hot(
                labels,
                num_classes=num_classes_global,
            ).to(dtype)

            c_raw += phi.t() @ phi
            b_stat += phi.t() @ one_hot

    eye = torch.eye(feature_dim, dtype=dtype, device=device)
    c_stat = c_raw + main_lam * eye
    w_stat = torch.linalg.solve(c_stat, b_stat)
    return w_stat, c_stat, b_stat


def local_compensation_train_tanh_scheme_a(
    data_loader: DataLoader,
    backbone: nn.Module,
    buffer: RandomBuffer,
    w_base: torch.Tensor,
    num_classes_global: int,
    current_phase_labels: List[int],
    device: torch.device,
    dtype: torch.dtype,
    com_lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train the compensation stream on current-phase residuals."""
    backbone.eval()
    buffer.eval()

    feature_dim = buffer.out_features
    c_raw = torch.zeros((feature_dim, feature_dim), dtype=dtype, device=device)
    b_stat = torch.zeros((feature_dim, num_classes_global), dtype=dtype, device=device)

    class_mask = torch.zeros(num_classes_global, dtype=dtype, device=device)
    class_mask[current_phase_labels] = 1.0
    class_mask = class_mask.view(1, -1)

    with torch.no_grad():
        for images, labels in tqdm(
            data_loader,
            desc="Local compensation (tanh, scheme A)",
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            features = backbone(images)
            phi_base = torch.relu(buffer(features)).to(dtype=dtype)
            logits_base = phi_base @ w_base

            comp_features = torch.dropout(features, p=0.2, train=False)
            phi_comp = torch.tanh(buffer(comp_features)).to(dtype=dtype)
            phi_comp = orthogonalize_compensation_features(phi_comp, phi_base)

            one_hot = torch.nn.functional.one_hot(
                labels,
                num_classes=num_classes_global,
            ).to(dtype)
            y_residual = (one_hot - logits_base) * class_mask

            c_raw += phi_comp.t() @ phi_comp
            b_stat += phi_comp.t() @ y_residual

    eye = torch.eye(feature_dim, dtype=dtype, device=device)
    c_stat = c_raw + com_lam * eye
    w_stat = torch.linalg.solve(c_stat, b_stat)
    return w_stat, c_stat, b_stat
