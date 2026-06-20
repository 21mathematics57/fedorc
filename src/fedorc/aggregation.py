"""Analytic model aggregation utilities."""

from __future__ import annotations

from typing import List, Tuple

import torch


def merge_two_analytic_models(
    w_left: torch.Tensor,
    c_left: torch.Tensor,
    b_left: torch.Tensor,
    w_right: torch.Tensor,
    c_right: torch.Tensor,
    b_right: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge two analytic model blocks by summing regularized statistics."""
    del w_left, w_right
    c_merge = c_left.to(device=device, dtype=dtype) + c_right.to(
        device=device,
        dtype=dtype,
    )
    b_merge = b_left.to(device=device, dtype=dtype) + b_right.to(
        device=device,
        dtype=dtype,
    )
    w_merge = torch.linalg.solve(c_merge, b_merge)
    return w_merge, c_merge, b_merge


def aggregate_local_models_phase(
    local_w_list: List[torch.Tensor],
    local_c_list: List[torch.Tensor],
    local_b_list: List[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate all client analytic models inside one phase."""
    if not local_w_list:
        raise ValueError("No local models to aggregate.")
    if not len(local_w_list) == len(local_c_list) == len(local_b_list):
        raise ValueError("local W/C/B list lengths do not match.")

    wt = local_w_list[0].to(device=device, dtype=dtype)
    ct = local_c_list[0].to(device=device, dtype=dtype)
    bt = local_b_list[0].to(device=device, dtype=dtype)

    for wk, ck, bk in zip(local_w_list[1:], local_c_list[1:], local_b_list[1:]):
        wt, ct, bt = merge_two_analytic_models(
            w_left=wt,
            c_left=ct,
            b_left=bt,
            w_right=wk,
            c_right=ck,
            b_right=bk,
            dtype=dtype,
            device=device,
        )

    return wt, ct, bt
