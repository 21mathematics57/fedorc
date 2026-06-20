"""Configuration objects for FedORC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ExperimentConfig:
    backbone_path: Path
    data_root: Path
    output_dir: Path = Path("runs/fedorc")
    dataset: str = "imagenet"
    num_clients: int = 5
    num_phases: int = 6
    local_batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    in_features: int = 512
    buffer_size: int = 8192
    iid_alpha: float = 1.0
    first_phase_ratio: float = 0.5
    phase_split_seed: int = 43
    dtype: torch.dtype = torch.double
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
