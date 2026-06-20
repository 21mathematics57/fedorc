"""Command line entry point for FedORC training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from .config import ExperimentConfig
from .experiment import grid_search


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "double": torch.double,
        "float64": torch.float64,
    }
    try:
        return mapping[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {value}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FedORC analytic FL-CIL training.")
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fedorc"))
    parser.add_argument("--dataset", default="imagenet", choices=["imagenet", "cifar100"])
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-phases", type=int, default=6)
    parser.add_argument("--local-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=8192)
    parser.add_argument("--iid-alpha", type=float, default=1.0)
    parser.add_argument("--first-phase-ratio", type=float, default=0.5)
    parser.add_argument("--phase-split-seed", type=int, default=43)
    parser.add_argument("--dtype", type=parse_dtype, default=torch.double)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--main-list", type=parse_float_list, default=[0.5])
    parser.add_argument("--com-list", type=parse_float_list, default=[0.01])
    parser.add_argument("--comp-list", type=parse_float_list, default=[0.6])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        backbone_path=args.backbone_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_phases=args.num_phases,
        local_batch_size=args.local_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        in_features=args.in_features,
        buffer_size=args.buffer_size,
        iid_alpha=args.iid_alpha,
        first_phase_ratio=args.first_phase_ratio,
        phase_split_seed=args.phase_split_seed,
        dtype=args.dtype,
        device=args.device,
    )
    summary_file = grid_search(
        config=config,
        main_list=args.main_list,
        com_list=args.com_list,
        comp_list=args.comp_list,
    )
    print(f"[INFO] Summary written to {summary_file}")


if __name__ == "__main__":
    main()
