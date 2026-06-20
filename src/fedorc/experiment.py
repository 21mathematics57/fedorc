"""FedORC experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from .aggregation import aggregate_local_models_phase, merge_two_analytic_models
from .backbone import load_backbone
from .buffers import RandomBuffer
from .config import ExperimentConfig
from .data import build_datasets, build_loader
from .evaluation import evaluate_dual_buffer
from .split import split_by_phases_for_clients
from .trainer import (
    local_analytic_train_acil,
    local_compensation_train_tanh_scheme_a,
)


def run_experiment(
    config: ExperimentConfig,
    main_lam: float,
    com_lam: float,
    comp_lam: float,
    summary_file: Path,
) -> None:
    device = config.resolve_device()
    print(f"[INFO] Using device: {device}")

    train_dataset, test_dataset, num_classes = build_datasets(
        config.dataset,
        config.data_root,
    )
    print(f"[INFO] Loaded {config.dataset} with {num_classes} classes.")

    backbone, feature_size = load_backbone(config.backbone_path, device)
    if feature_size != config.in_features:
        raise ValueError(
            f"feature_size({feature_size}) != in_features({config.in_features})"
        )

    shared_buffer = RandomBuffer(
        in_features=config.in_features,
        out_features=config.buffer_size,
        bias=False,
        device=device,
        dtype=config.dtype,
        activation=torch.nn.Identity(),
    )
    shared_buffer.eval()

    phase_client_subsets, phase_label_groups, _ = split_by_phases_for_clients(
        train_dataset,
        num_phases=config.num_phases,
        num_clients=config.num_clients,
        first_phase_ratio=config.first_phase_ratio,
        iid_alpha=config.iid_alpha,
        seed=config.phase_split_seed,
    )

    w_base_global: Optional[torch.Tensor] = None
    c_base_global: Optional[torch.Tensor] = None
    b_base_global: Optional[torch.Tensor] = None
    w_comp_global: Optional[torch.Tensor] = None
    c_comp_global: Optional[torch.Tensor] = None
    b_comp_global: Optional[torch.Tensor] = None

    test_targets = np.array(test_dataset.targets)
    phase_acc_list: List[float] = []

    for phase_id in range(config.num_phases):
        phase_labels = phase_label_groups[phase_id]
        seen_labels = sorted({
            label
            for labels in phase_label_groups[: phase_id + 1]
            for label in labels
        })
        old_labels = sorted(set(seen_labels) - set(phase_labels))

        client_subsets = phase_client_subsets[phase_id]
        sizes = [len(subset) for subset in client_subsets]
        print(
            f"[Split] Phase {phase_id} sizes per client: {sizes}, "
            f"min/max: {min(sizes)}/{max(sizes)}"
        )
        print(f"[Phase {phase_id}] current phase labels: {phase_labels}")
        print(f"[Phase {phase_id}] old labels: {old_labels}")
        print(f"[Phase {phase_id}] seen labels: {seen_labels}")

        base_local_w: List[torch.Tensor] = []
        base_local_c: List[torch.Tensor] = []
        base_local_b: List[torch.Tensor] = []

        for subset in client_subsets:
            if len(subset) == 0:
                continue
            client_loader = build_loader(
                subset,
                batch_size=config.local_batch_size,
                shuffle=True,
                num_workers=config.num_workers,
            )
            w_phase, c_phase, b_phase = local_analytic_train_acil(
                data_loader=client_loader,
                backbone=backbone,
                buffer=shared_buffer,
                num_classes_global=num_classes,
                device=device,
                dtype=config.dtype,
                main_lam=main_lam,
            )
            base_local_w.append(w_phase)
            base_local_c.append(c_phase)
            base_local_b.append(b_phase)

        if not base_local_w:
            print(f"[WARN] Phase {phase_id} has no valid base-stream local model.")
            continue

        w_base_phase, c_base_phase, b_base_phase = aggregate_local_models_phase(
            local_w_list=base_local_w,
            local_c_list=base_local_c,
            local_b_list=base_local_b,
            dtype=config.dtype,
            device=device,
        )

        if w_base_global is None:
            w_base_global = w_base_phase
            c_base_global = c_base_phase
            b_base_global = b_base_phase
        else:
            w_base_global, c_base_global, b_base_global = merge_two_analytic_models(
                w_left=w_base_global,
                c_left=c_base_global,
                b_left=b_base_global,
                w_right=w_base_phase,
                c_right=c_base_phase,
                b_right=b_base_phase,
                dtype=config.dtype,
                device=device,
            )

        comp_local_w: List[torch.Tensor] = []
        comp_local_c: List[torch.Tensor] = []
        comp_local_b: List[torch.Tensor] = []

        for subset in client_subsets:
            if len(subset) == 0:
                continue
            client_loader = build_loader(
                subset,
                batch_size=config.local_batch_size,
                shuffle=True,
                num_workers=config.num_workers,
            )
            w_comp, c_comp, b_comp = local_compensation_train_tanh_scheme_a(
                data_loader=client_loader,
                backbone=backbone,
                buffer=shared_buffer,
                w_base=w_base_global,
                num_classes_global=num_classes,
                current_phase_labels=phase_labels,
                device=device,
                dtype=config.dtype,
                com_lam=com_lam,
            )
            comp_local_w.append(w_comp)
            comp_local_c.append(c_comp)
            comp_local_b.append(b_comp)

        if comp_local_w:
            w_comp_phase, c_comp_phase, b_comp_phase = aggregate_local_models_phase(
                local_w_list=comp_local_w,
                local_c_list=comp_local_c,
                local_b_list=comp_local_b,
                dtype=config.dtype,
                device=device,
            )
            if w_comp_global is None:
                w_comp_global = w_comp_phase
                c_comp_global = c_comp_phase
                b_comp_global = b_comp_phase
            else:
                w_comp_global, c_comp_global, b_comp_global = merge_two_analytic_models(
                    w_left=w_comp_global,
                    c_left=c_comp_global,
                    b_left=b_comp_global,
                    w_right=w_comp_phase,
                    c_right=c_comp_phase,
                    b_right=b_comp_phase,
                    dtype=config.dtype,
                    device=device,
                )
        elif w_comp_global is None:
            w_comp_global = torch.zeros_like(w_base_global)
            c_comp_global = torch.zeros(
                (config.buffer_size, config.buffer_size),
                dtype=config.dtype,
                device=device,
            )
            b_comp_global = torch.zeros(
                (config.buffer_size, num_classes),
                dtype=config.dtype,
                device=device,
            )

        mask_seen = np.isin(test_targets, seen_labels)
        test_indices_phase = np.where(mask_seen)[0]
        if len(test_indices_phase) == 0:
            print(f"[WARN] Phase {phase_id} has no seen-class test data.")
            continue

        test_subset = Subset(test_dataset, test_indices_phase.tolist())
        test_loader = build_loader(
            test_subset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        test_loss, test_acc = evaluate_dual_buffer(
            data_loader=test_loader,
            backbone=backbone,
            buffer=shared_buffer,
            w_base=w_base_global,
            w_comp=w_comp_global if w_comp_global is not None else torch.zeros_like(w_base_global),
            comp_lam=comp_lam,
            device=device,
            desc=f"Phase {phase_id} Test",
        )

        phase_acc_list.append(test_acc)
        avg_acc = sum(phase_acc_list) / len(phase_acc_list)
        print(
            f"Phase {phase_id}: loss={test_loss:.4f}, "
            f"acc={test_acc:.4f}, avg_acc={avg_acc:.4f}"
        )

        with summary_file.open("a", encoding="utf-8") as file:
            file.write(
                f"{main_lam},{com_lam},{comp_lam},"
                f"{phase_id},{test_loss:.6f},{test_acc:.6f},{avg_acc:.6f}\n"
            )


def grid_search(
    config: ExperimentConfig,
    main_list: List[float],
    com_list: List[float],
    comp_list: List[float],
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = config.output_dir / "summary.csv"
    summary_file.write_text("main,com,lambda,phase,loss,acc,avg_acc\n", encoding="utf-8")

    for main_lam in main_list:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for com_lam in com_list:
            for comp_lam in comp_list:
                print("\n" + "=" * 50)
                print(f"RUN: main={main_lam}, com={com_lam}, lambda={comp_lam}")
                print("=" * 50)
                try:
                    run_experiment(
                        config=config,
                        main_lam=main_lam,
                        com_lam=com_lam,
                        comp_lam=comp_lam,
                        summary_file=summary_file,
                    )
                except Exception as error:
                    print(f"[ERROR] Failed: {error}")

    return summary_file
