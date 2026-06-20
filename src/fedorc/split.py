"""Phase and client dataset splitting."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset


def split_by_phases_for_clients(
    train_dataset: Dataset,
    num_phases: int,
    num_clients: int,
    first_phase_ratio: float = 0.5,
    rounding: str = "round",
    iid_alpha: float = 1.0,
    dirichlet_min: float = 0.01,
    dirichlet_scale: float = 10.0,
    seed: int = 123,
) -> Tuple[List[List[Subset]], List[List[int]], List[Subset]]:
    """Split labels into phases, then split each phase across clients."""
    if num_phases < 2:
        raise ValueError("num_phases must be >= 2.")
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1.")
    if not (0.0 < first_phase_ratio < 1.0):
        raise ValueError("first_phase_ratio must be in (0, 1).")
    if not (0.0 <= iid_alpha <= 1.0):
        raise ValueError("iid_alpha must be in [0, 1].")

    targets = np.array(getattr(train_dataset, "targets"))
    all_labels = sorted(set(targets.tolist()))
    num_classes = len(all_labels)

    raw_first = num_classes * first_phase_ratio
    if rounding == "floor":
        first_k = int(np.floor(raw_first))
    elif rounding == "ceil":
        first_k = int(np.ceil(raw_first))
    elif rounding == "round":
        first_k = int(np.round(raw_first))
    else:
        raise ValueError("rounding must be one of: 'round', 'floor', 'ceil'")

    first_k = max(1, min(first_k, num_classes - (num_phases - 1)))
    rng = np.random.RandomState(seed)

    labels_np = np.array(all_labels)
    first_group = labels_np[:first_k]
    rest_group = labels_np[first_k:]
    label_groups_np = [first_group] + list(np.array_split(rest_group, num_phases - 1))

    phase_client_subsets: List[List[Subset]] = []
    phase_label_groups: List[List[int]] = []
    client_all_indices: List[List[int]] = [[] for _ in range(num_clients)]
    dir_alpha = dirichlet_min + iid_alpha * dirichlet_scale

    def split_phase_iid(phase_indices: np.ndarray) -> List[List[int]]:
        perm = rng.permutation(phase_indices)
        return [split.tolist() for split in np.array_split(perm, num_clients)]

    def split_phase_dirichlet(
        label_group: List[int],
        phase_seed_offset: int,
    ) -> List[List[int]]:
        client_indices = [[] for _ in range(num_clients)]
        rng_phase = np.random.RandomState(phase_seed_offset)

        for class_id in label_group:
            class_indices = np.where(targets == class_id)[0]
            if class_indices.size == 0:
                continue
            rng_phase.shuffle(class_indices)

            proportions = rng_phase.dirichlet([dir_alpha] * num_clients)
            split_sizes = (proportions * len(class_indices)).astype(int)

            diff = len(class_indices) - split_sizes.sum()
            if diff > 0:
                for _ in range(diff):
                    split_sizes[rng_phase.randint(num_clients)] += 1
            elif diff < 0:
                for _ in range(-diff):
                    largest = int(np.argmax(split_sizes))
                    if split_sizes[largest] > 0:
                        split_sizes[largest] -= 1

            start = 0
            for client_id, size in enumerate(split_sizes):
                if size > 0:
                    end = start + size
                    client_indices[client_id].extend(class_indices[start:end].tolist())
                start += size

        for indices in client_indices:
            if len(indices) > 1:
                rng.shuffle(indices)

        return client_indices

    for phase_id, label_group_np in enumerate(label_groups_np):
        label_group = label_group_np.tolist()
        phase_label_groups.append(label_group)

        phase_indices = np.where(np.isin(targets, label_group))[0]
        if iid_alpha >= 0.999:
            client_index_lists = split_phase_iid(phase_indices)
        else:
            client_index_lists = split_phase_dirichlet(
                label_group=label_group,
                phase_seed_offset=seed + 1000 * (phase_id + 1),
            )

        phase_subsets = []
        for client_id, index_list in enumerate(client_index_lists):
            phase_subsets.append(Subset(train_dataset, index_list))
            client_all_indices[client_id].extend(index_list)
        phase_client_subsets.append(phase_subsets)

    client_union_subsets = [
        Subset(train_dataset, sorted(set(indices))) for indices in client_all_indices
    ]
    return phase_client_subsets, phase_label_groups, client_union_subsets
