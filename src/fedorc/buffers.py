"""Buffer layers used by analytic continual learning."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Union

import torch

Activation = Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]


class Buffer(torch.nn.Module, metaclass=ABCMeta):
    """Base class for non-trainable feature buffers."""

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomBuffer(torch.nn.Linear, Buffer):
    """Random linear projection stored as buffers instead of parameters.

    This is adapted from the ACIL project's ``analytic.Buffer.RandomBuffer`` so
    FedORC can run without importing another local repository through
    ``sys.path``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=torch.float,
        activation: Optional[Activation] = torch.relu_,
    ) -> None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation: Activation = (
            torch.nn.Identity() if activation is None else activation
        )

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        bias_tensor = torch.empty(out_features, **factory_kwargs) if bias else None

        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias_tensor)
        self.reset_parameters()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.weight)
        return self.activation(super().forward(inputs))
