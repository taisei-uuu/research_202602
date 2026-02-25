"""
Generic Multi-Layer Perceptron (MLP) building block.

Used as the backbone for ψ₁ (encoder), ψ₂ (attention), and ψ₄ (decoder).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple feed-forward MLP with ReLU activations.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_sizes : sequence of int
        Hidden layer sizes (e.g. ``[256, 256]``).
    out_dim : int
        Output feature dimension.
    act_final : bool
        If True, apply ReLU after the last linear layer.
    """

    def __init__(
        self,
        in_dim: int,
        hid_sizes: Sequence[int],
        out_dim: int,
        act_final: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hid_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if act_final:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

        # Xavier-uniform init (matches the reference code's default_nn_init)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
