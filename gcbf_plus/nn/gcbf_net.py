"""
GCBF Network  h(x)  —  outputs a scalar Control Barrier Function value per agent.

Architecture (Table I):
    Encoder ψ₁ : [edge_dim → 256 → 256 → 128]
    Attention ψ₂: [128 → 128 → 128 → 1]
    Decoder ψ₄ : [128 → 256 → 256 → 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gnn import GNNLayer
from ..utils.graph import GraphsTuple


class GCBFNetwork(nn.Module):
    """
    Graph Control Barrier Function network.

    Takes a ``GraphsTuple`` and returns a scalar h_i for every *agent* node.

    Parameters
    ----------
    edge_dim : int
        Dimension of raw edge features (default 4 for Double Integrator).
    n_agents : int
        Number of agents (used to slice the output).
    n_layers : int
        Number of stacked GNN layers (default 1, matching the paper).
    """

    AGENT_TYPE = 0  # node-type id for agents

    def __init__(
        self,
        edge_dim: int = 4,
        n_agents: int = 4,
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_agents = n_agents

        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = edge_dim if i == 0 else 128
            self.gnn_layers.append(
                GNNLayer(
                    edge_dim=in_dim,
                    msg_hid_sizes=(256, 256),
                    msg_out_dim=128,
                    attn_hid_sizes=(128, 128),
                    update_hid_sizes=(256, 256),
                    out_dim=1,  # scalar CBF value
                )
            )

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Parameters
        ----------
        graph : GraphsTuple

        Returns
        -------
        h : (n_agents, 1) — CBF value per agent.
        """
        out = self.gnn_layers[0](graph)  # (N, 1)

        # For multi-layer: feed back into next GNN layer (future extension)
        # For now we use 1 layer as in the paper.

        # Extract only agent nodes (first n_agents nodes)
        h = out[: self.n_agents]  # (n_agents, 1)
        return h
