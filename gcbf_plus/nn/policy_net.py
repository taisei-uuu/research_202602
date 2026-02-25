"""
Policy Network  π(x)  —  outputs a control action per agent.

Architecture (Table I):
    Encoder ψ₁ : [edge_dim → 256 → 256 → 128]
    Attention ψ₂: [128 → 128 → 128 → 1]
    Decoder ψ₄ : [128 → 256 → 256 → action_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gnn import GNNLayer
from ..utils.graph import GraphsTuple


class PolicyNetwork(nn.Module):
    """
    Distributed policy network.

    Takes a ``GraphsTuple`` and returns a control action u_i for every agent.

    Parameters
    ----------
    edge_dim : int
        Dimension of raw edge features (default 4).
    action_dim : int
        Dimension of the control output (default 2 for Double Integrator).
    n_agents : int
        Number of agents.
    n_layers : int
        Number of stacked GNN layers (default 1).
    """

    AGENT_TYPE = 0

    def __init__(
        self,
        edge_dim: int = 4,
        action_dim: int = 2,
        n_agents: int = 4,
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim

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
                    out_dim=action_dim,
                )
            )

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Parameters
        ----------
        graph : GraphsTuple

        Returns
        -------
        u : (n_agents, action_dim) — control action per agent.
        """
        out = self.gnn_layers[0](graph)  # (N, action_dim)

        # Extract only agent nodes
        u = out[: self.n_agents]  # (n_agents, action_dim)
        return u
