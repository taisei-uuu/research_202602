"""
Policy Network  π(x)  —  outputs bounded acceleration offsets per agent.

Architecture (Table I):
    Encoder ψ₁ : [node_dim*2 + edge_dim → 256 → 256 → 128]
    Attention ψ₂: [128 → 128 → 128 → 1]
    Value ψ₃   : [128 → 256 → 128]
    Decoder ψ₄ : [128 → 256 → 256 → action_dim]

Output is passed through tanh → values in [-1, 1].
Caller scales by (a_max_gnn, a_max_gnn, a_max_gnn_s) to get physical acceleration offsets
added directly to the nominal PD acceleration.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gnn import GNNLayer
from ..utils.graph import GraphsTuple


class PolicyNetwork(nn.Module):
    """
    Distributed policy network (velocity-command mode).

    Takes a ``GraphsTuple`` and returns bounded velocity commands for every
    agent:  (Δv_x, Δv_y, ṡ_target) ∈ [-1, 1]³  (pre-scaling).

    Parameters
    ----------
    node_dim : int
        Dimension of per-node features (default 3).
    edge_dim : int
        Dimension of raw edge features (default 4).
    action_dim : int
        Dimension of the control output (3 for velocity commands).
    n_agents : int
        Number of agents.
    n_layers : int
        Number of stacked GNN layers (default 1).
    """

    AGENT_TYPE = 0

    def __init__(
        self,
        node_dim: int = 3,
        edge_dim: int = 4,
        action_dim: int = 3,
        n_agents: int = 4,
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim

        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GNNLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    msg_hid_sizes=(256, 256),
                    msg_out_dim=128,
                    attn_hid_sizes=(128, 128),
                    value_hid_sizes=(256,),
                    value_out_dim=128,
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
        u : (n_agents, action_dim) — bounded velocity commands in [-1, 1].
        """
        out = self.gnn_layers[0](graph)  # (N, action_dim)

        # Extract only agent nodes (use graph node_type to handle variable agent counts)
        agent_mask = graph.node_type == self.AGENT_TYPE
        u = out[agent_mask]  # (n_agents_actual, action_dim)
        return torch.tanh(u)

