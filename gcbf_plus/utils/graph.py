"""
Lightweight graph data structure for GNN message passing.

Replaces torch_geometric with simple tensor-based graph representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GraphsTuple:
    """
    A minimal graph container for GNN operations.

    Attributes
    ----------
    nodes : Tensor (N, node_dim)
        Node feature matrix.
    edges : Tensor (E, edge_dim)
        Edge feature matrix.
    senders : LongTensor (E,)
        Source node index for each edge.
    receivers : LongTensor (E,)
        Target node index for each edge.
    n_node : int
        Total number of nodes.
    n_edge : int
        Total number of edges.
    node_type : LongTensor (N,)
        Integer type label per node (e.g. 0=agent, 1=goal, 2=obstacle).
    """

    nodes: torch.Tensor
    edges: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    n_node: int
    n_edge: int
    node_type: torch.Tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def type_nodes(self, node_type_id: int) -> torch.Tensor:
        """Return the feature matrix for nodes of a given type."""
        mask = self.node_type == node_type_id
        return self.nodes[mask]

    def to(self, device: torch.device) -> "GraphsTuple":
        """Move all tensors to *device*."""
        return GraphsTuple(
            nodes=self.nodes.to(device),
            edges=self.edges.to(device),
            senders=self.senders.to(device),
            receivers=self.receivers.to(device),
            n_node=self.n_node,
            n_edge=self.n_edge,
            node_type=self.node_type.to(device),
        )


def build_graph_from_states(
    agent_states: torch.Tensor,
    goal_states: torch.Tensor,
    obstacle_positions: Optional[torch.Tensor],
    comm_radius: float,
    node_dim: int = 3,
    edge_dim: int = 4,
) -> GraphsTuple:
    """
    Build a ``GraphsTuple`` from raw state tensors.

    Parameters
    ----------
    agent_states : (n_agents, 4)  —  [px, py, vx, vy]
    goal_states  : (n_agents, 4)  —  [px, py, vx, vy]
    obstacle_positions : (n_obs, 4) or None  —  [px, py, 0, 0]
    comm_radius : float — sensing / communication radius *R*
    node_dim : int — dimension of the one-hot node indicator (default 3)
    edge_dim : int — dimension of edge features (default 4)

    Returns
    -------
    GraphsTuple
    """
    n_agents = agent_states.shape[0]
    device = agent_states.device

    # ---- Collect all nodes ----
    # Node types: 0 = agent, 1 = goal, 2 = obstacle
    all_states = [agent_states, goal_states]
    node_types = [
        torch.zeros(n_agents, dtype=torch.long, device=device),       # agents
        torch.ones(n_agents, dtype=torch.long, device=device),        # goals
    ]

    if obstacle_positions is not None and obstacle_positions.shape[0] > 0:
        n_obs = obstacle_positions.shape[0]
        all_states.append(obstacle_positions)
        node_types.append(
            torch.full((n_obs,), 2, dtype=torch.long, device=device)  # obstacles
        )

    all_states = torch.cat(all_states, dim=0)  # (N, 4)
    node_type_vec = torch.cat(node_types, dim=0)  # (N,)
    n_total = all_states.shape[0]

    # ---- One-hot node features ----
    node_feats = torch.zeros(n_total, node_dim, device=device)
    for t in range(node_dim):
        node_feats[node_type_vec == t, t] = 1.0

    # ---- Build edges within comm_radius ----
    # Edges are FROM every node TO every agent (agents are the receivers that
    # need neighbourhood information).  Self-loops for agents are excluded.
    positions = all_states[:, :2]  # (N, 2)
    agent_positions = positions[:n_agents]  # (n_agents, 2)

    # Pairwise distances: (N, n_agents)
    diff = positions.unsqueeze(1) - agent_positions.unsqueeze(0)  # (N, n_agents, 2)
    dist = torch.norm(diff, dim=-1)  # (N, n_agents)

    # Mask: within radius and not self-loop for agent<->agent
    mask = dist <= comm_radius  # (N, n_agents)

    # Remove self-loops (agent i -> agent i)
    self_loop_mask = torch.zeros_like(mask, dtype=torch.bool)
    for i in range(n_agents):
        self_loop_mask[i, i] = True
    mask = mask & ~self_loop_mask

    sender_idx, receiver_idx = torch.where(mask)  # (E,), (E,)

    # ---- Edge features: relative state  [Δpx, Δpy, Δvx, Δvy] ----
    if sender_idx.numel() > 0:
        edge_feats = all_states[sender_idx] - all_states[receiver_idx]  # (E, 4)
    else:
        edge_feats = torch.zeros(0, edge_dim, device=device)

    return GraphsTuple(
        nodes=node_feats,
        edges=edge_feats,
        senders=sender_idx,
        receivers=receiver_idx,
        n_node=n_total,
        n_edge=sender_idx.shape[0],
        node_type=node_type_vec,
    )
