"""
Simplified swarm graph building — Bounding Circle approach.

Each swarm is treated as a point mass with a bounding circle of radius
r_swarm.  No rotation, no 9-point drone distances.

Edge creation:  CoM distance < R_sensing
Edge features (4D):  [Δpx, Δpy, Δvx, Δvy]
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .graph import GraphsTuple


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π].  (Kept for backward compat, rarely used now.)"""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------
# Per-sample graph builder (used by SwarmIntegrator / visualize.py)
# ------------------------------------------------------------------

def build_swarm_graph_from_states(
    agent_states: torch.Tensor,
    goal_states: torch.Tensor,
    obstacle_positions: Optional[torch.Tensor],
    comm_radius: float,
    node_dim: int = 3,
    edge_dim: int = 4,
) -> GraphsTuple:
    """
    Build a GraphsTuple for swarm agents using CoM-based bounding circles.

    Parameters
    ----------
    agent_states : (n_agents, 4)  — [px, py, vx, vy]
    goal_states  : (n_agents, 4)
    obstacle_positions : (n_obs, 4) or None
    comm_radius : float
    node_dim : int (default 3)
    edge_dim : int (default 4)

    Returns
    -------
    GraphsTuple with 4D edge features.
    """
    n_agents = agent_states.shape[0]
    device = agent_states.device

    # ---- Collect all nodes ----
    all_states = [agent_states, goal_states]
    node_types = [
        torch.zeros(n_agents, dtype=torch.long, device=device),
        torch.ones(n_agents, dtype=torch.long, device=device),
    ]

    n_obs = 0
    if obstacle_positions is not None and obstacle_positions.shape[0] > 0:
        n_obs = obstacle_positions.shape[0]
        all_states.append(obstacle_positions)
        node_types.append(
            torch.full((n_obs,), 2, dtype=torch.long, device=device)
        )

    all_states_cat = torch.cat(all_states, dim=0)   # (N_total, 4)
    node_type_vec = torch.cat(node_types, dim=0)
    n_total = all_states_cat.shape[0]

    # ---- One-hot node features ----
    node_feats = torch.zeros(n_total, node_dim, device=device)
    for t in range(node_dim):
        node_feats[node_type_vec == t, t] = 1.0

    # ---- All-to-agent edges (CoM distance) ----
    agent_pos = agent_states[:, :2]                        # (n, 2)
    all_pos = all_states_cat[:, :2]                        # (N, 2)
    diff = all_pos.unsqueeze(1) - agent_pos.unsqueeze(0)   # (N, n, 2)
    dist = torch.norm(diff, dim=-1)                        # (N, n)

    mask = dist <= comm_radius
    # Remove self-loops (agent→itself)
    for i in range(n_agents):
        mask[i, i] = False

    s_idx, r_idx = torch.where(mask)

    if s_idx.numel() > 0:
        delta_pos = all_states_cat[s_idx, :2] - agent_states[r_idx, :2]
        delta_vel = all_states_cat[s_idx, 2:4] - agent_states[r_idx, 2:4]
        edges = torch.cat([delta_pos, delta_vel], dim=-1)  # (E, 4)
    else:
        s_idx = torch.zeros(0, dtype=torch.long, device=device)
        r_idx = torch.zeros(0, dtype=torch.long, device=device)
        edges = torch.zeros(0, edge_dim, device=device)

    return GraphsTuple(
        nodes=node_feats,
        edges=edges,
        senders=s_idx,
        receivers=r_idx,
        n_node=n_total,
        n_edge=s_idx.shape[0],
        node_type=node_type_vec,
    )


# ------------------------------------------------------------------
# Vectorized mega-graph builder (batch dimension)
# ------------------------------------------------------------------

def build_vectorized_swarm_graph(
    agent_states: torch.Tensor,
    goal_states: torch.Tensor,
    obstacle_states: torch.Tensor,
    comm_radius: float,
    node_dim: int = 3,
    edge_dim: int = 4,
) -> GraphsTuple:
    """
    Build ONE mega-graph from B independent swarm environments.

    Parameters
    ----------
    agent_states    : (B, n_agents, 4)
    goal_states     : (B, n_agents, 4)
    obstacle_states : (B, n_obs, 4)
    comm_radius     : float
    node_dim, edge_dim : int

    Returns
    -------
    GraphsTuple — mega-graph with B * N_per nodes.
    """
    B, n, _ = agent_states.shape
    device = agent_states.device
    n_obs = obstacle_states.shape[1] if obstacle_states is not None and obstacle_states.ndim == 3 else 0
    N_per = n * 2 + n_obs

    # ---- Node features (one-hot, tiled) ----
    nf = torch.zeros(N_per, node_dim, device=device)
    nf[:n, 0] = 1.0
    nf[n:2*n, 1] = 1.0
    if n_obs > 0:
        nf[2*n:, 2] = 1.0
    node_feats = nf.unsqueeze(0).expand(B, -1, -1).reshape(B * N_per, node_dim)

    nt_parts = [
        torch.zeros(n, dtype=torch.long, device=device),
        torch.ones(n, dtype=torch.long, device=device),
    ]
    if n_obs > 0:
        nt_parts.append(torch.full((n_obs,), 2, dtype=torch.long, device=device))
    node_types = torch.cat(nt_parts).unsqueeze(0).expand(B, -1).reshape(B * N_per)

    # ---- All non-agent states ----
    if n_obs > 0:
        non_agent = torch.cat([goal_states, obstacle_states], dim=1)  # (B, n+n_obs, 4)
    else:
        non_agent = goal_states  # (B, n, 4)
    K = non_agent.shape[1]

    # ---- Build all edges: (agents + non_agents) → agents ----
    senders_parts = []
    receivers_parts = []
    edges_parts = []

    # Agent–Agent edges
    if n > 1:
        ap = agent_states[:, :, :2]  # (B, n, 2)
        diff_aa = ap.unsqueeze(2) - ap.unsqueeze(1)  # (B, n, n, 2)
        dist_aa = torch.norm(diff_aa, dim=-1)         # (B, n, n)
        aa_mask = dist_aa <= comm_radius
        eye = torch.eye(n, dtype=torch.bool, device=device)
        aa_mask = aa_mask & ~eye.unsqueeze(0)

        b_idx, s_idx, r_idx = torch.where(aa_mask)
        if b_idx.numel() > 0:
            dpos = agent_states[b_idx, s_idx, :2] - agent_states[b_idx, r_idx, :2]
            dvel = agent_states[b_idx, s_idx, 2:4] - agent_states[b_idx, r_idx, 2:4]
            senders_parts.append(s_idx + b_idx * N_per)
            receivers_parts.append(r_idx + b_idx * N_per)
            edges_parts.append(torch.cat([dpos, dvel], dim=-1))

    # Non-agent → Agent edges
    pos_na = non_agent[:, :, :2]           # (B, K, 2)
    pos_a = agent_states[:, :, :2]         # (B, n, 2)
    diff_na = pos_na.unsqueeze(2) - pos_a.unsqueeze(1)  # (B, K, n, 2)
    dist_na = torch.norm(diff_na, dim=-1)  # (B, K, n)
    na_mask = dist_na <= comm_radius
    b_na, k_na, a_na = torch.where(na_mask)
    if b_na.numel() > 0:
        dpos = non_agent[b_na, k_na, :2] - agent_states[b_na, a_na, :2]
        dvel = non_agent[b_na, k_na, 2:4] - agent_states[b_na, a_na, 2:4]
        senders_parts.append((n + k_na) + b_na * N_per)
        receivers_parts.append(a_na + b_na * N_per)
        edges_parts.append(torch.cat([dpos, dvel], dim=-1))

    # ---- Assemble ----
    if senders_parts:
        all_s = torch.cat(senders_parts)
        all_r = torch.cat(receivers_parts)
        all_e = torch.cat(edges_parts)
    else:
        all_s = torch.zeros(0, dtype=torch.long, device=device)
        all_r = torch.zeros(0, dtype=torch.long, device=device)
        all_e = torch.zeros(0, edge_dim, device=device)

    return GraphsTuple(
        nodes=node_feats,
        edges=all_e,
        senders=all_s,
        receivers=all_r,
        n_node=B * N_per,
        n_edge=all_s.shape[0],
        node_type=node_types,
    )
