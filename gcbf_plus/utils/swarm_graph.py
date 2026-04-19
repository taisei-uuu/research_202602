"""
Simplified swarm graph building — Bounding Circle approach.

Each swarm is treated as a point mass with a bounding circle of radius
r_swarm.  No rotation, no 9-point drone distances.

Edge creation:  CoM distance < R_sensing(s) = comm_base * s
Edge features (5D):  [Δpx, Δpy, Δvx, Δvy, dist_surface]
  dist_surface = max(0, ||Δp|| - obs_radius) for obstacle edges, else 0.
  Obstacle velocity is treated as 0 (Δvx, Δvy = 0 - agent_vel).
"""

from __future__ import annotations

import math
from typing import Optional, Union

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
    obstacle_positions: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    comm_radius: Union[float, torch.Tensor],
    node_dim: int = 3,
    edge_dim: int = 4,
    payload_states: Optional[torch.Tensor] = None,  # (n_agents, 4): [gx, gy, gx_dot, gy_dot]
) -> GraphsTuple:
    """
    Build a GraphsTuple for swarm agents using LiDAR-based hit points.

    Parameters
    ----------
    agent_states    : (n_agents, 4)
    goal_states     : (n_agents, 4)
    obstacle_positions : List[torch.Tensor] (per-agent hits) or (n_obs, 4) tensor
    comm_radius     : float or (n_agents,) tensor
    payload_states  : (n_agents, 4) optional — appended to agent node features when provided
    """
    n_agents = agent_states.shape[0]
    device = agent_states.device

    # ---- Collect all nodes ----
    all_states = [agent_states, goal_states]
    node_types = [
        torch.zeros(n_agents, dtype=torch.long, device=device),
        torch.ones(n_agents, dtype=torch.long, device=device),
    ]

    # Handle per-agent obstacle hit points (LiDAR style)
    if isinstance(obstacle_positions, list):
        # Flatten all hit points into the global node list
        for hits in obstacle_positions:
            if hits.shape[0] > 0:
                all_states.append(hits)
                node_types.append(
                    torch.full((hits.shape[0],), 2, dtype=torch.long, device=device)
                )
    elif obstacle_positions is not None and obstacle_positions.shape[0] > 0:
        # Backward compatibility for global center nodes
        all_states.append(obstacle_positions)
        node_types.append(
            torch.full((obstacle_positions.shape[0],), 2, dtype=torch.long, device=device)
        )

    all_states_cat = torch.cat(all_states, dim=0)   # (N_total, 4)
    node_type_vec = torch.cat(node_types, dim=0)
    n_total = all_states_cat.shape[0]

    # ---- One-hot node features (first 3 dims) + optional payload (dims 3-6) ----
    node_feats = torch.zeros(n_total, node_dim, device=device)
    for t in range(3):  # one-hot part is always 3D
        node_feats[node_type_vec == t, t] = 1.0
    # Append payload state [gx, gy, gx_dot, gy_dot] to agent nodes (type 0)
    if payload_states is not None and node_dim == 7:
        node_feats[:n_agents, 3:7] = payload_states[:, :4].to(device)

    # ---- Build Mask Matrix (N_total, n_agents) ----
    # mask[i, j] = True means there is a sender-edge from node i to agent j
    mask = torch.zeros((n_total, n_agents), dtype=torch.bool, device=device)

    # 1. Agent-Agent edges (CoM distance)
    agent_pos = agent_states[:, :2]                        # (n, 2)
    # distance between first n_agents and all n_agents
    dist_aa = torch.norm(agent_pos.unsqueeze(1) - agent_pos.unsqueeze(0), dim=-1) # (n, n)
    
    if isinstance(comm_radius, torch.Tensor):
        cr = comm_radius.unsqueeze(0)  # (1, n)
    else:
        cr = comm_radius
    
    aa_mask = dist_aa <= cr
    for i in range(n_agents):
        aa_mask[i, i] = False # No self-loops
    mask[:n_agents, :] = aa_mask

    # 2. Agent-Goal Visibility (Identity)
    for i in range(n_agents):
        mask[n_agents + i, i] = True

    # 3. Agent-Obstacle (LiDAR Hits)
    if isinstance(obstacle_positions, list):
        current_idx = 2 * n_agents
        for i in range(n_agents):
            n_hits = obstacle_positions[i].shape[0]
            if n_hits > 0:
                # Agent i only sees its own hit points
                mask[current_idx : current_idx + n_hits, i] = True
                current_idx += n_hits
    elif obstacle_positions is not None and obstacle_positions.shape[0] > 0:
        # Global distance-based fallback
        obs_pos = all_states_cat[2 * n_agents :, :2]
        dist_oa = torch.norm(obs_pos.unsqueeze(1) - agent_pos.unsqueeze(0), dim=-1)
        mask[2 * n_agents :, :] = (dist_oa <= cr)

    s_idx, r_idx = torch.where(mask)

    if s_idx.numel() > 0:
        delta_pos = all_states_cat[s_idx, :2] - agent_states[r_idx, :2]
        # Obstacle nodes (type 2) have radius in slot [2], not velocity.
        # Use 0 as obstacle velocity so Δv = -agent_vel.
        is_obs = node_type_vec[s_idx] == 2
        vel_src = all_states_cat[s_idx, 2:4].clone()
        vel_src[is_obs] = 0.0
        delta_vel = vel_src - agent_states[r_idx, 2:4]
        # 5th feature: surface distance for obstacle edges, 0 otherwise
        dist_surface = torch.zeros(s_idx.shape[0], 1, device=device)
        if is_obs.any():
            obs_r = all_states_cat[s_idx[is_obs], 2]   # radius stored at slot [2]
            dist_surface[is_obs, 0] = (
                delta_pos[is_obs].norm(dim=-1) - obs_r
            ).clamp(min=0.0)
        edges = torch.cat([delta_pos, delta_vel, dist_surface], dim=-1)  # (E, 5)
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
    comm_radius: Union[float, torch.Tensor],
    node_dim: int = 3,
    edge_dim: int = 4,
    payload_states: Optional[torch.Tensor] = None,  # (B, n_agents, 4)
) -> GraphsTuple:
    """
    Build ONE mega-graph from B independent swarm environments.

    Parameters
    ----------
    agent_states    : (B, n_agents, 4)
    goal_states     : (B, n_agents, 4)
    obstacle_states : (B, n_obs, 4)
    comm_radius     : float or (B, n_agents) tensor
        If tensor, each agent in each batch has its own sensing radius.
    node_dim, edge_dim : int
    payload_states  : (B, n_agents, 4) optional — appended to agent node features when provided

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
    # Tile across batch: (B, N_per, node_dim), then add payload per-agent
    node_feats = nf.unsqueeze(0).expand(B, -1, -1).clone().reshape(B, N_per, node_dim)
    # Append payload state [gx, gy, gx_dot, gy_dot] to agent nodes when provided
    if payload_states is not None and node_dim == 7:
        node_feats[:, :n, 3:7] = payload_states[:, :, :4]
    node_feats = node_feats.reshape(B * N_per, node_dim)

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

    # Prepare dynamic comm_radius threshold
    # comm_radius_bn: (B, n) — per-receiver agent threshold
    if isinstance(comm_radius, torch.Tensor):
        comm_radius_bn = comm_radius  # (B, n)
    else:
        comm_radius_bn = None  # use scalar

    # Agent–Agent edges
    if n > 1:
        ap = agent_states[:, :, :2]  # (B, n, 2)
        diff_aa = ap.unsqueeze(2) - ap.unsqueeze(1)  # (B, n, n, 2)
        dist_aa = torch.norm(diff_aa, dim=-1)         # (B, n, n)

        if comm_radius_bn is not None:
            # Per-receiver threshold: (B, 1, n) — receiver is dim 2
            aa_mask = dist_aa <= comm_radius_bn.unsqueeze(1)  # (B, n, n)
        else:
            aa_mask = dist_aa <= comm_radius

        eye = torch.eye(n, dtype=torch.bool, device=device)
        aa_mask = aa_mask & ~eye.unsqueeze(0)

        b_idx, s_idx, r_idx = torch.where(aa_mask)
        if b_idx.numel() > 0:
            dpos = agent_states[b_idx, s_idx, :2] - agent_states[b_idx, r_idx, :2]
            dvel = agent_states[b_idx, s_idx, 2:4] - agent_states[b_idx, r_idx, 2:4]
            zeros5 = torch.zeros(b_idx.shape[0], 1, device=device)
            senders_parts.append(s_idx + b_idx * N_per)
            receivers_parts.append(r_idx + b_idx * N_per)
            edges_parts.append(torch.cat([dpos, dvel, zeros5], dim=-1))

    # Non-agent → Agent edges
    pos_na = non_agent[:, :, :2]           # (B, K, 2)
    pos_a = agent_states[:, :, :2]         # (B, n, 2)
    diff_na = pos_na.unsqueeze(2) - pos_a.unsqueeze(1)  # (B, K, n, 2)
    dist_na = torch.norm(diff_na, dim=-1)  # (B, K, n)

    if comm_radius_bn is not None:
        # Per-receiver threshold: (B, 1, n)
        na_mask = dist_na <= comm_radius_bn.unsqueeze(1)  # (B, K, n)
    else:
        na_mask = dist_na <= comm_radius

    # ---- NEW: Global Goal Visibility ----
    # The first `n` elements of the `K` dimension in `non_agent` are the goal nodes.
    # We want goal `i` to always be visible to agent `i`.
    # Create an identity matrix to represent this one-to-one mapping
    # and force those elements in the mask to be True.
    goal_identity = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
    
    # K could be n (no obstacles) or n + n_obs. We just overwrite the first nx_n block.
    na_mask[:, :n, :] = na_mask[:, :n, :] | goal_identity

    b_na, k_na, a_na = torch.where(na_mask)
    if b_na.numel() > 0:
        dpos = non_agent[b_na, k_na, :2] - agent_states[b_na, a_na, :2]
        # k_na >= n → obstacle node: radius is in slot [2], velocity = 0
        is_obs = k_na >= n
        vel_src = non_agent[b_na, k_na, 2:4].clone()
        vel_src[is_obs] = 0.0
        dvel = vel_src - agent_states[b_na, a_na, 2:4]
        # 5th feature: surface distance for obstacle edges, 0 for goal edges
        dist_surface = torch.zeros(b_na.shape[0], 1, device=device)
        if is_obs.any():
            obs_r = non_agent[b_na[is_obs], k_na[is_obs], 2]  # radius at slot [2]
            dist_surface[is_obs, 0] = (
                dpos[is_obs].norm(dim=-1) - obs_r
            ).clamp(min=0.0)
        senders_parts.append((n + k_na) + b_na * N_per)
        receivers_parts.append(a_na + b_na * N_per)
        edges_parts.append(torch.cat([dpos, dvel, dist_surface], dim=-1))

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
