"""
Swarm-aware graph building with drone-to-drone sensing.

Edge creation:  For agent-agent edges, compute 3×3=9 pairwise distances
between constituent drones.  An edge exists iff min(9_dist) < R_sensing.

Edge features (8D):
    [Δpx, Δpy, Δvx, Δvy, Δθ_wrapped, min_dist, closest_dx, closest_dy]
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .graph import GraphsTuple


# ------------------------------------------------------------------
# Rotation helpers
# ------------------------------------------------------------------

def _rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    """
    Build 2×2 rotation matrices from angles.

    Parameters
    ----------
    theta : (*batch,)

    Returns
    -------
    R : (*batch, 2, 2)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1),
    ], dim=-2)
    return R


def _compute_drone_positions(
    com_pos: torch.Tensor,
    theta: torch.Tensor,
    local_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute global positions of the 3 drones in each swarm.

    Parameters
    ----------
    com_pos : (N, 2)  — CoM positions
    theta   : (N,)    — yaw angles
    local_offsets : (3, 2) — local drone offsets (equilateral triangle)

    Returns
    -------
    drone_pos : (N, 3, 2) — global positions of each drone
    """
    R = _rotation_matrix(theta)          # (N, 2, 2)
    # Rotate offsets: (N, 2, 2) @ (3, 2).T → broadcast via einsum
    # rotated_offsets[i, k, :] = R[i] @ offsets[k]
    rotated = torch.einsum("nij,kj->nki", R, local_offsets)  # (N, 3, 2)
    drone_pos = com_pos.unsqueeze(1) + rotated               # (N, 3, 2)
    return drone_pos


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------
# Local offsets for equilateral triangle
# ------------------------------------------------------------------

def get_equilateral_offsets(R_form: float, device: torch.device) -> torch.Tensor:
    """
    Return (3, 2) local offsets for drones at vertices of an equilateral
    triangle with circumradius R_form.
    """
    s32 = math.sqrt(3.0) / 2.0
    offsets = torch.tensor(
        [
            [R_form, 0.0],
            [-R_form / 2.0,  R_form * s32],
            [-R_form / 2.0, -R_form * s32],
        ],
        dtype=torch.float32,
        device=device,
    )
    return offsets


# ------------------------------------------------------------------
# Main graph builder
# ------------------------------------------------------------------

def build_swarm_graph_from_states(
    agent_states: torch.Tensor,
    goal_states: torch.Tensor,
    obstacle_positions: Optional[torch.Tensor],
    comm_radius: float,
    node_dim: int = 3,
    edge_dim: int = 8,
    R_form: float = 0.3,
) -> GraphsTuple:
    """
    Build a ``GraphsTuple`` for swarm agents with drone-to-drone sensing.

    Parameters
    ----------
    agent_states : (n_agents, 6)  — [px, py, vx, vy, θ, ω]
    goal_states  : (n_agents, 6)  — [px, py, vx, vy, θ_goal, ω_goal]
    obstacle_positions : (n_obs, 6) or None
    comm_radius : float — R_sensing for drone-to-drone edge creation
    node_dim : int — one-hot node indicator dimension (default 3)
    edge_dim : int — edge feature dimension (default 8)
    R_form : float — formation circumradius

    Returns
    -------
    GraphsTuple with 8D edge features using drone-to-drone sensing.
    """
    n_agents = agent_states.shape[0]
    device = agent_states.device
    local_offsets = get_equilateral_offsets(R_form, device)  # (3, 2)

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

    all_states_cat = torch.cat(all_states, dim=0)  # (N_total, 6)
    node_type_vec = torch.cat(node_types, dim=0)    # (N_total,)
    n_total = all_states_cat.shape[0]

    # ---- One-hot node features ----
    node_feats = torch.zeros(n_total, node_dim, device=device)
    for t in range(node_dim):
        node_feats[node_type_vec == t, t] = 1.0

    # ---- Compute drone positions for all agents ----
    agent_pos = agent_states[:, :2]     # (n_agents, 2)
    agent_theta = agent_states[:, 4]    # (n_agents,)
    agent_drone_pos = _compute_drone_positions(
        agent_pos, agent_theta, local_offsets
    )  # (n_agents, 3, 2)

    # ==================================================================
    # Agent-to-Agent edges (drone-to-drone sensing)
    # ==================================================================
    senders_list = []
    receivers_list = []
    edge_feats_list = []

    if n_agents > 1:
        # Pairwise drone distances: (n_agents, n_agents, 3, 3)
        # drone_pos_i[i, k] vs drone_pos_j[j, l]
        # Expand: (n, 1, 3, 2) - (1, n, 3, 2) → (n, n, 3, 3, 2)
        diff_all = (
            agent_drone_pos[:, None, :, None, :]    # (n, 1, 3, 1, 2)
            - agent_drone_pos[None, :, None, :, :]  # (1, n, 1, 3, 2)
        )  # (n, n, 3, 3, 2)
        dist_all = torch.norm(diff_all, dim=-1)  # (n, n, 3, 3)

        # Min over the 9 pairs → (n, n)
        dist_min, flat_idx = dist_all.reshape(n_agents, n_agents, -1).min(dim=-1)
        # flat_idx ∈ [0, 8] → unravel to (k_closest, l_closest)
        k_closest = flat_idx // 3  # which drone in swarm i
        l_closest = flat_idx % 3   # which drone in swarm j

        # Edge mask: within comm_radius, no self-loops
        mask = dist_min <= comm_radius
        mask.fill_diagonal_(False)

        sender_idx, receiver_idx = torch.where(mask)  # (E_aa,)

        if sender_idx.numel() > 0:
            # CoM-based relative features
            delta_pos = agent_states[sender_idx, :2] - agent_states[receiver_idx, :2]    # (E, 2)
            delta_vel = agent_states[sender_idx, 2:4] - agent_states[receiver_idx, 2:4]  # (E, 2)
            delta_theta = _wrap_angle(
                agent_states[sender_idx, 4] - agent_states[receiver_idx, 4]
            )  # (E,)

            # Closest drone pair info
            min_d = dist_min[sender_idx, receiver_idx]  # (E,)

            # Gather the positions of the closest drones
            batch_idx_e = torch.arange(sender_idx.numel(), device=device)
            k_sel = k_closest[sender_idx, receiver_idx]  # (E,)
            l_sel = l_closest[sender_idx, receiver_idx]  # (E,)

            closest_pos_i = agent_drone_pos[sender_idx, k_sel]    # (E, 2)
            closest_pos_j = agent_drone_pos[receiver_idx, l_sel]  # (E, 2)
            closest_delta = closest_pos_i - closest_pos_j          # (E, 2)

            edge_feat = torch.stack([
                delta_pos[:, 0], delta_pos[:, 1],
                delta_vel[:, 0], delta_vel[:, 1],
                delta_theta,
                min_d,
                closest_delta[:, 0], closest_delta[:, 1],
            ], dim=-1)  # (E, 8)

            senders_list.append(sender_idx)
            receivers_list.append(receiver_idx)
            edge_feats_list.append(edge_feat)

    # ==================================================================
    # Goal/Obstacle → Agent edges (CoM-based, standard radius)
    # ==================================================================
    non_agent_start = n_agents  # goals start here
    non_agent_states = all_states_cat[non_agent_start:]  # (n_goals + n_obs, 6)
    n_non_agent = non_agent_states.shape[0]

    if n_non_agent > 0 and n_agents > 0:
        # Distances: non-agent CoM → agent CoM  (n_non_agent, n_agents)
        pos_non_agent = non_agent_states[:, :2]
        diff_na = pos_non_agent.unsqueeze(1) - agent_pos.unsqueeze(0)  # (K, n_agents, 2)
        dist_na = torch.norm(diff_na, dim=-1)  # (K, n_agents)

        mask_na = dist_na <= comm_radius
        na_sender, na_receiver = torch.where(mask_na)

        if na_sender.numel() > 0:
            # Global sender indices (offset by n_agents)
            global_sender = na_sender + non_agent_start

            delta_pos = all_states_cat[global_sender, :2] - agent_states[na_receiver, :2]
            delta_vel = all_states_cat[global_sender, 2:4] - agent_states[na_receiver, 2:4]
            delta_theta = _wrap_angle(
                all_states_cat[global_sender, 4] - agent_states[na_receiver, 4]
            )

            # For goal/obstacle → agent, min_dist = CoM distance, closest = CoM delta
            min_d = dist_na[na_sender, na_receiver]
            closest_dx = delta_pos[:, 0]
            closest_dy = delta_pos[:, 1]

            edge_feat = torch.stack([
                delta_pos[:, 0], delta_pos[:, 1],
                delta_vel[:, 0], delta_vel[:, 1],
                delta_theta,
                min_d,
                closest_dx, closest_dy,
            ], dim=-1)  # (E, 8)

            senders_list.append(global_sender)
            receivers_list.append(na_receiver)
            edge_feats_list.append(edge_feat)

    # ---- Assemble ----
    if len(senders_list) > 0:
        all_senders = torch.cat(senders_list, dim=0)
        all_receivers = torch.cat(receivers_list, dim=0)
        all_edges = torch.cat(edge_feats_list, dim=0)
    else:
        all_senders = torch.zeros(0, dtype=torch.long, device=device)
        all_receivers = torch.zeros(0, dtype=torch.long, device=device)
        all_edges = torch.zeros(0, edge_dim, device=device)

    return GraphsTuple(
        nodes=node_feats,
        edges=all_edges,
        senders=all_senders,
        receivers=all_receivers,
        n_node=n_total,
        n_edge=all_senders.shape[0],
        node_type=node_type_vec,
    )
