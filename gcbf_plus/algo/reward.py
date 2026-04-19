"""
Reward functions for Hierarchical Velocity-Command Swarm training.

    R_progress: Reward actual distance reduction toward goal (one-step lookahead).
                Gradient flows through v_target → pi_scaled → GNN.
    R_arrival:  Sparse bonus when agent is within arrival_radius of goal (no grad).
    R_qp:       QP-intervention penalty (penalize ||u_QP - u_nom||²)
    R_avoid:    Proactive obstacle avoidance reward.
                Penalizes approach velocity toward nearby obstacles,
                weighted by exp(-d/σ) so the signal is felt from far away.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_reward(
    pi_action: torch.Tensor,
    u_nom: torch.Tensor,
    u_qp: torch.Tensor,
    dist_reduction: torch.Tensor,
    dist_to_goal: Optional[torch.Tensor] = None,
    agent_pos: Optional[torch.Tensor] = None,
    agent_vel: Optional[torch.Tensor] = None,
    obs_centers: Optional[torch.Tensor] = None,
    obs_radii: Optional[torch.Tensor] = None,
    coef_progress: float = 1.0,
    coef_qp: float = 2.0,
    coef_avoid: float = 1.0,
    avoid_sigma: float = 1.5,
    coef_arrival: float = 5.0,
    arrival_radius: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the hierarchical velocity-command training reward (as minimizable loss).

    Parameters
    ----------
    pi_action : (N, 3)
        GNN output π_φ(x) = (Δv_x, Δv_y, Δṡ), in [-1,1] (tanh).
    u_nom : (N, 3)
        Nominal acceleration [a_cx, a_cy, a_s] (has gradient through GNN).
    u_qp : (N, 3)
        QP-corrected safe accelerations (detached).
    dist_reduction : (N,)
        Actual distance decrease per step (has gradient through GNN).
    dist_to_goal : (N,) or None
        Current distance to goal (detached). Used for arrival bonus.
    agent_pos : (N, 2) or None
        Agent positions. Required for R_avoid.
    agent_vel : (N, 2) or None
        Agent velocities. Required for R_avoid.
    obs_centers : (N, n_obs, 2) or None
        Obstacle center positions per agent.
    obs_radii : (N, n_obs) or None
        Obstacle radii per agent.
    coef_progress : float
        Weight for goal progress reward.
    coef_qp : float
        Weight for QP intervention penalty.
    coef_avoid : float
        Weight for proactive obstacle avoidance reward.
    avoid_sigma : float
        Distance scale [m] for exponential approach penalty.
        Signal felt within ~3σ of obstacle surface.
    coef_arrival : float
        Weight for sparse arrival bonus.
    arrival_radius : float
        Goal-reached threshold in metres.

    Returns
    -------
    total_loss : scalar tensor (negative reward, minimized by optimizer)
    info : dict of scalar reward values for logging
    """
    # ── R_progress: reward actual distance reduction ──
    r_progress = dist_reduction.mean()

    # ── R_arrival: sparse bonus when within arrival_radius (no grad) ──
    r_arrival = torch.tensor(0.0, device=pi_action.device)
    if coef_arrival > 0.0 and dist_to_goal is not None:
        arrived = (dist_to_goal < arrival_radius).float()
        r_arrival = arrived.mean()

    # ── R_qp: penalize QP correction (all 3 axes) ──
    r_qp = -(u_qp - u_nom).pow(2).sum(dim=-1).mean()

    # ── R_avoid: proactive obstacle avoidance ──
    r_avoid = torch.tensor(0.0, device=pi_action.device)
    if (coef_avoid > 0.0
            and agent_pos is not None
            and agent_vel is not None
            and obs_centers is not None
            and obs_radii is not None
            and obs_centers.shape[1] > 0):

        # outward normal: (N, n_obs, 2)
        diff = agent_pos.unsqueeze(1) - obs_centers          # (N, n_obs, 2)
        dist_center = diff.norm(dim=-1).clamp(min=1e-6)      # (N, n_obs)
        n_out = diff / dist_center.unsqueeze(-1)             # (N, n_obs, 2)

        # distance to obstacle surface
        dist_surface = (dist_center - obs_radii).clamp(min=0.0)  # (N, n_obs)

        # approach rate: positive when moving toward obstacle
        approach = (-agent_vel.unsqueeze(1) * n_out).sum(dim=-1).clamp(min=0.0)  # (N, n_obs)

        weight = torch.exp(-dist_surface / avoid_sigma)       # (N, n_obs)
        r_avoid = -(approach * weight).sum(dim=-1).mean()

    # ── Total reward (negated for minimization) ──
    total_reward = (
        coef_progress * r_progress
        + coef_arrival * r_arrival
        + coef_qp     * r_qp
        + coef_avoid  * r_avoid
    )
    total_loss = -total_reward

    info = {
        "reward/total":    total_reward.item(),
        "reward/progress": r_progress.item(),
        "reward/arrival":  r_arrival.item(),
        "reward/qp":       r_qp.item(),
        "reward/avoid":    r_avoid.item(),
    }

    return total_loss, info
