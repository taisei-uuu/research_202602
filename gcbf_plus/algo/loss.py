"""
Loss functions for Hierarchical Velocity-Command Swarm training.

    L_progress: Reward velocity toward goal (dot product with goal direction)
    L_qp:       QP-intervention penalty (penalize ||u_QP - u_nom||²)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def compute_affine_loss(
    pi_action: torch.Tensor,
    u_nom: torch.Tensor,
    u_qp: torch.Tensor,
    v_target: torch.Tensor,
    goal_dir: torch.Tensor,
    coef_progress: float = 1.0,
    coef_qp: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the hierarchical velocity-command training loss.

    Parameters
    ----------
    pi_action : (N, 3)
        GNN output π_φ(x) = (Δv_x, Δv_y, Δṡ).
    u_nom : (N, 3)
        Nominal acceleration [a_cx, a_cy, a_s] (has gradient through GNN).
    u_qp : (N, 3)
        QP-corrected safe accelerations (detached).
    v_target : (N, 2)
        Target velocity (LQR + GNN offset), has gradient through GNN.
    goal_dir : (N, 2)
        Direction vector from agent to goal (goal_pos - agent_pos), detached.
    coef_progress, coef_qp : float
        Loss coefficients.

    Returns
    -------
    total_loss : scalar tensor
    info : dict of scalar loss values for logging
    """
    # ── L_progress: reward velocity component toward goal ──
    # Normalize goal direction to unit vector
    goal_dist = goal_dir.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    goal_hat = goal_dir / goal_dist  # (N, 2) unit vector toward goal

    # Dot product: how much of v_target points toward goal
    progress = (v_target * goal_hat).sum(dim=-1)  # (N,) scalar per agent
    loss_progress = -progress.mean()  # maximize → minimize negative

    # ── L_qp: penalize QP correction ──
    # When QP modifies u_nom, GNN should learn to anticipate and avoid it
    loss_qp = (u_qp - u_nom).pow(2).sum(dim=-1).mean()

    # ── Total ──
    total_loss = coef_progress * loss_progress + coef_qp * loss_qp

    info = {
        "loss/total": total_loss.item(),
        "loss/progress": loss_progress.item(),
        "loss/qp": loss_qp.item(),
    }

    return total_loss, info
