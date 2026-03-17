"""
Loss functions for Hierarchical Velocity-Command Swarm training.

    L_goal:   Penalize GNN translation offset (keep velocity close to LQR)
    L_qp:     QP-intervention penalty (penalize ||u_QP - u_nom||²)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def compute_affine_loss(
    pi_action: torch.Tensor,
    u_nom: torch.Tensor,
    u_qp: torch.Tensor,
    coef_goal: float = 1.0,
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
    coef_goal, coef_qp : float
        Loss coefficients.

    Returns
    -------
    total_loss : scalar tensor
    info : dict of scalar loss values for logging
    """
    # ── L_goal: penalize GNN offset from nominal controllers ──
    # pi_action[:, :2] = (Δv_x, Δv_y), keep close to zero = follow LQR
    # pi_action[:, 2]  = Δṡ, keep close to zero = follow expansion PD
    loss_goal = (pi_action ** 2).sum(dim=-1).mean()

    # ── L_qp: penalize QP correction ──
    # When QP modifies u_nom, GNN should learn to anticipate and avoid it
    loss_qp = (u_qp - u_nom).pow(2).sum(dim=-1).mean()

    # ── Total ──
    total_loss = coef_goal * loss_goal + coef_qp * loss_qp

    info = {
        "loss/total": total_loss.item(),
        "loss/goal": loss_goal.item(),
        "loss/qp": loss_qp.item(),
    }

    return total_loss, info
