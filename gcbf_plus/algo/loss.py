"""
Loss functions for Hierarchical Velocity-Command Swarm training.

    L_goal:   Penalize GNN translation offset (keep velocity close to LQR)
    L_qp:     QP-intervention penalty (penalize ||u_QP - u_nom||²)
    L_scale:  Scale expansion incentive: -mean(ṡ_target * (s_max - s))
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def compute_affine_loss(
    pi_action: torch.Tensor,
    u_nom: torch.Tensor,
    u_qp: torch.Tensor,
    s_current: torch.Tensor,
    s_dot_target: torch.Tensor,
    s_max: float,
    coef_goal: float = 1.0,
    coef_qp: float = 2.0,
    coef_scale: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the hierarchical velocity-command training loss.

    Parameters
    ----------
    pi_action : (N, 3)
        GNN output π_φ(x) = (Δv_x, Δv_y, ṡ_target).
    u_nom : (N, 3)
        Nominal acceleration [a_cx, a_cy, a_s] (has gradient through GNN).
    u_qp : (N, 3)
        QP-corrected safe accelerations (detached).
    s_current : (N,)
        Current scale value per agent.
    s_dot_target : (N,)
        GNN's scale rate output ṡ_target (has gradient).
    s_max : float
        Maximum scale value.
    coef_goal, coef_qp, coef_scale : float
        Loss coefficients.

    Returns
    -------
    total_loss : scalar tensor
    info : dict of scalar loss values for logging
    """
    # ── L_goal: penalize translation velocity offset from LQR ──
    # pi_action[:, :2] = (Δv_x, Δv_y), keep close to zero = follow LQR
    loss_goal = (pi_action[:, :2] ** 2).sum(dim=-1).mean()

    # ── L_qp: penalize QP correction on TRANSLATION only ──
    # Scale corrections are excluded so L_qp doesn't incentivize shrinking
    qp_correction_trans = u_qp[:, :2] - u_nom[:, :2]
    loss_qp = (qp_correction_trans ** 2).sum(dim=-1).mean()

    # ── L_scale: incentivize expansion toward s_max ──
    # -mean(ṡ_target * (s_max - s))
    # When s < s_max: positive ṡ → loss decreases (good)
    # When s ≈ s_max: term ≈ 0 regardless
    loss_scale = -(s_dot_target * (s_max - s_current)).mean()

    # ── Total ──
    total_loss = (
        coef_goal * loss_goal
        + coef_qp * loss_qp
        + coef_scale * loss_scale
    )

    info = {
        "loss/total": total_loss.item(),
        "loss/goal": loss_goal.item(),
        "loss/qp": loss_qp.item(),
        "loss/scale": loss_scale.item(),
    }

    return total_loss, info
