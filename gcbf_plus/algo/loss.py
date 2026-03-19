"""
Loss functions for Hierarchical Velocity-Command Swarm training.

    L_progress: Reward actual distance reduction toward goal (one-step lookahead).
                Gradient flows through v_target → pi_scaled → GNN.
    L_arrival:  Sparse bonus when agent is within arrival_radius of goal (no grad).
    L_qp:       QP-intervention penalty (penalize ||u_QP - u_nom||²)
    L_effort:   GNN offset effort penalty (prefer small deviations from nominal)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_affine_loss(
    pi_action: torch.Tensor,
    u_nom: torch.Tensor,
    u_qp: torch.Tensor,
    dist_reduction: torch.Tensor,
    dist_to_goal: Optional[torch.Tensor] = None,
    coef_progress: float = 1.0,
    coef_qp: float = 2.0,
    coef_effort: float = 0.3,
    w_scale: float = 2.0,
    coef_arrival: float = 5.0,
    arrival_radius: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the hierarchical velocity-command training loss.

    Parameters
    ----------
    pi_action : (N, 3)
        GNN output π_φ(x) = (Δv_x, Δv_y, Δṡ), in [-1,1] (tanh).
    u_nom : (N, 3)
        Nominal acceleration [a_cx, a_cy, a_s] (has gradient through GNN).
    u_qp : (N, 3)
        QP-corrected safe accelerations (detached).
    dist_reduction : (N,)
        Actual distance decrease per step: dist_now - dist_next.
        dist_next = ||goal - (pos + v_target * dt)||, has gradient through GNN.
    dist_to_goal : (N,) or None
        Current distance to goal (detached, from pool state). Used for arrival bonus.
    coef_progress : float
        Weight for goal progress reward.
    coef_qp : float
        Weight for QP intervention penalty.
    coef_effort : float
        Weight for GNN offset effort penalty.
    w_scale : float
        Extra weight on scale offset in L_effort.
        Higher → GNN prefers translation over shrinking.
    coef_arrival : float
        Weight for sparse arrival bonus (default 5.0).
    arrival_radius : float
        Goal-reached threshold in metres (default 0.3m).

    Returns
    -------
    total_loss : scalar tensor
    info : dict of scalar loss values for logging
    """
    # ── L_progress: reward actual distance reduction ──
    # dist_reduction > 0 means moving toward goal; gradient through v_target → GNN
    loss_progress = -dist_reduction.mean()

    # ── L_arrival: sparse bonus when within arrival_radius (no grad) ──
    loss_arrival = torch.tensor(0.0, device=pi_action.device)
    if coef_arrival > 0.0 and dist_to_goal is not None:
        arrived = (dist_to_goal < arrival_radius).float()
        loss_arrival = -arrived.mean()  # negative → bonus when arrived

    # ── L_qp: penalize QP correction (all 3 axes) ──
    loss_qp = (u_qp - u_nom).pow(2).sum(dim=-1).mean()

    # ── L_effort: penalize GNN offset magnitude ──
    # Δv_x² + Δv_y² + w_scale · Δṡ²
    effort_trans = pi_action[:, :2].pow(2).sum(dim=-1)
    effort_scale = w_scale * pi_action[:, 2].pow(2)
    loss_effort = (effort_trans + effort_scale).mean()

    # ── Total ──
    total_loss = (
        coef_progress * loss_progress
        + coef_arrival * loss_arrival
        + coef_qp * loss_qp
        + coef_effort * loss_effort
    )

    info = {
        "loss/total": total_loss.item(),
        "loss/progress": loss_progress.item(),
        "loss/arrival": loss_arrival.item(),
        "loss/qp": loss_qp.item(),
        "loss/effort": loss_effort.item(),
    }

    return total_loss, info
