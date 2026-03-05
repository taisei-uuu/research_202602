"""
Loss functions for Affine-Transform Swarm training.

New RL-style losses (no safe/unsafe labeling):
    L_goal:   Goal-reaching incentive (penalize distance to goal)
    L_qp:     QP-intervention penalty (penalize ‖u_QP - u_AT‖²)
    L_reg:    Action regularization (penalize ‖π(x)‖²)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_affine_loss(
    pi_action: torch.Tensor,
    u_at: torch.Tensor,
    u_qp: torch.Tensor,
    goal_dist: torch.Tensor,
    coef_goal: float = 1.0,
    coef_qp: float = 1.0,
    coef_reg: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the affine-transform training loss.

    Parameters
    ----------
    pi_action : (N, 3)
        GNN offset output π_φ(x).
    u_at : (N, 3)
        Full affine parameter u_AT = u_nom + π_φ(x).
    u_qp : (N, 3)
        QP-corrected safe affine parameters.
    goal_dist : (N,)
        Distance to goal for each agent (‖P_c - P_goal‖).
    coef_goal, coef_qp, coef_reg : float
        Loss coefficients.

    Returns
    -------
    total_loss : scalar tensor
    info : dict of scalar loss values for logging
    """
    # ── L_goal: penalize distance to goal (reward shaping) ──
    loss_goal = goal_dist.mean()

    # ── L_qp: penalize QP correction magnitude ──
    # This encourages the policy to output already-safe actions
    qp_correction = u_qp - u_at
    loss_qp = (qp_correction ** 2).sum(dim=-1).mean()

    # ── L_reg: L2 regularization on GNN offset ──
    loss_reg = (pi_action ** 2).sum(dim=-1).mean()

    # ── Total ──
    total_loss = (
        coef_goal * loss_goal
        + coef_qp * loss_qp
        + coef_reg * loss_reg
    )

    info = {
        "loss/total": total_loss.item(),
        "loss/goal": loss_goal.item(),
        "loss/qp": loss_qp.item(),
        "loss/reg": loss_reg.item(),
    }

    return total_loss, info
