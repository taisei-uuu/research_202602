"""
Batched CBF-QP solver in pure PyTorch — GPU-accelerated, no cvxpy.

Solves per-agent (in parallel via batched tensor ops):

    min_u  ½ ‖u − u_nom‖²
    s.t.   L_g h · u + L_f h + α h(x) ≥ 0       (CBF constraint)
           u_min ≤ u ≤ u_max                      (optional input bounds)

Closed-form via KKT (single linear inequality = half-space projection):

    c = L_g h · u_nom + L_f h + α h
    If c ≥ 0:  constraint satisfied → u* = u_nom
    If c < 0:  λ = −c / ‖L_g h‖²,  u* = u_nom + λ · L_g h

When input bounds are present, the solution is clipped after projection.
If clipping breaks the CBF constraint, the solver falls back to an
iterative projected correction (a few gradient-ascent steps on the
dual) to find a feasible point inside the box.

All operations are batched over agents (and optionally over batch dim)
using standard torch ops — runs entirely on GPU.
"""

from __future__ import annotations

from typing import Optional

import torch


def solve_cbf_qp_batched(
    u_nom: torch.Tensor,
    h: torch.Tensor,
    dh_dx: torch.Tensor,
    x_dot_f: torch.Tensor,
    B_mat: torch.Tensor,
    alpha: float = 1.0,
    u_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Batched analytical CBF-QP solver.

    Parameters
    ----------
    u_nom : (n_agents, action_dim)
        Nominal (LQR) control input — the goal-seeking reference.
    h : (n_agents,)
        CBF value for each agent.
    dh_dx : (n_agents, state_dim)
        Gradient ∂h/∂x for each agent.
    x_dot_f : (n_agents, state_dim)
        Drift dynamics f(x) = [vx, vy, 0, 0].
    B_mat : (state_dim, action_dim)
        Continuous-time control-input matrix g(x) = B.
        Must be a torch.Tensor on the same device as other inputs.
    alpha : float
        Extended class-K function coefficient.
    u_max : float or None
        If given, box-constrains the control: -u_max <= u <= u_max.

    Returns
    -------
    u_qp : (n_agents, action_dim)
        QP-optimal safe control for each agent.
        Always detached from the computation graph.
    """
    # ---- Lie derivative components (batched) ----
    # L_g h = (∂h/∂x) · B   →  (n_agents, action_dim)
    Lg_h = dh_dx @ B_mat

    # L_f h = (∂h/∂x) · f(x)  →  (n_agents,)
    Lf_h = (dh_dx * x_dot_f).sum(dim=-1)

    # ---- Constraint value at u_nom ----
    # c = L_g h · u_nom + L_f h + α·h   →  (n_agents,)
    c = (Lg_h * u_nom).sum(dim=-1) + Lf_h + alpha * h

    # ---- Analytical half-space projection (KKT) ----
    # ‖L_g h‖² per agent
    Lg_h_norm_sq = (Lg_h * Lg_h).sum(dim=-1)  # (n_agents,)

    # Dual variable: λ = max(0, -c / ‖L_g h‖²)
    # When c ≥ 0 → constraint satisfied at u_nom → λ = 0
    # When c < 0 → active constraint → λ = -c / ‖L_g h‖²
    lam = torch.relu(-c / (Lg_h_norm_sq + 1e-8))  # (n_agents,)

    # Primal: u* = u_nom + λ · L_g h
    u_qp = u_nom + lam.unsqueeze(-1) * Lg_h

    # ---- Input bounds (box constraint) ----
    if u_max is not None:
        u_qp = torch.clamp(u_qp, -u_max, u_max)

        # Check if clipping broke the CBF constraint; if so, do a
        # few projected gradient-ascent steps to find a feasible point
        # inside the box that best satisfies the CBF constraint.
        c_after = (Lg_h * u_qp).sum(dim=-1) + Lf_h + alpha * h
        violated = c_after < -1e-6  # (n_agents,) bool mask

        if violated.any():
            u_fix = u_qp.clone()
            for _ in range(10):  # lightweight inner loop
                # Gradient of constraint w.r.t. u is L_g h
                c_fix = (Lg_h * u_fix).sum(dim=-1) + Lf_h + alpha * h
                deficit = torch.relu(-c_fix)  # (n_agents,)

                # Only fix the violated agents
                step_size = deficit / (Lg_h_norm_sq + 1e-8)  # (n_agents,)
                u_fix = u_fix + step_size.unsqueeze(-1) * Lg_h

                # Re-apply box constraints
                u_fix = torch.clamp(u_fix, -u_max, u_max)

                # Check convergence
                c_new = (Lg_h * u_fix).sum(dim=-1) + Lf_h + alpha * h
                if (c_new >= -1e-6).all():
                    break

            # Apply fix only to originally violated agents
            u_qp = torch.where(
                violated.unsqueeze(-1).expand_as(u_qp),
                u_fix,
                u_qp,
            )

    return u_qp.detach()
