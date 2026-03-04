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
    A_extra: Optional[torch.Tensor] = None,
    b_extra: Optional[torch.Tensor] = None,
    slack_weight: float = 100.0,
) -> torch.Tensor:
    """
    Batched analytical CBF-QP solver with multi-constraint support.

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
    alpha : float
        Extended class-K function coefficient.
    u_max : float or None
        If given, box-constrains the control: -u_max <= u <= u_max.
    A_extra : (n_agents, K, action_dim) or None
        Additional inequality constraints: A_extra · u <= b_extra.
        These are treated as **soft constraints** with slack relaxation.
    b_extra : (n_agents, K) or None
        Right-hand side of extra constraints.
    slack_weight : float
        Penalty weight p for slack variables δ²: higher = harder constraint.

    Returns
    -------
    u_qp : (n_agents, action_dim)
        QP-optimal safe control for each agent.
    """
    # ---- Lie derivative components (batched) ----
    Lg_h = dh_dx @ B_mat                          # (n_agents, action_dim)
    Lf_h = (dh_dx * x_dot_f).sum(dim=-1)          # (n_agents,)

    # ---- Constraint value at u_nom ----
    c = (Lg_h * u_nom).sum(dim=-1) + Lf_h + alpha * h   # (n_agents,)

    # ---- Analytical half-space projection (KKT) — HARD constraint ----
    Lg_h_norm_sq = (Lg_h * Lg_h).sum(dim=-1)      # (n_agents,)
    lam = torch.relu(-c / (Lg_h_norm_sq + 1e-8))  # (n_agents,)
    u_qp = u_nom + lam.unsqueeze(-1) * Lg_h

    # ---- Extra constraints with slack relaxation (SOFT) ----
    if A_extra is not None and b_extra is not None:
        # A_extra: (N, K, action_dim), b_extra: (N, K)
        K = A_extra.shape[1]
        for k in range(K):
            a_k = A_extra[:, k, :]                # (N, action_dim)
            b_k = b_extra[:, k]                   # (N,)

            # Violation: v_k = a_k · u - b_k  (want <= 0)
            v_k = (a_k * u_qp).sum(dim=-1) - b_k  # (N,)

            # Only project if violated (v_k > 0)
            violated = v_k > 0                     # (N,) bool

            if violated.any():
                a_k_norm_sq = (a_k * a_k).sum(dim=-1)  # (N,)

                # Slack-relaxed projection:
                #   min ||u'- u||^2 + p * δ²
                #   s.t.  a_k · u' <= b_k + δ
                # Optimal δ = v_k / (1 + p * ||a_k||²)
                # Correction = (v_k - δ) = v_k * p*||a_k||² / (1 + p*||a_k||²)
                denom = 1.0 + slack_weight * a_k_norm_sq  # (N,)
                correction = v_k * slack_weight * a_k_norm_sq / (denom + 1e-8)

                # Project: u' = u - correction * a_k / ||a_k||²
                step = correction / (a_k_norm_sq + 1e-8)  # (N,)
                u_fix = u_qp - step.unsqueeze(-1) * a_k

                # Apply only to violated agents
                u_qp = torch.where(
                    violated.unsqueeze(-1).expand_as(u_qp),
                    u_fix,
                    u_qp,
                )

        # ---- Re-project onto CBF hard constraint after soft corrections ----
        c_re = (Lg_h * u_qp).sum(dim=-1) + Lf_h + alpha * h
        cbf_broken = c_re < -1e-6
        if cbf_broken.any():
            lam_re = torch.relu(-c_re / (Lg_h_norm_sq + 1e-8))
            u_re = u_qp + lam_re.unsqueeze(-1) * Lg_h
            u_qp = torch.where(
                cbf_broken.unsqueeze(-1).expand_as(u_qp),
                u_re,
                u_qp,
            )

    # ---- Input bounds (box constraint) ----
    if u_max is not None:
        u_qp = torch.clamp(u_qp, -u_max, u_max)

        # Check if clipping broke the CBF constraint; if so, do a
        # few projected gradient-ascent steps to find a feasible point.
        c_after = (Lg_h * u_qp).sum(dim=-1) + Lf_h + alpha * h
        violated = c_after < -1e-6

        if violated.any():
            u_fix = u_qp.clone()
            for _ in range(10):
                c_fix = (Lg_h * u_fix).sum(dim=-1) + Lf_h + alpha * h
                deficit = torch.relu(-c_fix)
                step_size = deficit / (Lg_h_norm_sq + 1e-8)
                u_fix = u_fix + step_size.unsqueeze(-1) * Lg_h
                u_fix = torch.clamp(u_fix, -u_max, u_max)
                c_new = (Lg_h * u_fix).sum(dim=-1) + Lf_h + alpha * h
                if (c_new >= -1e-6).all():
                    break
            u_qp = torch.where(
                violated.unsqueeze(-1).expand_as(u_qp),
                u_fix,
                u_qp,
            )

    return u_qp.detach()

