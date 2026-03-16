"""
Analytical Affine-Transform QP Solver — GPU-accelerated, pure PyTorch.

Solves per-swarm (batched):

    min_{X, δ}  ½‖X - u_nom‖² + p·Σδᵢ²

    where X = [a_cx, a_cy, a_s] (affine parameters)

subject to:
    1. HOCBF  payload swing (SOFT, a_cx/a_cy only):
       C_xi·a_cx ≥ D_xi - δ_xi
       C_yi·a_cy ≥ D_yi - δ_yi
    2. Obstacle CBF (HARD - 2nd Order HOCBF):
       A·X + (ḣ_drift + (α₁+α₂)ḣ + α₁α₂h) ≥ 0
    3. Scale  CBF (ABSOLUTE HARD):
       a_s ≥ -α₁·ṡ - α₂·(s - s_min)
       a_s ≤ -α₁·ṡ + α₂·(s_max - s)

# Constraint order: HOCBF → Scale → Obstacle / Agent-Agent (last applied = highest priority).
# Uses Dykstra-style alternating projection (3-5 iterations) for convergence.

After QP: box clamp on a_cx, a_cy (u_max) applied.
"""

from __future__ import annotations

from typing import Optional

import torch


def solve_affine_qp(
    u_nom: torch.Tensor,
    # Obstacle CBF data
    obs_centers: Optional[torch.Tensor] = None,
    obs_half_sizes: Optional[torch.Tensor] = None,
    agent_pos: Optional[torch.Tensor] = None,
    agent_vel: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,
    s_dot: Optional[torch.Tensor] = None,
    R_form: float = 0.5,
    r_margin: float = 0.2,
    mass: float = 0.1,
    # Agent-Agent CBF data
    other_agent_pos: Optional[torch.Tensor] = None,
    other_agent_vel: Optional[torch.Tensor] = None,
    other_agent_s: Optional[torch.Tensor] = None,
    other_agent_s_dot: Optional[torch.Tensor] = None,
    # Scale CBF params
    s_min: float = 0.4,
    s_max: float = 1.5,
    alpha_scale: float = 2.0,
    # Obstacle CBF params
    alpha_obs: float = 1.0, 
    alpha_obs_hoc1: float = 0.8,
    alpha_obs_hoc2: float = 0.8,
    # HOCBF payload swing data (dynamic γ_max)
    payload_states: Optional[torch.Tensor] = None,
    cable_length: float = 1.0,
    gravity: float = 9.81,
    gamma_min: float = 0.2,          # γ_max at s=s_min (strict)
    gamma_max_full: float = 0.75,    # γ_max at s=s_max (relaxed)
    payload_damping: float = 0.03,
    hocbf_alpha1: float = 2.0,
    hocbf_alpha2: float = 2.0,
    # Slack weight for soft constraints
    slack_weight: float = 100.0,
    # Input bounds
    u_max: Optional[float] = None,
    # Dykstra alternating projection iterations
    num_proj_iters: int = 3,
) -> torch.Tensor:
    """
    Analytical affine-transform QP solver with Dykstra alternating projection.

    Parameters
    ----------
    u_nom : (N, 3)
        Nominal affine accelerations [a_cx, a_cy, a_s] from Level 2 PD.
    num_proj_iters : int
        Number of Dykstra alternating projection iterations (default 3).

    Returns
    -------
    u_qp : (N, 3) — QP-safe affine parameters.
    """
    N = u_nom.shape[0]
    device = u_nom.device
    X = u_nom.clone()  # Start from nominal

    # Pre-compute HOCBF coefficients (constant across Dykstra iterations)
    has_hocbf = (payload_states is not None and s is not None)
    if has_hocbf:
        gx = payload_states[:, 0]
        gy = payload_states[:, 1]
        gx_dot = payload_states[:, 2]
        gy_dot = payload_states[:, 3]

        l = cable_length
        g_val = gravity
        c_damp = payload_damping
        a1 = hocbf_alpha1
        a2 = hocbf_alpha2

        # Dynamic γ_max(s)
        t_scale = ((s - s_min) / (s_max - s_min + 1e-8)).clamp(0.0, 1.0)
        gamma_dyn = gamma_min + (gamma_max_full - gamma_min) * t_scale

        # X-axis HOCBF coefficients
        h1x = gamma_dyn**2 - gx**2
        h1dx = -2 * gx * gx_dot
        h2x = h1dx + a1 * h1x
        Cx = 2 * gx * torch.cos(gx) / l
        Dx = (2 * gx_dot**2
              - 2 * gx * (-(g_val / l) * torch.sin(gx) - c_damp * gx_dot)
              + a1 * (-2 * gx * gx_dot)
              - a2 * h2x)

        # Y-axis HOCBF coefficients
        h1y = gamma_dyn**2 - gy**2
        h1dy = -2 * gy * gy_dot
        h2y = h1dy + a1 * h1y
        Cy = 2 * gy * torch.cos(gy) / l
        Dy = (2 * gy_dot**2
              - 2 * gy * (-(g_val / l) * torch.sin(gy) - c_damp * gy_dot)
              + a1 * (-2 * gy * gy_dot)
              - a2 * h2y)

    # Pre-compute obstacle CBF data (constant across iterations)
    has_obs = (obs_centers is not None and obs_half_sizes is not None
               and agent_pos is not None and agent_vel is not None
               and s is not None and s_dot is not None)
    
    # Pre-compute agent-agent CBF data
    has_other_agents = (other_agent_pos is not None and other_agent_vel is not None
                        and other_agent_s is not None and other_agent_s_dot is not None
                        and agent_pos is not None and agent_vel is not None
                        and s is not None and s_dot is not None)

    # Pre-compute scale CBF bounds
    has_scale = (s is not None and s_dot is not None)

    # ================================================================
    # Dykstra-style alternating projection
    # ================================================================
    for _iter in range(num_proj_iters):

        # ------------------------------------------------------------
        # 1. HOCBF Payload swing constraints (SOFT, a_cx/a_cy only)
        #    — lowest priority, applied first
        # ------------------------------------------------------------
        if has_hocbf:
            eps_c = 1e-6

            # X-axis: project a_cx
            ux = X[:, 0]
            violate_x = (Cx * ux < Dx) & (Cx.abs() > eps_c)
            if violate_x.any():
                target_x = Dx / (Cx + eps_c * torch.sign(Cx))
                violation_x = target_x - ux
                correction_x = violation_x * slack_weight / (1.0 + slack_weight)
                ux_new = ux + correction_x
                X[:, 0] = torch.where(violate_x, ux_new, ux)

            # Y-axis: project a_cy
            uy = X[:, 1]
            violate_y = (Cy * uy < Dy) & (Cy.abs() > eps_c)
            if violate_y.any():
                target_y = Dy / (Cy + eps_c * torch.sign(Cy))
                violation_y = target_y - uy
                correction_y = violation_y * slack_weight / (1.0 + slack_weight)
                uy_new = uy + correction_y
                X[:, 1] = torch.where(violate_y, uy_new, uy)

        # ------------------------------------------------------------
        # 2. Scale CBF constraints
        #    — medium priority, applied before collisions
        # ------------------------------------------------------------
        if has_scale:
            # s and s_dot are verified by has_scale
            lower_bound = -alpha_scale * s_dot - alpha_scale * (s - s_min)
            X[:, 2] = torch.max(X[:, 2], lower_bound)

            upper_bound = -alpha_scale * s_dot + alpha_scale * (s_max - s)
            X[:, 2] = torch.min(X[:, 2], upper_bound)

        # ------------------------------------------------------------
        # 3. Obstacle CBF constraints (HARD - 2nd Order HOCBF)
        #    — high priority, applied near-last
        # ------------------------------------------------------------
        if has_obs:
            # Re-check for lint/type safety
            if obs_centers is not None and obs_half_sizes is not None:
                n_obs = obs_centers.shape[1]
                if n_obs > 0:
                    a1, a2 = alpha_obs_hoc1, alpha_obs_hoc2
                    # Broadened dims for vectorization: (N, n_obs, dim)
                    dp = agent_pos.unsqueeze(1) - obs_centers
                    dist_sq = (dp * dp).sum(dim=-1)

                    R_obs = torch.max(obs_half_sizes, dim=-1).values
                    r_sw = R_form * s + r_margin
                    safe_dist = r_sw.unsqueeze(1) + R_obs
                    h_obs = dist_sq - safe_dist ** 2

                    # 1st derivative: h_dot
                    r_dot = R_form * s_dot
                    h_dot = 2.0 * (dp * agent_vel.unsqueeze(1)).sum(dim=-1) - 2.0 * safe_dist * r_dot.unsqueeze(1)

                    # 2nd derivative drift term
                    h_ddot_drift = 2.0 * agent_vel.pow(2).sum(dim=-1).unsqueeze(1) - 2.0 * r_dot.pow(2).unsqueeze(1)

                    A_cx = 2.0 * dp[..., 0] / mass
                    A_cy = 2.0 * dp[..., 1] / mass
                    A_as = -2.0 * safe_dist * R_form
                    A_vec = torch.stack([A_cx, A_cy, A_as], dim=-1) # (N, n_obs, 3)

                    # HOCBF form: h_ddot + (a1 + a2)*h_dot + (a1 * a2)*h_obs >= 0
                    rhs = h_ddot_drift + (a1 + a2) * h_dot + (a1 * a2) * h_obs

                    # Calculate constraint values for all obstacles simultaneously
                    c_vals = (A_vec * X.unsqueeze(1)).sum(dim=-1) + rhs # (N, n_obs)
                    
                    # Pick the most violated constraint for each agent
                    worst_c, worst_idx = torch.min(c_vals, dim=1)
                    violated = worst_c < 0

                    if violated.any():
                        row_idx = torch.arange(N, device=device)[violated]
                        col_idx = worst_idx[violated]
                        A_worst = A_vec[row_idx, col_idx]
                        c_worst = worst_c[violated]

                        A_norm_sq = (A_worst * A_worst).sum(dim=-1)
                        lam = torch.relu(-c_worst / (A_norm_sq + 1e-8))
                        X[violated] += lam.unsqueeze(-1) * A_worst

        # ------------------------------------------------------------
        # 4. Agent-Agent CBF constraints (HARD)
        #    — highest priority, applied last
        # ------------------------------------------------------------
        if has_other_agents:
            if other_agent_pos is not None and other_agent_vel is not None:
                n_other = other_agent_pos.shape[1]
                if n_other > 0:
                    a1, a2 = alpha_obs_hoc1, alpha_obs_hoc2
                    # Vectorized dims: (N, n_other, dim)
                    dp = agent_pos.unsqueeze(1) - other_agent_pos
                    dv = agent_vel.unsqueeze(1) - other_agent_vel
                    dist_sq = (dp * dp).sum(dim=-1)

                    r_sw = R_form * s + r_margin
                    r_other = R_form * other_agent_s + r_margin
                    safe_dist = r_sw.unsqueeze(1) + r_other
                    h_agent = dist_sq - safe_dist ** 2

                    r_dot_total = R_form * (s_dot.unsqueeze(1) + other_agent_s_dot)
                    h_dot = 2.0 * (dp * dv).sum(dim=-1) - 2.0 * safe_dist * r_dot_total
                    h_ddot_drift = 2.0 * dv.pow(2).sum(dim=-1) - 2.0 * r_dot_total.pow(2)

                    A_cx = 2.0 * dp[..., 0] / mass
                    A_cy = 2.0 * dp[..., 1] / mass
                    A_as = -2.0 * safe_dist * R_form
                    A_vec = torch.stack([A_cx, A_cy, A_as], dim=-1)

                    # HOCBF form with Reciprocal Collision Avoidance
                    rhs = (h_ddot_drift + (a1 + a2) * h_dot + (a1 * a2) * h_agent) * 0.5
                    c_vals = (A_vec * X.unsqueeze(1)).sum(dim=-1) + rhs

                    worst_c, worst_idx = torch.min(c_vals, dim=1)
                    violated = worst_c < 0

                    if violated.any():
                        row_idx = torch.arange(N, device=device)[violated]
                        col_idx = worst_idx[violated]
                        A_worst = A_vec[row_idx, col_idx]
                        c_worst = worst_c[violated]

                        A_norm_sq = (A_worst * A_worst).sum(dim=-1)
                        lam = torch.relu(-c_worst / (A_norm_sq + 1e-8))
                        X[violated] += lam.unsqueeze(-1) * A_worst

    # ================================================================
    # 5. Box clamp on translation (u_max)
    # ================================================================
    if u_max is not None:
        X[:, :2] = torch.clamp(X[:, :2], -u_max, u_max)

    return X.detach()
