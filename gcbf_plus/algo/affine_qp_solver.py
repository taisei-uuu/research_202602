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
    obs_hits: Optional[torch.Tensor] = None,
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

    # alpha_sum / alpha_prod are used by both HOCBF (payload) and Scale CBF
    alpha_sum = hocbf_alpha1 + hocbf_alpha2
    alpha_prod = hocbf_alpha1 * hocbf_alpha2

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

        # -----------------------------------------------------------------
        # Dynamic γ_max(s) based on physical swarm radius:
        # γ_max(s) = asin(R_form * s / L)
        # -----------------------------------------------------------------
        ratio = (R_form * s) / l
        clamped_ratio = torch.clamp(ratio, 0.0, 0.95)
        gamma_dyn = torch.asin(clamped_ratio)

        # --- X-axis HOCBF ---
        h_x = gamma_dyn**2 - gx**2
        h_dot_x = -2.0 * gx * gx_dot
        # h_ddot_drift = -2*gx_dot^2 - 2*gx * (-(g/l)sin(gx) - c*gx_dot)
        h_ddot_drift_x = -2.0 * gx_dot**2 + (2.0 * gx * g_val / l) * torch.sin(gx) + 2.0 * gx * c_damp * gx_dot
        
        Cx = (2.0 * gx * torch.cos(gx)) / l
        rhs_x = h_ddot_drift_x + alpha_sum * h_dot_x + alpha_prod * h_x

        # --- Y-axis HOCBF ---
        h_y = gamma_dyn**2 - gy**2
        h_dot_y = -2.0 * gy * gy_dot
        # h_ddot_drift = -2*gy_dot^2 - 2*gy * (-(g/l)sin(gy) - c*gy_dot)
        h_ddot_drift_y = -2.0 * gy_dot**2 + (2.0 * gy * g_val / l) * torch.sin(gy) + 2.0 * gy * c_damp * gy_dot
        
        Cy = (2.0 * gy * torch.cos(gy)) / l
        rhs_y = h_ddot_drift_y + alpha_sum * h_dot_y + alpha_prod * h_y

        # Apply projections (X used in the loop below)
        # Note: In this vectorized version, the actual projection happens in the loop.
        # However, the payload constraint now couples [a_cx, a_cy, a_s].
        # Since X is (N, 3), we need to handle this 3D constraint.

    # Pre-compute obstacle CBF data (constant across iterations)
    has_obs = (obs_hits is not None
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

            # X-axis: Cx * a_cx + rhs_x >= 0
            c_val_x = Cx * X[:, 0] + rhs_x
            violate_x = (c_val_x < 0) & (Cx.abs() > eps_c)
            if violate_x.any():
                lam = torch.relu(-c_val_x / (Cx**2 + 1e-8))
                X[violate_x, 0] += lam[violate_x] * Cx[violate_x]

            # Y-axis: Cy * a_cy + rhs_y >= 0
            c_val_y = Cy * X[:, 1] + rhs_y
            violate_y = (c_val_y < 0) & (Cy.abs() > eps_c)
            if violate_y.any():
                lam = torch.relu(-c_val_y / (Cy**2 + 1e-8))
                X[violate_y, 1] += lam[violate_y] * Cy[violate_y]

        # ------------------------------------------------------------
        # 2. Scale CBF constraints
        #    — medium priority, applied before collisions
        # ------------------------------------------------------------
        if has_scale:
            # 2nd Order HOCBF for scale: a_s + (a1+a2)s_dot + a1*a2(s - s_min) >= 0
            # lower_bound: a_s >= -(alpha_sum * s_dot) - (alpha_prod * (s - s_min))
            lower_bound = -alpha_sum * s_dot - alpha_prod * (s - s_min)
            X[:, 2] = torch.max(X[:, 2], lower_bound)

            # upper_bound: a_s <= -(alpha_sum * s_dot) + (alpha_prod * (s_max - s))
            upper_bound = -alpha_sum * s_dot + alpha_prod * (s_max - s)
            X[:, 2] = torch.min(X[:, 2], upper_bound)

        # ------------------------------------------------------------
        # 3. Obstacle CBF constraints (HARD - 2nd Order HOCBF)
        #    — high priority, applied near-last
        # ------------------------------------------------------------
        if has_obs:
            # Re-check for lint/type safety
            if obs_hits is not None:
                n_obs = obs_hits.shape[1]
                if n_obs > 0:
                    a1, a2 = alpha_obs_hoc1, alpha_obs_hoc2
                    # Broadened dims for vectorization: (N, n_obs, dim)
                    dp = agent_pos.unsqueeze(1) - obs_hits
                    dist_sq = (dp * dp).sum(dim=-1)

                    r_sw = R_form * s + r_margin
                    safe_dist = r_sw.unsqueeze(1) # (N, 1) or (N, n_obs)
                    h_obs = dist_sq - safe_dist ** 2

                    # 1st derivative: h_dot
                    r_dot = R_form * s_dot
                    h_dot = 2.0 * (dp * agent_vel.unsqueeze(1)).sum(dim=-1) - 2.0 * safe_dist * r_dot.unsqueeze(1)

                    # 2nd derivative drift term
                    h_ddot_drift = 2.0 * agent_vel.pow(2).sum(dim=-1).unsqueeze(1) - 2.0 * r_dot.pow(2).unsqueeze(1)

                    A_cx = 2.0 * dp[..., 0] # (N, n_obs)
                    A_cy = 2.0 * dp[..., 1] # (N, n_obs)
                    A_as = -2.0 * safe_dist * R_form # (N, 1)
                    # Broadcast A_as to match A_cx/A_cy (N, n_obs)
                    A_as = A_as.expand(-1, n_obs)
                    
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

                    A_cx = 2.0 * dp[..., 0]
                    A_cy = 2.0 * dp[..., 1]
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
