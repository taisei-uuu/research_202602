"""
Analytical Affine-Transform QP Solver — GPU-accelerated, pure PyTorch.

Solves per-swarm (batched):

    min_{X, δ}  ½‖X - u_AT‖² + p·Σδᵢ²

    where X = [a_cx, a_cy, a_s] (affine parameters)

subject to:
    1. Obstacle CBF (HARD):
       ḣ_obs + α·h_obs ≥ 0
       where h_obs = ‖P_c - P_obs‖² - (r_swarm(s) + R_obs)²

    2. Scale-CBF lower (HARD):
       a_s + α₁·ṡ + α₂·(s - s_min) ≥ 0

    3. Scale-CBF upper (HARD):
       -a_s - α₁·ṡ + α₂·(s_max - s) ≥ 0

    4. HOCBF payload swing (SOFT):
       C_xi·(a_cx + a_s·p_ix) ≥ D_xi - δ_xi
       C_yi·(a_cy + a_s·p_iy) ≥ D_yi - δ_yi

Uses iterative half-space projection for multiple linear inequality constraints.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch


def solve_affine_qp(
    u_at: torch.Tensor,
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
    # Scale CBF params
    s_min: float = 0.4,
    s_max: float = 1.5,
    alpha_scale: float = 2.0,
    # Obstacle CBF params
    alpha_obs: float = 1.0,
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
) -> torch.Tensor:
    """
    Analytical affine-transform QP solver.

    Parameters
    ----------
    u_at : (N, 3)
        Nominal affine parameters [a_cx, a_cy, a_s] per swarm.
    obs_centers : (N, n_obs, 2) or None
        Obstacle center positions.
    obs_half_sizes : (N, n_obs, 2) or None
        Obstacle half sizes.
    agent_pos : (N, 2)
        Current CoM positions.
    agent_vel : (N, 2)
        Current CoM velocities.
    s : (N,)
        Current scale factors.
    s_dot : (N,)
        Current scale rates.
    R_form : float
        Base formation radius.
    r_margin : float
        Bounding circle margin.
    mass : float
        Agent mass.
    payload_states : (N, 4) or None
        [γ_x, γ_y, γ̇_x, γ̇_y] per agent.

    Returns
    -------
    u_qp : (N, 3) — QP-safe affine parameters.
    """
    N = u_at.shape[0]
    device = u_at.device
    X = u_at.clone()  # Start from nominal

    # ================================================================
    # 1. Scale CBF constraints (HARD) — simple 1D projections
    # ================================================================
    if s is not None and s_dot is not None:
        # Lower bound: h_s1 = s - s_min ≥ 0
        #   HOCBF: a_s + α₁·ṡ + α₂·(s - s_min) ≥ 0
        #   → a_s ≥ -α₁·ṡ - α₂·(s - s_min)
        lower_bound = -alpha_scale * s_dot - alpha_scale * (s - s_min)  # (N,)
        X[:, 2] = torch.max(X[:, 2], lower_bound)

        # Upper bound: h_s2 = s_max - s ≥ 0
        #   HOCBF: -a_s - α₁·ṡ + α₂·(s_max - s) ≥ 0
        #   → a_s ≤ -α₁·ṡ + α₂·(s_max - s)
        upper_bound = -alpha_scale * s_dot + alpha_scale * (s_max - s)  # (N,)
        X[:, 2] = torch.min(X[:, 2], upper_bound)

    # ================================================================
    # 2. Obstacle CBF constraints (HARD) — per obstacle
    # ================================================================
    if (obs_centers is not None and obs_half_sizes is not None
            and agent_pos is not None and agent_vel is not None
            and s is not None and s_dot is not None):
        n_obs = obs_centers.shape[1]

        for j in range(n_obs):
            oc = obs_centers[:, j, :]    # (N, 2)
            ohs = obs_half_sizes[:, j, :]  # (N, 2)

            # Approximate obstacle as circle with radius R_obs = max(half_size)
            R_obs = torch.max(ohs, dim=-1).values  # (N,)

            # r_swarm(s) = R_form * s + r_margin
            r_sw = R_form * s + r_margin  # (N,)

            # h_obs = ‖P_c - P_obs‖² - (r_swarm(s) + R_obs)²
            dp = agent_pos - oc  # (N, 2)
            dist_sq = (dp * dp).sum(dim=-1)  # (N,)
            safe_dist = r_sw + R_obs  # (N,)
            h_obs = dist_sq - safe_dist ** 2  # (N,)

            # ḣ_obs = 2·(P_c - P_obs)·V_c - 2·(r_sw + R_obs)·R_form·ṡ
            # (since d/dt r_swarm = R_form · ṡ)
            h_dot_drift = 2.0 * (dp * agent_vel).sum(dim=-1) \
                          - 2.0 * safe_dist * R_form * s_dot  # (N,)

            # Control-dependent part of ḣ_obs:
            # ∂ḣ_obs/∂a_cx = 2·dp_x / mass,  ∂ḣ_obs/∂a_cy = 2·dp_y / mass
            # ∂ḣ_obs/∂a_s = -2·(r_sw + R_obs)·R_form  (through ṡ evolution after dt)
            # But since a_cx, a_cy are forces and ḣ depends on velocity:
            # In continuous time: ḣ = 2·dp·v - 2·safe_dist·R_form·ṡ
            # The CBF condition is: ḣ + α·h ≥ 0
            # We can only affect ḣ through acceleration (second derivative),
            # so we use a relative-degree-1 approximation on the discrete system.

            # Linearized CBF constraint:
            # 2·dp·(v + a_trans/m·dt) - 2·(r_sw + R_obs)·R_form·(ṡ + a_s·dt) + α·h ≥ 0
            # → (2·dp_x·dt/m)·a_cx + (2·dp_y·dt/m)·a_cy + (-2·safe_dist·R_form·dt)·a_s
            #   ≥ -(2·dp·v - 2·safe_dist·R_form·ṡ + α·h)

            dt_proxy = 1.0  # Use unit scaling for the inequality
            A_cx = 2.0 * dp[:, 0] / mass  # (N,)
            A_cy = 2.0 * dp[:, 1] / mass  # (N,)
            A_as = -2.0 * safe_dist * R_form  # (N,)

            # Constraint: A_cx·a_cx + A_cy·a_cy + A_as·a_s + (ḣ_drift + α·h) ≥ 0
            rhs = h_dot_drift + alpha_obs * h_obs  # (N,)

            # Build constraint vector A = [A_cx, A_cy, A_as]
            A_vec = torch.stack([A_cx, A_cy, A_as], dim=-1)  # (N, 3)

            # Constraint value at current X
            c_val = (A_vec * X).sum(dim=-1) + rhs  # (N,)

            # Project if violated (c_val < 0)
            violated = c_val < 0
            if violated.any():
                A_norm_sq = (A_vec * A_vec).sum(dim=-1)  # (N,)
                lam = torch.relu(-c_val / (A_norm_sq + 1e-8))  # (N,)
                correction = lam.unsqueeze(-1) * A_vec  # (N, 3)
                X = torch.where(
                    violated.unsqueeze(-1).expand_as(X),
                    X + correction,
                    X,
                )

    # ================================================================
    # 3. HOCBF Payload swing constraints (SOFT)
    # ================================================================
    if payload_states is not None and s is not None:
        gx = payload_states[:, 0]
        gy = payload_states[:, 1]
        gx_dot = payload_states[:, 2]
        gy_dot = payload_states[:, 3]

        l = cable_length
        g_val = gravity
        c_damp = payload_damping
        a1 = hocbf_alpha1
        a2 = hocbf_alpha2

        # ── Dynamic γ_max(s): linear interpolation ──
        # s=s_min → γ_min (strict), s=s_max → γ_max_full (relaxed)
        t_scale = ((s - s_min) / (s_max - s_min + 1e-8)).clamp(0.0, 1.0)
        gamma_dyn = gamma_min + (gamma_max_full - gamma_min) * t_scale  # (N,)

        # X-axis HOCBF (using dynamic γ_max per agent)
        h1x = gamma_dyn**2 - gx**2
        h1dx = -2 * gx * gx_dot
        h2x = h1dx + a1 * h1x
        Cx = 2 * gx * torch.cos(gx) / l
        Dx = (2 * gx_dot**2
              - 2 * gx * (-(g_val / l) * torch.sin(gx) - c_damp * gx_dot)
              + a1 * (-2 * gx * gx_dot)
              - a2 * h2x)

        # Y-axis HOCBF (using dynamic γ_max per agent)
        h1y = gamma_dyn**2 - gy**2
        h1dy = -2 * gy * gy_dot
        h2y = h1dy + a1 * h1y
        Cy = 2 * gy * torch.cos(gy) / l
        Dy = (2 * gy_dot**2
              - 2 * gy * (-(g_val / l) * torch.sin(gy) - c_damp * gy_dot)
              + a1 * (-2 * gy * gy_dot)
              - a2 * h2y)

        # Build soft constraint: Cx * u_x ≥ Dx  →  -Cx * u_x ≤ -Dx
        # Here u_x = a_cx (the affine centroid acceleration)
        # For now, we apply the HOCBF on the centroid acceleration directly.
        # (The per-drone distribution adds a_s * p_i terms, but for the centroid
        #  the average effect is just a_cx, a_cy since Σp_i = 0)
        eps_c = 1e-6

        # X-axis: project a_cx if Cx * a_cx < Dx
        ux = X[:, 0]
        violate_x = (Cx * ux < Dx) & (Cx.abs() > eps_c)
        if violate_x.any():
            target_x = Dx / (Cx + eps_c * torch.sign(Cx))
            # Slack relaxation
            violation_x = target_x - ux
            correction_x = violation_x * slack_weight / (1.0 + slack_weight)
            ux_new = ux + correction_x
            X[:, 0] = torch.where(violate_x, ux_new, ux)

        # Y-axis: project a_cy if Cy * a_cy < Dy
        uy = X[:, 1]
        violate_y = (Cy * uy < Dy) & (Cy.abs() > eps_c)
        if violate_y.any():
            target_y = Dy / (Cy + eps_c * torch.sign(Cy))
            violation_y = target_y - uy
            correction_y = violation_y * slack_weight / (1.0 + slack_weight)
            uy_new = uy + correction_y
            X[:, 1] = torch.where(violate_y, uy_new, uy)

    # ================================================================
    # 4. Input bounds (box constraint)
    # ================================================================
    if u_max is not None:
        X[:, :2] = torch.clamp(X[:, :2], -u_max, u_max)

    return X.detach()
