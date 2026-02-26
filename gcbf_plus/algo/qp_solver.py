"""
CBF-Quadratic Program (QP) solver — Equation 17 from the GCBF+ paper.

    min_u  ‖u - u_nom‖²
    s.t.   ḣ(x, u) + α·h(x) ≥ 0     (CBF constraint)

Uses `cvxpy` to solve a per-agent QP.  This is **not differentiable** and
is used only to generate target labels for the policy network.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch

from typing import Optional


def solve_cbf_qp(
    u_nom: torch.Tensor,
    h: torch.Tensor,
    dh_dx: torch.Tensor,
    x_dot_f: torch.Tensor,
    B_mat: np.ndarray,
    alpha: float = 1.0,
    u_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Solve the CBF-QP for each agent independently.

    Parameters
    ----------
    u_nom : (n_agents, action_dim)
        Nominal (LQR) control input.
    h : (n_agents,)
        CBF value for each agent.
    dh_dx : (n_agents, state_dim)
        Gradient ∂h/∂x for each agent (computed via autograd).
    x_dot_f : (n_agents, state_dim)
        Drift dynamics f(x) = [vx, vy, 0, 0] (without control input).
    B_mat : (state_dim, action_dim)  numpy
        Control-input matrix  g(x) = B  (constant for double integrator).
    alpha : float
        Extended class-K function coefficient: α(h) = α·h.
    u_max : float or None
        If given, box-constrains the control: -u_max <= u <= u_max.

    Returns
    -------
    u_qp : (n_agents, action_dim)
        QP-optimal safe control for each agent.

    Notes
    -----
    The CBF constraint is:

        L_f h + L_g h · u + α·h ≥ 0

    where  L_f h = (∂h/∂x) · f(x),  L_g h = (∂h/∂x) · B.
    """
    n_agents, action_dim = u_nom.shape
    u_nom_np = u_nom.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()
    dh_dx_np = dh_dx.detach().cpu().numpy()
    x_dot_f_np = x_dot_f.detach().cpu().numpy()
    B_np = B_mat.astype(np.float64)

    u_qp_list = []

    for i in range(n_agents):
        u_var = cp.Variable(action_dim)

        # Objective: min ‖u - u_nom‖²
        objective = cp.Minimize(cp.sum_squares(u_var - u_nom_np[i]))

        # CBF constraint:  dh/dx · f(x) + dh/dx · B · u + α·h ≥ 0
        Lf_h = float(dh_dx_np[i] @ x_dot_f_np[i])
        Lg_h = dh_dx_np[i] @ B_np  # (action_dim,)

        constraints = [
            Lg_h @ u_var + Lf_h + alpha * float(h_np[i]) >= 0
        ]

        if u_max is not None:
            constraints += [u_var >= -u_max, u_var <= u_max]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                u_qp_list.append(u_var.value)
            else:
                # Fallback to nominal if infeasible
                u_qp_list.append(u_nom_np[i])
        except cp.SolverError:
            u_qp_list.append(u_nom_np[i])

    u_qp = np.stack(u_qp_list, axis=0)
    return torch.tensor(u_qp, dtype=u_nom.dtype, device=u_nom.device)
