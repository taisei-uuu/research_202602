"""
Nominal controller for affine-transform swarm.

Control pipeline (per call):
  1. pos_error  = goal_pos - agent_pos
  2. Clip ||pos_error|| to comm_radius   (goal perceived at most comm_radius away)
  3. u_trans = K @ [pos_error_clipped, vel_error]   (LQR gain, computed at init)
  4. Clip u_trans to [-u_max, u_max]

Scale channel:
  s_dot_ref = K_s_pos * (s_max - s)     (expand toward s_max)
  a_s       = K_s * (s_dot_ref - s_dot)
  Clip a_s  to [-u_max_scale, u_max_scale]

GNN offset (pi_scaled) is added by the caller AFTER this controller returns.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


def _dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Discrete-time LQR via iterative DARE.  Returns gain K (m x n)."""
    P = Q.copy()
    for _ in range(1000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_new - P)) < 1e-10:
            break
        P = P_new
    return K


class NominalController:
    """Nominal LQR controller for the CoM + scale affine system.

    Parameters
    ----------
    dt          : simulation timestep [s]
    mass        : swarm effective mass [kg]
    comm_radius : communication / clipping radius [m]
    u_max       : max translational acceleration [m/s²]
    u_max_scale : max scale acceleration [s⁻²]
    K_s_pos     : P-gain for scale expansion toward s_max
    K_s         : PD-gain for scale rate tracking
    s_max       : target (max) scale value
    Q, R        : LQR weight matrices (default: 5*I, I)
    """

    def __init__(
        self,
        dt: float,
        mass: float,
        comm_radius: float,
        u_max: float,
        u_max_scale: float,
        K_s_pos: float = 1.0,
        K_s: float = 2.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        self.comm_radius = comm_radius
        self.u_max = u_max
        self.u_max_scale = u_max_scale
        self.K_s_pos = K_s_pos
        self.K_s = K_s

        # Discrete-time double-integrator for 4D translational state
        A = np.eye(4, dtype=np.float32)
        A[0, 2] = dt
        A[1, 3] = dt
        B = np.zeros((4, 2), dtype=np.float32)
        B[2, 0] = dt / mass
        B[3, 1] = dt / mass

        if Q is None:
            Q = np.eye(4, dtype=np.float32) * 5.0
        if R is None:
            R = np.eye(2, dtype=np.float32) * 10.0

        K_np = _dlqr(A, B, Q, R)  # (2, 4)
        self._K = torch.tensor(K_np, dtype=torch.float32)  # registered on CPU; moved in __call__

    def __call__(
        self,
        agent_states: torch.Tensor,   # (..., 4)  [px, py, vx, vy]
        goal_states: torch.Tensor,    # (..., 4)  [gx, gy, ...]
        scale_states: torch.Tensor,   # (..., 2)  [s, s_dot]
    ) -> torch.Tensor:
        """Compute nominal acceleration u_nom (..., 3) = [a_cx, a_cy, a_s]."""
        device = agent_states.device
        K = self._K.to(device)

        # ── Translation ──────────────────────────────────────────────────
        pos       = agent_states[..., :2]
        vel       = agent_states[..., 2:4]
        goal_pos  = goal_states[..., :2]
        goal_vel  = goal_states[..., 2:4]

        # 1. Position error
        pos_error = goal_pos - pos                                       # (..., 2)

        # 2. Clip error magnitude to comm_radius
        dist = pos_error.norm(dim=-1, keepdim=True).clamp(min=1e-6)      # (..., 1)
        pos_error_clipped = pos_error * (dist.clamp(max=self.comm_radius) / dist)

        # 3. Full state error with clipped position part
        vel_error = goal_vel - vel                                        # (..., 2)
        state_error = torch.cat([pos_error_clipped, vel_error], dim=-1)  # (..., 4)

        # LQR: K (2,4) @ state_error (...,4,1) → (...,2)
        u_trans = (K @ state_error.unsqueeze(-1)).squeeze(-1)

        # 4. Clip to actuator limit
        u_trans = u_trans.clamp(-self.u_max, self.u_max)

        # ── Scale ────────────────────────────────────────────────────────
        # Nominal targets s=1.0 (neutral formation).
        # GNN offset (added by caller) shrinks the formation for obstacle avoidance.
        s         = scale_states[..., 0]
        s_dot     = scale_states[..., 1]
        s_dot_ref = self.K_s_pos * (1.0 - s)
        a_s       = self.K_s * (s_dot_ref - s_dot)
        a_s       = a_s.clamp(-self.u_max_scale, self.u_max_scale)

        return torch.cat([u_trans, a_s.unsqueeze(-1)], dim=-1)  # (..., 3)
