"""
Nominal controller for affine-transform swarm.

Control pipeline (per call):
  1. pos_error  = goal_pos - agent_pos
  2. Clip ||pos_error|| to comm_radius
  3. u_trans = Kp * pos_error_clipped + Kd * vel_error
  4. Clip u_trans to [-u_max, u_max]

Scale channel:
  s_dot_ref = K_s_pos * (1.0 - s)
  a_s       = K_s * (s_dot_ref - s_dot)
  Clip a_s  to [-u_max_scale, u_max_scale]

GNN offset (pi_scaled) is added by the caller AFTER this controller returns.
"""

from __future__ import annotations

import torch


class NominalController:
    """Nominal PD controller for the CoM + scale affine system.

    Parameters
    ----------
    comm_radius : communication / clipping radius [m]
    u_max       : max translational acceleration [m/s²]
    u_max_scale : max scale acceleration [s⁻²]
    Kp          : position P-gain
    Kd          : velocity D-gain
    K_s_pos     : P-gain for scale toward s=1.0
    K_s         : D-gain for scale rate tracking
    """

    def __init__(
        self,
        comm_radius: float,
        u_max: float,
        u_max_scale: float,
        Kp: float = 5.0,
        Kd: float = 2.0,
        K_s_pos: float = 1.0,
        K_s: float = 2.0,
    ):
        self.comm_radius = comm_radius
        self.u_max = u_max
        self.u_max_scale = u_max_scale
        self.Kp = Kp
        self.Kd = Kd
        self.K_s_pos = K_s_pos
        self.K_s = K_s

    def __call__(
        self,
        agent_states: torch.Tensor,   # (..., 4)  [px, py, vx, vy]
        goal_states: torch.Tensor,    # (..., 4)  [gx, gy, ...]
        scale_states: torch.Tensor,   # (..., 2)  [s, s_dot]
    ) -> torch.Tensor:
        """Compute nominal acceleration u_nom (..., 3) = [a_cx, a_cy, a_s]."""

        # ── Translation ──────────────────────────────────────────────────
        pos      = agent_states[..., :2]
        vel      = agent_states[..., 2:4]
        goal_pos = goal_states[..., :2]
        goal_vel = goal_states[..., 2:4]

        pos_error = goal_pos - pos                                        # (..., 2)

        dist = pos_error.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pos_error_clipped = pos_error * (dist.clamp(max=self.comm_radius) / dist)

        vel_error = goal_vel - vel                                        # (..., 2)
        u_trans = self.Kp * pos_error_clipped + self.Kd * vel_error
        u_trans = u_trans.clamp(-self.u_max, self.u_max)

        # ── Scale ────────────────────────────────────────────────────────
        s         = scale_states[..., 0]
        s_dot     = scale_states[..., 1]
        s_dot_ref = self.K_s_pos * (1.0 - s)
        a_s       = self.K_s * (s_dot_ref - s_dot)
        a_s       = a_s.clamp(-self.u_max_scale, self.u_max_scale)

        return torch.cat([u_trans, a_s.unsqueeze(-1)], dim=-1)  # (..., 3)
