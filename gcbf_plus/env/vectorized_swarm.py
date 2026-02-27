"""
Vectorized (batched) swarm environment.

All B environments run in parallel on GPU using (B, n_agents, 6) state
tensors.  This eliminates the Python for-loop over batch_size during
data collection, yielding ~100× speedup on GPU.

Key methods:
    reset()              → reset all B environments, returns mega-graph
    step(action)         → batched Euler integration
    unsafe_mask()        → (B, n) via batched 9-point drone distances
    nominal_controller() → (B, n, 3) batched LQR+PD
    build_batch_graph()  → ONE mega-graph from all B environments
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils.graph import GraphsTuple
from ..utils.swarm_graph import (
    _rotation_matrix,
    _wrap_angle,
    get_equilateral_offsets,
    build_vectorized_swarm_graph,
)


# ---------------------------------------------------------------------------
# LQR helper  (discrete-time, infinite-horizon, for the translational part)
# ---------------------------------------------------------------------------

def _dlqr(A, B, Q, R):
    """Solve discrete-time LQR via iterative DARE.  Returns gain K."""
    P = Q.copy()
    for _ in range(1000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_new - P)) < 1e-10:
            break
        P = P_new
    return K


class VectorizedSwarmEnv:
    """
    Vectorized (batched) swarm environment.

    Parameters
    ----------
    num_agents : int — number of swarms
    batch_size : int — number of parallel environments
    area_size  : float — arena side length
    dt         : float — simulation timestep
    max_steps  : int
    params     : dict — override defaults
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "drone_radius": 0.05,
        "R_form": 0.3,
        "comm_radius": 2.0,
        "n_obs": 2,
        "mass": 0.1,
        "inertia": 0.01,
        "u_max": 0.3,
        "alpha_max": 0.1,
        "v_max": 1.0,
        "omega_max": 2.0,
    }

    def __init__(
        self,
        num_agents: int = 3,
        batch_size: int = 256,
        area_size: float = 10.0,
        dt: float = 0.03,
        max_steps: int = 256,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.area_size = area_size
        self.dt = dt
        self.max_steps = max_steps

        self.params = {**self.DEFAULT_PARAMS}
        if params is not None:
            self.params.update(params)

        # LQR gain for translational subsystem (4D: px,py,vx,vy)
        m = self.params["mass"]
        A_ct = np.zeros((4, 4), dtype=np.float32)
        A_ct[0, 2] = 1.0
        A_ct[1, 3] = 1.0
        A_t = A_ct * dt + np.eye(4, dtype=np.float32)
        B_t = np.array(
            [[0, 0], [0, 0], [1/m, 0], [0, 1/m]], dtype=np.float32
        ) * dt
        Q_t = np.eye(4, dtype=np.float32) * 5.0
        R_t = np.eye(2, dtype=np.float32)
        self._K_trans = torch.tensor(_dlqr(A_t, B_t, Q_t, R_t), dtype=torch.float32)

        # PD gains for rotational subsystem
        self._Kp_theta = 2.0
        self._Kd_theta = 1.0

        # Pre-compute local offsets
        self._local_offsets = get_equilateral_offsets(
            self.params["R_form"], device=torch.device("cpu")
        )  # (3, 2)

        # Internal batched state tensors (set on reset)
        self._agent_states: Optional[torch.Tensor] = None    # (B, n, 6)
        self._goal_states: Optional[torch.Tensor] = None     # (B, n, 6)
        self._obstacle_centers: Optional[torch.Tensor] = None   # (B, n_obs, 2)
        self._obstacle_half_sizes: Optional[torch.Tensor] = None  # (B, n_obs, 2)
        self._obstacle_states: Optional[torch.Tensor] = None    # (B, n_obs, 6)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return 6

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def node_dim(self) -> int:
        return 3

    @property
    def edge_dim(self) -> int:
        return 8

    @property
    def comm_radius(self) -> float:
        return self.params["comm_radius"]

    @property
    def n_obs(self) -> int:
        return self.params["n_obs"]

    @property
    def g_x_matrix(self) -> np.ndarray:
        m = self.params["mass"]
        I = self.params["inertia"]
        return np.array(
            [[0, 0, 0], [0, 0, 0], [1/m, 0, 0],
             [0, 1/m, 0], [0, 0, 0], [0, 0, 1/I]],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Reset — all B environments
    # ------------------------------------------------------------------
    def reset(self, device: torch.device, seed: Optional[int] = None):
        """
        Reset all B environments at once.

        Each environment gets its own random agents, goals, and obstacles.
        """
        rng = np.random.default_rng(seed)
        B = self.batch_size
        n = self.num_agents
        area = self.area_size
        R_form = self.params["R_form"]
        drone_r = self.params["drone_radius"]
        margin = R_form + drone_r
        n_obs = self.params["n_obs"]
        obs_lo, obs_hi = self.params.get("obs_len_range", (0.1, 0.3))
        self._step_count = 0

        # ---- Generate obstacles for all B envs ----
        if n_obs > 0:
            obs_cx = rng.uniform(margin, area - margin, size=(B, n_obs))
            obs_cy = rng.uniform(margin, area - margin, size=(B, n_obs))
            obs_hw = rng.uniform(obs_lo, obs_hi, size=(B, n_obs)) / 2.0
            obs_hh = rng.uniform(obs_lo, obs_hi, size=(B, n_obs)) / 2.0
            self._obstacle_centers = torch.tensor(
                np.stack([obs_cx, obs_cy], axis=-1), dtype=torch.float32, device=device
            )  # (B, n_obs, 2)
            self._obstacle_half_sizes = torch.tensor(
                np.stack([obs_hw, obs_hh], axis=-1), dtype=torch.float32, device=device
            )  # (B, n_obs, 2)
            obs_6d = np.zeros((B, n_obs, 6), dtype=np.float32)
            obs_6d[:, :, 0] = obs_cx
            obs_6d[:, :, 1] = obs_cy
            self._obstacle_states = torch.tensor(obs_6d, dtype=torch.float32, device=device)
        else:
            self._obstacle_centers = None
            self._obstacle_half_sizes = None
            self._obstacle_states = torch.zeros(B, 0, 6, dtype=torch.float32, device=device)

        # ---- Generate agent/goal positions ----
        # Simple rejection sampling per env (fast — only runs once at reset)
        all_start_pos = np.empty((B, n, 2), dtype=np.float32)
        all_goal_pos = np.empty((B, n, 2), dtype=np.float32)
        for b in range(B):
            all_start_pos[b] = self._sample_free_pos(rng, n, margin, b)
            all_goal_pos[b] = self._sample_free_pos(rng, n, margin, b)

        start_theta = rng.uniform(-math.pi, math.pi, size=(B, n, 1)).astype(np.float32)

        agent_6d = np.concatenate([
            all_start_pos,
            np.zeros((B, n, 2), dtype=np.float32),
            start_theta,
            np.zeros((B, n, 1), dtype=np.float32),
        ], axis=-1)
        goal_6d = np.concatenate([
            all_goal_pos,
            np.zeros((B, n, 2), dtype=np.float32),
            np.zeros((B, n, 1), dtype=np.float32),
            np.zeros((B, n, 1), dtype=np.float32),
        ], axis=-1)

        self._agent_states = torch.tensor(agent_6d, dtype=torch.float32, device=device)
        self._goal_states = torch.tensor(goal_6d, dtype=torch.float32, device=device)

        # Move persistent tensors to device
        self._K_trans = self._K_trans.to(device)
        self._local_offsets = self._local_offsets.to(device)

    def _sample_free_pos(self, rng, count, margin, batch_idx):
        """Sample ``count`` CoM positions that don't overlap obstacles (for batch b)."""
        area = self.area_size
        positions = np.empty((0, 2))
        while positions.shape[0] < count:
            cands = rng.uniform(margin, area - margin, size=(count * 4, 2)).astype(np.float32)
            free = np.ones(cands.shape[0], dtype=bool)
            if self._obstacle_centers is not None:
                oc = self._obstacle_centers[batch_idx].cpu().numpy()  # (n_obs, 2)
                ohs = self._obstacle_half_sizes[batch_idx].cpu().numpy() + margin
                for j in range(oc.shape[0]):
                    inside = (np.abs(cands[:, 0] - oc[j, 0]) < ohs[j, 0]) & \
                             (np.abs(cands[:, 1] - oc[j, 1]) < ohs[j, 1])
                    free &= ~inside
            cands = cands[free]
            positions = np.concatenate([positions, cands], axis=0)
        return positions[:count]

    # ------------------------------------------------------------------
    # Step — batched Euler integration
    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """
        Apply action (B, n, 3) and advance one timestep for all B envs.
        """
        m = self.params["mass"]
        I = self.params["inertia"]
        dt = self.dt
        u_max = self.params.get("u_max")
        alpha_max = self.params.get("alpha_max")
        v_max = self.params.get("v_max")
        omega_max = self.params.get("omega_max")

        # Clamp
        if u_max is not None:
            action = torch.cat([
                torch.clamp(action[:, :, :2], -u_max, u_max),
                action[:, :, 2:3],
            ], dim=-1)
        if alpha_max is not None:
            action = torch.cat([
                action[:, :, :2],
                torch.clamp(action[:, :, 2:3], -alpha_max, alpha_max),
            ], dim=-1)

        x = self._agent_states
        accel = action[:, :, :2] / m
        new_pos = x[:, :, :2] + x[:, :, 2:4] * dt + 0.5 * accel * dt ** 2
        new_vel = x[:, :, 2:4] + accel * dt

        angular_accel = action[:, :, 2:3] / I
        new_theta = x[:, :, 4:5] + x[:, :, 5:6] * dt + 0.5 * angular_accel * dt ** 2
        new_omega = x[:, :, 5:6] + angular_accel * dt

        if v_max is not None:
            new_vel = torch.clamp(new_vel, -v_max, v_max)
        if omega_max is not None:
            new_omega = torch.clamp(new_omega, -omega_max, omega_max)
        new_theta = _wrap_angle(new_theta)

        self._agent_states = torch.cat([new_pos, new_vel, new_theta, new_omega], dim=-1)
        self._step_count += 1

    # ------------------------------------------------------------------
    # Nominal controller — batched LQR+PD
    # ------------------------------------------------------------------
    def nominal_controller(self) -> torch.Tensor:
        """
        Compute u_nom = [a_x, a_y, α] for all B envs.  Returns (B, n, 3).
        """
        x = self._agent_states   # (B, n, 6)
        g = self._goal_states    # (B, n, 6)
        u_max = self.params.get("u_max")
        alpha_max = self.params.get("alpha_max")

        # Translational LQR
        err = x[:, :, :4] - g[:, :, :4]    # (B, n, 4)
        u_trans = -torch.einsum("...j,ij->...i", err, self._K_trans)  # (B, n, 2)
        if u_max is not None:
            u_trans = torch.clamp(u_trans, -u_max, u_max)

        # Angular PD
        theta_err = _wrap_angle(x[:, :, 4] - g[:, :, 4])  # (B, n)
        omega = x[:, :, 5]
        u_alpha = -(self._Kp_theta * theta_err + self._Kd_theta * omega)
        if alpha_max is not None:
            u_alpha = torch.clamp(u_alpha, -alpha_max, alpha_max)

        return torch.cat([u_trans, u_alpha.unsqueeze(-1)], dim=-1)  # (B, n, 3)

    # ------------------------------------------------------------------
    # Unsafe mask — batched 9-point drone-to-drone distances
    # ------------------------------------------------------------------
    def unsafe_mask(self) -> torch.Tensor:
        """
        Compute per-agent collision mask for all B envs.

        Returns (B, n_agents) bool tensor.

        Agent–agent: 3×3=9 pairwise drone distances, collision if any < 2*r.
        Agent–obstacle: each drone vs padded obstacle rectangle.
        """
        B = self.batch_size
        n = self.num_agents
        drone_r = self.params["drone_radius"]
        device = self._agent_states.device

        # ---- Drone positions (B, n, 3, 2) ----
        offsets = self._local_offsets
        R_mat = _rotation_matrix(self._agent_states[:, :, 4])  # (B, n, 2, 2)
        rotated = torch.einsum("bnij,kj->bnki", R_mat, offsets)
        drone_pos = self._agent_states[:, :, :2].unsqueeze(2) + rotated

        # ---- Agent–agent collision ----
        agent_collision = torch.zeros(B, n, dtype=torch.bool, device=device)
        if n > 1:
            diff = (
                drone_pos[:, :, None, :, None, :]
                - drone_pos[:, None, :, None, :, :]
            )  # (B, n, n, 3, 3, 2)
            dist = torch.norm(diff, dim=-1)  # (B, n, n, 3, 3)
            dist_min = dist.reshape(B, n, n, -1).min(dim=-1).values  # (B, n, n)
            dist_min = dist_min + torch.eye(n, device=device).unsqueeze(0) * 1e6
            collision_matrix = dist_min < 2 * drone_r
            agent_collision = collision_matrix.any(dim=2)  # (B, n)

        # ---- Agent–obstacle collision ----
        obs_collision = torch.zeros(B, n, dtype=torch.bool, device=device)
        if self._obstacle_centers is not None and self._obstacle_centers.shape[1] > 0:
            n_obs = self._obstacle_centers.shape[1]
            # drone_pos: (B, n, 3, 2), obs centers: (B, n_obs, 2)
            # Reshape for broadcasting:
            # drone_pos → (B, n, 3, 1, 2)
            # obs_centers → (B, 1, 1, n_obs, 2)
            dp = drone_pos.unsqueeze(3)                               # (B, n, 3, 1, 2)
            oc = self._obstacle_centers.reshape(B, 1, 1, n_obs, 2)   # (B, 1, 1, n_obs, 2)
            ohs = self._obstacle_half_sizes.reshape(B, 1, 1, n_obs, 2) + drone_r

            diff_obs = torch.abs(dp - oc)  # (B, n, 3, n_obs, 2)
            inside = (diff_obs[..., 0] < ohs[..., 0]) & (diff_obs[..., 1] < ohs[..., 1])
            # inside: (B, n, 3, n_obs) — any drone k, any obs j
            obs_collision = inside.any(dim=-1).any(dim=-1)  # (B, n)

        return agent_collision | obs_collision

    # ------------------------------------------------------------------
    # Build mega-graph from all B environments
    # ------------------------------------------------------------------
    def build_batch_graph(self, agent_states=None) -> GraphsTuple:
        """
        Build ONE mega-graph from all B environments using the fully
        vectorized builder.  Returns GraphsTuple with B*N_per nodes.
        """
        if agent_states is None:
            agent_states = self._agent_states
        return build_vectorized_swarm_graph(
            agent_states=agent_states,
            goal_states=self._goal_states,
            obstacle_states=self._obstacle_states,
            local_offsets=self._local_offsets,
            comm_radius=self.comm_radius,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
        )
