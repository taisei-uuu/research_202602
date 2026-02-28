"""
Vectorized (batched) swarm environment — 4D Bounding Circle.

All B environments run in parallel on GPU using (B, n_agents, 4) state
tensors.  Bounding circle safety (r_swarm = 0.4).

State:   [px, py, vx, vy]  (4D)
Control: [ax, ay]           (2D)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils.graph import GraphsTuple
from ..utils.swarm_graph import build_vectorized_swarm_graph


# ── LQR helper ───────────────────────────────────────────────────────

def _dlqr(A, B, Q, R):
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
    Vectorized swarm env — 4D point mass with bounding circle safety.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "r_swarm": 0.4,
        "comm_radius": 2.0,
        "n_obs": 2,
        "obs_len_range": (0.1, 0.3),
        "mass": 0.1,
        "u_max": 0.3,
        "v_max": 1.0,
        "R_form": 0.3,
    }

    def __init__(
        self,
        num_agents: int = 3,
        batch_size: int = 256,
        area_size: float = 4.0,
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

        # LQR gain (4D → 2D)
        m = self.params["mass"]
        A_ct = np.zeros((4, 4), dtype=np.float32)
        A_ct[0, 2] = 1.0
        A_ct[1, 3] = 1.0
        A_t = A_ct * dt + np.eye(4, dtype=np.float32)
        B_t = np.array([[0, 0], [0, 0], [1/m, 0], [0, 1/m]], dtype=np.float32) * dt
        Q_t = np.eye(4, dtype=np.float32) * 5.0
        R_t = np.eye(2, dtype=np.float32)
        self._K = torch.tensor(_dlqr(A_t, B_t, Q_t, R_t), dtype=torch.float32)

        # State tensors (set on reset)
        self._agent_states: Optional[torch.Tensor] = None
        self._goal_states: Optional[torch.Tensor] = None
        self._obstacle_centers: Optional[torch.Tensor] = None
        self._obstacle_half_sizes: Optional[torch.Tensor] = None
        self._obstacle_states: Optional[torch.Tensor] = None
        self._step_count = 0

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def node_dim(self) -> int:
        return 3

    @property
    def edge_dim(self) -> int:
        return 4

    @property
    def comm_radius(self) -> float:
        return self.params["comm_radius"]

    @property
    def n_obs(self) -> int:
        return self.params["n_obs"]

    @property
    def g_x_matrix(self) -> np.ndarray:
        m = self.params["mass"]
        return np.array([[0, 0], [0, 0], [1/m, 0], [0, 1/m]], dtype=np.float32)

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, device: torch.device, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        B = self.batch_size
        n = self.num_agents
        area = self.area_size
        r_swarm = self.params["r_swarm"]
        margin = r_swarm + 0.1
        n_obs = self.params["n_obs"]
        obs_lo, obs_hi = self.params.get("obs_len_range", (0.1, 0.3))
        self._step_count = 0

        # Obstacles
        if n_obs > 0:
            obs_cx = rng.uniform(margin, area - margin, size=(B, n_obs))
            obs_cy = rng.uniform(margin, area - margin, size=(B, n_obs))
            obs_hw = rng.uniform(obs_lo, obs_hi, size=(B, n_obs)) / 2.0
            obs_hh = rng.uniform(obs_lo, obs_hi, size=(B, n_obs)) / 2.0
            self._obstacle_centers = torch.tensor(
                np.stack([obs_cx, obs_cy], axis=-1), dtype=torch.float32, device=device
            )
            self._obstacle_half_sizes = torch.tensor(
                np.stack([obs_hw, obs_hh], axis=-1), dtype=torch.float32, device=device
            )
            obs_4d = np.zeros((B, n_obs, 4), dtype=np.float32)
            obs_4d[:, :, 0] = obs_cx
            obs_4d[:, :, 1] = obs_cy
            self._obstacle_states = torch.tensor(obs_4d, dtype=torch.float32, device=device)
        else:
            self._obstacle_centers = None
            self._obstacle_half_sizes = None
            self._obstacle_states = torch.zeros(B, 0, 4, dtype=torch.float32, device=device)

        # Agent and goal positions
        all_start = np.empty((B, n, 2), dtype=np.float32)
        all_goal = np.empty((B, n, 2), dtype=np.float32)
        for b in range(B):
            all_start[b] = self._sample_free_pos(rng, n, margin, b)
            all_goal[b] = self._sample_free_pos(rng, n, margin, b)

        agent_4d = np.concatenate([all_start, np.zeros((B, n, 2), dtype=np.float32)], axis=-1)
        goal_4d = np.concatenate([all_goal, np.zeros((B, n, 2), dtype=np.float32)], axis=-1)

        self._agent_states = torch.tensor(agent_4d, dtype=torch.float32, device=device)
        self._goal_states = torch.tensor(goal_4d, dtype=torch.float32, device=device)
        self._K = self._K.to(device)

    def _sample_free_pos(self, rng, count, margin, batch_idx):
        area = self.area_size
        positions = np.empty((0, 2))
        while positions.shape[0] < count:
            cands = rng.uniform(margin, area - margin, size=(count * 4, 2)).astype(np.float32)
            free = np.ones(cands.shape[0], dtype=bool)
            if self._obstacle_centers is not None:
                oc = self._obstacle_centers[batch_idx].cpu().numpy()
                ohs = self._obstacle_half_sizes[batch_idx].cpu().numpy() + margin
                for j in range(oc.shape[0]):
                    inside = (np.abs(cands[:, 0] - oc[j, 0]) < ohs[j, 0]) & \
                             (np.abs(cands[:, 1] - oc[j, 1]) < ohs[j, 1])
                    free &= ~inside
            cands = cands[free]
            positions = np.concatenate([positions, cands], axis=0)
        return positions[:count]

    # ── Step ──────────────────────────────────────────────────────────
    def step(self, action: torch.Tensor):
        """action: (B, n, 2)."""
        m = self.params["mass"]
        dt = self.dt
        u_max = self.params.get("u_max")
        v_max = self.params.get("v_max")

        if u_max is not None:
            action = torch.clamp(action, -u_max, u_max)

        accel = action / m
        x = self._agent_states
        new_pos = x[:, :, :2] + x[:, :, 2:4] * dt + 0.5 * accel * dt**2
        new_vel = x[:, :, 2:4] + accel * dt
        if v_max is not None:
            new_vel = torch.clamp(new_vel, -v_max, v_max)

        self._agent_states = torch.cat([new_pos, new_vel], dim=-1)
        self._step_count += 1

    # ── Nominal controller (LQR) ─────────────────────────────────────
    def nominal_controller(self) -> torch.Tensor:
        """Returns (B, n, 2)."""
        err = self._agent_states - self._goal_states  # (B, n, 4)
        u = -torch.einsum("...j,ij->...i", err, self._K)  # (B, n, 2)
        u_max = self.params.get("u_max")
        if u_max is not None:
            u = torch.clamp(u, -u_max, u_max)
        return u

    # ── Unsafe mask (bounding circle) ─────────────────────────────────
    def unsafe_mask(self) -> torch.Tensor:
        """Returns (B, n) bool."""
        B = self.batch_size
        n = self.num_agents
        r_swarm = self.params["r_swarm"]
        device = self._agent_states.device

        pos = self._agent_states[:, :, :2]  # (B, n, 2)
        agent_collision = torch.zeros(B, n, dtype=torch.bool, device=device)

        if n > 1:
            diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, n, n, 2)
            dist = torch.norm(diff, dim=-1)              # (B, n, n)
            dist = dist + torch.eye(n, device=device).unsqueeze(0) * 1e6
            collision_matrix = dist < 2 * r_swarm
            agent_collision = collision_matrix.any(dim=2)

        obs_collision = torch.zeros(B, n, dtype=torch.bool, device=device)
        if self._obstacle_centers is not None and self._obstacle_centers.shape[1] > 0:
            n_obs = self._obstacle_centers.shape[1]
            # pos: (B, n, 2) vs obs: (B, n_obs, 2)
            dp = pos.unsqueeze(2)                                        # (B, n, 1, 2)
            oc = self._obstacle_centers.unsqueeze(1)                     # (B, 1, n_obs, 2)
            ohs = self._obstacle_half_sizes.unsqueeze(1) + r_swarm      # (B, 1, n_obs, 2)
            diff_obs = torch.abs(dp - oc)                                # (B, n, n_obs, 2)
            inside = (diff_obs[..., 0] < ohs[..., 0]) & (diff_obs[..., 1] < ohs[..., 1])
            obs_collision = inside.any(dim=-1)  # (B, n)

        return agent_collision | obs_collision

    # ── Graph builder ─────────────────────────────────────────────────
    def build_batch_graph(self, agent_states=None) -> GraphsTuple:
        if agent_states is None:
            agent_states = self._agent_states
        return build_vectorized_swarm_graph(
            agent_states=agent_states,
            goal_states=self._goal_states,
            obstacle_states=self._obstacle_states,
            comm_radius=self.comm_radius,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
        )
