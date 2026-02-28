"""
Swarm environment — 4D Bounding Circle approach.

Each swarm is modeled as a point mass with a large bounding circle
(r_swarm = 0.4 m) that safely encompasses the 3-drone triangle formation.

State:   [px, py, vx, vy]        (4D)
Control: [ax, ay]                 (2D)
Dynamics: Standard double integrator (translation only).
Safety:  Bounding circle collision (CoM distance < 2 * r_swarm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..utils.graph import GraphsTuple
from ..utils.swarm_graph import build_swarm_graph_from_states


# ── Obstacle dataclass ───────────────────────────────────────────────

@dataclass
class Obstacle:
    center: torch.Tensor   # (2,)
    half_size: torch.Tensor  # (2,)


# ── LQR helper ───────────────────────────────────────────────────────

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


class SwarmIntegrator:
    """
    4D bounding-circle swarm environment (single-instance, for visualization).

    Functionally identical to DoubleIntegrator but with r_swarm = 0.4 m.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "r_swarm": 0.4,
        "comm_radius": 2.0,
        "n_obs": 2,
        "obs_len_range": (0.1, 0.3),
        "mass": 0.1,
        "u_max": 0.3,
        "v_max": 1.0,
        "R_form": 0.3,   # kept for visualization of triangle
    }

    def __init__(
        self,
        num_agents: int = 3,
        area_size: float = 4.0,
        dt: float = 0.03,
        max_steps: int = 256,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.num_agents = num_agents
        self.area_size = area_size
        self.dt = dt
        self.max_steps = max_steps
        self.params = {**self.DEFAULT_PARAMS}
        if params is not None:
            self.params.update(params)

        # LQR gain for 4D translational system
        m = self.params["mass"]
        A_ct = np.zeros((4, 4), dtype=np.float32)
        A_ct[0, 2] = 1.0
        A_ct[1, 3] = 1.0
        A_t = A_ct * dt + np.eye(4, dtype=np.float32)
        B_t = np.array([[0, 0], [0, 0], [1/m, 0], [0, 1/m]], dtype=np.float32) * dt
        Q_t = np.eye(4, dtype=np.float32) * 5.0
        R_t = np.eye(2, dtype=np.float32)
        self._K = torch.tensor(_dlqr(A_t, B_t, Q_t, R_t), dtype=torch.float32)

        self._obstacles: List[Obstacle] = []
        self._obstacle_states: Optional[torch.Tensor] = None
        self.agent_states: Optional[torch.Tensor] = None
        self.goal_states: Optional[torch.Tensor] = None
        self._step_count = 0

    # ── Properties ────────────────────────────────────────────────────
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
    def g_x_matrix(self) -> np.ndarray:
        m = self.params["mass"]
        return np.array(
            [[0, 0], [0, 0], [1/m, 0], [0, 1/m]],
            dtype=np.float32,
        )

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        n = self.num_agents
        area = self.area_size
        r_swarm = self.params["r_swarm"]
        margin = r_swarm + 0.1
        n_obs = self.params["n_obs"]

        # Obstacles
        self._obstacles = []
        obs_lo, obs_hi = self.params.get("obs_len_range", (0.1, 0.3))
        for _ in range(n_obs):
            cx = rng.uniform(margin, area - margin)
            cy = rng.uniform(margin, area - margin)
            hw = rng.uniform(obs_lo, obs_hi) / 2.0
            hh = rng.uniform(obs_lo, obs_hi) / 2.0
            self._obstacles.append(Obstacle(
                center=torch.tensor([cx, cy], dtype=torch.float32),
                half_size=torch.tensor([hw, hh], dtype=torch.float32),
            ))

        obs_states = []
        for obs in self._obstacles:
            s = torch.zeros(4)
            s[0] = obs.center[0]
            s[1] = obs.center[1]
            obs_states.append(s)
        self._obstacle_states = torch.stack(obs_states) if obs_states else None

        # Agent / goal positions (collision-free)
        start_pos = self._sample_free_positions(rng, n, margin)
        goal_pos = self._sample_free_positions(rng, n, margin)

        self.agent_states = torch.tensor(
            np.concatenate([start_pos, np.zeros((n, 2))], axis=1),
            dtype=torch.float32,
        )
        self.goal_states = torch.tensor(
            np.concatenate([goal_pos, np.zeros((n, 2))], axis=1),
            dtype=torch.float32,
        )
        self._step_count = 0

    def _sample_free_positions(self, rng, count, margin):
        """Sample positions that avoid obstacles AND are >= 2*r_swarm apart."""
        area = self.area_size
        min_dist = 2 * self.params["r_swarm"]
        positions = []
        for _ in range(100_000):
            if len(positions) >= count:
                break
            p = rng.uniform(margin, area - margin, size=2).astype(np.float32)
            # Check obstacle collision
            ok = True
            for obs in self._obstacles:
                oc = obs.center.numpy()
                ohs = obs.half_size.numpy() + margin
                if abs(p[0] - oc[0]) < ohs[0] and abs(p[1] - oc[1]) < ohs[1]:
                    ok = False
                    break
            if not ok:
                continue
            # Check inter-agent distance
            for existing in positions:
                if np.linalg.norm(p - existing) < min_dist:
                    ok = False
                    break
            if ok:
                positions.append(p)
        return np.array(positions)

    def to(self, device):
        self.agent_states = self.agent_states.to(device)
        self.goal_states = self.goal_states.to(device)
        if self._obstacle_states is not None:
            self._obstacle_states = self._obstacle_states.to(device)
        self._K = self._K.to(device)
        return self

    # ── Step ──────────────────────────────────────────────────────────
    def step(self, action: torch.Tensor):
        m = self.params["mass"]
        dt = self.dt
        u_max = self.params.get("u_max")
        v_max = self.params.get("v_max")

        if u_max is not None:
            action = torch.clamp(action, -u_max, u_max)

        accel = action / m
        new_pos = self.agent_states[:, :2] + self.agent_states[:, 2:4] * dt + 0.5 * accel * dt**2
        new_vel = self.agent_states[:, 2:4] + accel * dt
        if v_max is not None:
            new_vel = torch.clamp(new_vel, -v_max, v_max)

        self.agent_states = torch.cat([new_pos, new_vel], dim=-1)
        self._step_count += 1
        return self.agent_states, {"done": self._step_count >= self.max_steps}

    # ── Nominal controller (LQR) ─────────────────────────────────────
    def nominal_controller(self) -> torch.Tensor:
        err = self.agent_states - self.goal_states
        u = -err @ self._K.T
        u_max = self.params.get("u_max")
        if u_max is not None:
            u = torch.clamp(u, -u_max, u_max)
        return u

    # ── Unsafe mask (bounding circle) ─────────────────────────────────
    def unsafe_mask(self) -> torch.Tensor:
        n = self.num_agents
        r_swarm = self.params["r_swarm"]
        device = self.agent_states.device

        pos = self.agent_states[:, :2]
        agent_collision = torch.zeros(n, dtype=torch.bool, device=device)

        if n > 1:
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (n, n, 2)
            dist = torch.norm(diff, dim=-1)                # (n, n)
            dist = dist + torch.eye(n, device=device) * 1e6
            collision_matrix = dist < 2 * r_swarm
            agent_collision = collision_matrix.any(dim=1)

        # Obstacle collision
        obs_collision = torch.zeros(n, dtype=torch.bool, device=device)
        for obs in self._obstacles:
            c = obs.center.to(device)
            hs = obs.half_size.to(device) + r_swarm
            inside = (torch.abs(pos[:, 0] - c[0]) < hs[0]) & \
                     (torch.abs(pos[:, 1] - c[1]) < hs[1])
            obs_collision = obs_collision | inside

        return agent_collision | obs_collision

    # ── Graph builder ─────────────────────────────────────────────────
    def _get_graph(self) -> GraphsTuple:
        return build_swarm_graph_from_states(
            agent_states=self.agent_states,
            goal_states=self.goal_states,
            obstacle_positions=self._obstacle_states,
            comm_radius=self.comm_radius,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
        )
