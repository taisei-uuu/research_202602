"""
Double Integrator multi-agent environment.

State  x = [p_x, p_y, v_x, v_y]   (dim 4)
Action u = [a_x, a_y]              (dim 2)

Dynamics (Euler):
    p_{t+1} = p_t + v_t · dt + 0.5 · a_t · dt²
    v_{t+1} = v_t + a_t · dt

The environment is designed to be modular so that it can later be extended
to "Swarm formation" with leader–follower offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..utils.graph import GraphsTuple, build_graph_from_states


# ---------------------------------------------------------------------------
# LQR helper  (discrete-time, infinite-horizon, for nominal controller)
# ---------------------------------------------------------------------------

def _dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solve the discrete-time LQR problem via iterative DARE.

    Returns the gain matrix K such that u = -K x.
    """
    P = Q.copy()
    for _ in range(1000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_new - P)) < 1e-10:
            break
        P = P_new
    return K


# ---------------------------------------------------------------------------
# Obstacle data  (simple rectangle — can be extended)
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """Axis-aligned rectangle obstacle."""
    center: torch.Tensor     # (2,)
    half_size: torch.Tensor  # (2,)  — half-width and half-height

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """Check if *points* (N,2) are inside the obstacle."""
        diff = torch.abs(points - self.center.unsqueeze(0))
        return (diff[:, 0] < self.half_size[0]) & (diff[:, 1] < self.half_size[1])


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DoubleIntegrator:
    """
    2-D multi-agent system with double-integrator dynamics.

    Parameters
    ----------
    num_agents : int
        Number of agents.
    area_size : float
        Side length of the square arena.
    dt : float
        Simulation time-step.
    max_steps : int
        Maximum episode length.
    params : dict, optional
        Override default parameters (see ``DEFAULT_PARAMS``).
    """

    # Node-type identifiers
    AGENT = 0
    GOAL = 1
    OBSTACLE = 2

    DEFAULT_PARAMS: Dict[str, Any] = {
        "car_radius": 0.05,
        "comm_radius": 0.5,       # sensing / communication radius  R
        "n_rays": 32,             # (reserved for LiDAR extension)
        "obs_len_range": (0.1, 0.5),
        "n_obs": 8,               # number of random obstacles
        "mass": 0.1,
        "u_max": 0.1,             # max control input (gives max accel [-1.0, 1.0])
        "v_max": 0.5,             # max velocity [-0.5, 0.5]
    }

    def __init__(
        self,
        num_agents: int = 4,
        area_size: float = 10.0,
        dt: float = 0.03,
        max_steps: int = 256,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.num_agents = num_agents
        self.area_size = area_size
        self.dt = dt
        self.max_steps = max_steps

        # Merge user params with defaults
        self.params = {**self.DEFAULT_PARAMS}
        if params is not None:
            self.params.update(params)

        # Continuous-time system matrices → discrete
        state_dim = self.state_dim
        action_dim = self.action_dim
        m = self.params["mass"]

        A_ct = np.zeros((state_dim, state_dim), dtype=np.float32)
        A_ct[0, 2] = 1.0
        A_ct[1, 3] = 1.0
        self._A = A_ct * dt + np.eye(state_dim, dtype=np.float32)

        self._B = np.array(
            [[0.0, 0.0],
             [0.0, 0.0],
             [1.0 / m, 0.0],
             [0.0, 1.0 / m]],
            dtype=np.float32,
        ) * dt

        # LQR nominal controller:  u_ref = -K (x - x_goal)
        Q = np.eye(state_dim, dtype=np.float32) * 5.0
        R = np.eye(action_dim, dtype=np.float32)
        self._K = torch.tensor(_dlqr(self._A, self._B, Q, R), dtype=torch.float32)

        # Internal state
        self._step_count: int = 0
        self._agent_states: Optional[torch.Tensor] = None   # (n, 4)
        self._goal_states: Optional[torch.Tensor] = None     # (n, 4)
        self._obstacles: list[Obstacle] = []
        self._obstacle_states: Optional[torch.Tensor] = None  # (n_obs, 4)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        """Per-agent state dimension: [px, py, vx, vy]."""
        return 4

    @property
    def action_dim(self) -> int:
        """Per-agent action dimension: [ax, ay]."""
        return 2

    @property
    def node_dim(self) -> int:
        """One-hot node-type indicator dimension."""
        return 3  # agent / goal / obstacle

    @property
    def edge_dim(self) -> int:
        """Edge feature dimension: [Δpx, Δpy, Δvx, Δvy]."""
        return 4

    @property
    def comm_radius(self) -> float:
        return self.params["comm_radius"]

    @property
    def g_x_matrix(self) -> np.ndarray:
        """Continuous-time control input matrix g(x)."""
        m = self.params.get("mass", 0.1)
        return np.array(
            [[0.0, 0.0],
             [0.0, 0.0],
             [1.0 / m, 0.0],
             [0.0, 1.0 / m]],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> GraphsTuple:
        """
        Reset the environment and return the initial graph.

        Randomly places agents, goals, and rectangular obstacles.
        """
        rng = np.random.default_rng(seed)
        self._step_count = 0

        n = self.num_agents
        area = self.area_size
        car_r = self.params["car_radius"]
        n_obs = self.params["n_obs"]
        obs_lo, obs_hi = self.params["obs_len_range"]

        # ---- Generate obstacles ----
        self._obstacles = []
        obs_positions = []
        for _ in range(n_obs):
            cx, cy = rng.uniform(0.0, area, size=2)
            hw = rng.uniform(obs_lo, obs_hi) / 2.0
            hh = rng.uniform(obs_lo, obs_hi) / 2.0
            obs = Obstacle(
                center=torch.tensor([cx, cy], dtype=torch.float32),
                half_size=torch.tensor([hw, hh], dtype=torch.float32),
            )
            self._obstacles.append(obs)
            obs_positions.append([cx, cy, 0.0, 0.0])

        if n_obs > 0:
            self._obstacle_states = torch.tensor(obs_positions, dtype=torch.float32)
        else:
            self._obstacle_states = None

        # ---- Generate agent and goal positions (collision-free) ----
        def _sample_free_positions(count: int) -> np.ndarray:
            """Sample *count* positions that don't overlap obstacles."""
            positions = np.empty((0, 2))
            while positions.shape[0] < count:
                candidates = rng.uniform(car_r, area - car_r, size=(count * 4, 2))
                free = np.ones(candidates.shape[0], dtype=bool)
                for obs in self._obstacles:
                    c = obs.center.numpy()
                    hs = obs.half_size.numpy() + car_r
                    inside = (np.abs(candidates[:, 0] - c[0]) < hs[0]) & (
                        np.abs(candidates[:, 1] - c[1]) < hs[1]
                    )
                    free &= ~inside
                candidates = candidates[free]
                positions = np.concatenate([positions, candidates], axis=0)
            return positions[:count]

        start_pos = _sample_free_positions(n)
        goal_pos = _sample_free_positions(n)

        self._agent_states = torch.tensor(
            np.concatenate([start_pos, np.zeros((n, 2))], axis=1),
            dtype=torch.float32,
        )
        self._goal_states = torch.tensor(
            np.concatenate([goal_pos, np.zeros((n, 2))], axis=1),
            dtype=torch.float32,
        )

        return self._get_graph()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> Tuple[GraphsTuple, Dict[str, Any]]:
        """
        Apply *action* (n_agents, 2) and advance one time-step.

        Returns
        -------
        graph : GraphsTuple — the new observation.
        info  : dict        — contains ``done``, ``step``.
        """
        assert action.shape == (self.num_agents, self.action_dim)
        
        u_max = self.params.get("u_max")
        if u_max is not None:
            action = torch.clamp(action, -u_max, u_max)

        x = self._agent_states
        m = self.params["mass"]
        dt = self.dt

        accel = action / m                     # a = F / m
        new_pos = x[:, :2] + x[:, 2:] * dt + 0.5 * accel * dt ** 2
        new_vel = x[:, 2:] + accel * dt

        v_max = self.params.get("v_max")
        if v_max is not None:
            new_vel = torch.clamp(new_vel, -v_max, v_max)

        self._agent_states = torch.cat([new_pos, new_vel], dim=1)
        self._step_count += 1

        done = self._step_count >= self.max_steps
        info = {"done": done, "step": self._step_count}

        return self._get_graph(), info

    # ------------------------------------------------------------------
    # Nominal controller (LQR)
    # ------------------------------------------------------------------
    def nominal_controller(self) -> torch.Tensor:
        """
        Compute the LQR reference control  u_ref = -K (x − x_goal).

        Returns
        -------
        u_ref : (n_agents, action_dim)
        """
        error = self._agent_states - self._goal_states  # (n, 4)
        u_ref = -error @ self._K.T                       # (n, 2)
        u_max = self.params.get("u_max")
        if u_max is not None:
            u_ref = torch.clamp(u_ref, -u_max, u_max)
        return u_ref

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _get_graph(self) -> GraphsTuple:
        """Build the ``GraphsTuple`` from the current state."""
        return build_graph_from_states(
            agent_states=self._agent_states,
            goal_states=self._goal_states,
            obstacle_positions=self._obstacle_states,
            comm_radius=self.comm_radius,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @property
    def agent_states(self) -> torch.Tensor:
        """Current agent states (n_agents, 4)."""
        return self._agent_states

    @property
    def goal_states(self) -> torch.Tensor:
        """Current goal states (n_agents, 4)."""
        return self._goal_states

    def to(self, device: torch.device) -> "DoubleIntegrator":
        """
        Move all internal tensors to *device* (e.g. ``torch.device('cuda')``).

        Returns ``self`` for chaining.
        """
        self._K = self._K.to(device)
        if self._agent_states is not None:
            self._agent_states = self._agent_states.to(device)
        if self._goal_states is not None:
            self._goal_states = self._goal_states.to(device)
        if self._obstacle_states is not None:
            self._obstacle_states = self._obstacle_states.to(device)
        for obs in self._obstacles:
            obs.center = obs.center.to(device)
            obs.half_size = obs.half_size.to(device)
        return self

    # ------------------------------------------------------------------
    # Safety masks  (needed by CBF training)
    # ------------------------------------------------------------------
    def unsafe_mask(self) -> torch.Tensor:
        """
        True for each agent that is in collision (with obstacle or another agent).

        Returns : (n_agents,) bool tensor.
        """
        n = self.num_agents
        pos = self._agent_states[:, :2]       # (n, 2)
        car_r = self.params["car_radius"]

        # Agent–obstacle collision
        obs_collision = torch.zeros(n, dtype=torch.bool, device=pos.device)
        for obs in self._obstacles:
            padded_hs = obs.half_size + car_r
            diff = torch.abs(pos - obs.center.unsqueeze(0))
            inside = (diff[:, 0] < padded_hs[0]) & (diff[:, 1] < padded_hs[1])
            obs_collision = obs_collision | inside

        # Agent–agent collision (distance < 2 * car_radius)
        dists = torch.cdist(pos, pos)         # (n, n)
        dists = dists + torch.eye(n, device=pos.device) * 1e6    # ignore self
        agent_collision = (dists < 2 * car_r).any(dim=1)

        return obs_collision | agent_collision

    def safe_mask(self) -> torch.Tensor:
        """True for each agent that is collision-free."""
        return ~self.unsafe_mask()

    # ------------------------------------------------------------------
    # Differentiable forward simulation
    # ------------------------------------------------------------------
    def forward_step(
        self, agent_states: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate one Euler step (differentiable, does NOT modify env state).

        Parameters
        ----------
        agent_states : (n_agents, 4)  — may have requires_grad=True
        action       : (n_agents, 2)

        Returns
        -------
        next_states : (n_agents, 4)
        """
        m = self.params["mass"]
        dt = self.dt
        
        u_max = self.params.get("u_max")
        if u_max is not None:
            action = torch.clamp(action, -u_max, u_max)
            
        accel = action / m
        new_pos = agent_states[:, :2] + agent_states[:, 2:] * dt + 0.5 * accel * dt ** 2
        new_vel = agent_states[:, 2:] + accel * dt
        
        v_max = self.params.get("v_max")
        if v_max is not None:
            new_vel = torch.clamp(new_vel, -v_max, v_max)
            
        return torch.cat([new_pos, new_vel], dim=1)

    def build_graph_differentiable(
        self, agent_states: torch.Tensor
    ) -> GraphsTuple:
        """
        Build a ``GraphsTuple`` where edge features are differentiable w.r.t.
        *agent_states*. The graph topology (senders/receivers) is computed
        with detached positions so that it does not create discrete gradients.

        Parameters
        ----------
        agent_states : (n_agents, 4) — may have requires_grad = True.

        Returns
        -------
        GraphsTuple whose ``.edges`` carry gradient information.
        """
        return build_graph_from_states(
            agent_states=agent_states,
            goal_states=self._goal_states,
            obstacle_positions=self._obstacle_states,
            comm_radius=self.comm_radius,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
        )

    def state_dot(
        self, agent_states: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the continuous-time derivative  ẋ = f(x) + g(x)u.

        For the double integrator:
            ẋ = [v_x, v_y, a_x/m, a_y/m]

        Parameters
        ----------
        agent_states : (n, 4)
        action       : (n, 2)

        Returns
        -------
        x_dot : (n, 4)
        """
        m = self.params["mass"]
        u_max = self.params.get("u_max")
        if u_max is not None:
            action = torch.clamp(action, -u_max, u_max)
            
        vel = agent_states[:, 2:]           # (n, 2)
        accel = action / m                  # (n, 2)
        return torch.cat([vel, accel], dim=1)  # (n, 4)
