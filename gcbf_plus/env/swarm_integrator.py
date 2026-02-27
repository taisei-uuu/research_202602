"""
Swarm Integrator multi-agent environment.

Each "agent" is a rigid-body swarm of 3 drones forming an equilateral
triangle around the Center of Mass (CoM).

State  x = [p_x, p_y, v_x, v_y, θ, ω]   (dim 6)
Action u = [a_x, a_y, α]                   (dim 3)

Dynamics (continuous-time double integrator for translation + rotation):
    ṗ = v
    v̇ = a / m
    θ̇ = ω
    ω̇ = α / I

Euler integration:
    p_{t+1} = p_t + v_t · dt + 0.5 · (a/m) · dt²
    v_{t+1} = v_t + (a/m) · dt
    θ_{t+1} = θ_t + ω_t · dt + 0.5 · (α/I) · dt²
    ω_{t+1} = ω_t + (α/I) · dt

Collision detection uses the full 3×3 = 9 pairwise drone distances
between swarms, NOT the CoM distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..utils.graph import GraphsTuple
from ..utils.swarm_graph import (
    build_swarm_graph_from_states,
    get_equilateral_offsets,
    _compute_drone_positions,
    _wrap_angle,
)


# ---------------------------------------------------------------------------
# LQR helper  (discrete-time, infinite-horizon, for the translational part)
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
# Obstacle data  (simple rectangle — reused from double_integrator)
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

class SwarmIntegrator:
    """
    2-D multi-swarm system with rigid body dynamics.

    Each agent is a swarm of 3 drones in equilateral triangle formation.

    Parameters
    ----------
    num_agents : int
        Number of swarms.
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
        "drone_radius": 0.05,       # individual drone radius
        "R_form": 0.3,              # formation circumradius
        "comm_radius": 2.0,         # sensing / communication radius R
        "n_rays": 32,               # (reserved for LiDAR extension)
        "obs_len_range": (0.1, 0.3),
        "n_obs": 8,                 # number of random obstacles
        "mass": 0.1,                # total swarm mass
        "inertia": 0.01,            # rotational inertia I
        "u_max": 0.3,               # max translational force
        "alpha_max": 0.1,           # max angular torque
        "v_max": 1.0,               # max velocity per axis
        "omega_max": 2.0,           # max angular velocity
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

        # Build LQR gain for the translational subsystem (4D: px,py,vx,vy)
        m = self.params["mass"]
        A_ct_t = np.zeros((4, 4), dtype=np.float32)
        A_ct_t[0, 2] = 1.0
        A_ct_t[1, 3] = 1.0
        A_t = A_ct_t * dt + np.eye(4, dtype=np.float32)

        B_t = np.array(
            [[0.0, 0.0],
             [0.0, 0.0],
             [1.0 / m, 0.0],
             [0.0, 1.0 / m]],
            dtype=np.float32,
        ) * dt

        Q_t = np.eye(4, dtype=np.float32) * 5.0
        R_t = np.eye(2, dtype=np.float32)
        self._K_trans = torch.tensor(
            _dlqr(A_t, B_t, Q_t, R_t), dtype=torch.float32
        )  # (2, 4)

        # PD gains for the rotational subsystem (θ, ω)
        self._Kp_theta = 2.0   # proportional gain for θ → α
        self._Kd_theta = 1.0   # derivative gain for ω → α

        # Internal state
        self._step_count: int = 0
        self._agent_states: Optional[torch.Tensor] = None   # (n, 6)
        self._goal_states: Optional[torch.Tensor] = None     # (n, 6)
        self._obstacles: list[Obstacle] = []
        self._obstacle_states: Optional[torch.Tensor] = None  # (n_obs, 6)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        """Per-agent state dimension: [px, py, vx, vy, θ, ω]."""
        return 6

    @property
    def action_dim(self) -> int:
        """Per-agent action dimension: [ax, ay, α]."""
        return 3

    @property
    def node_dim(self) -> int:
        """One-hot node-type indicator dimension."""
        return 3  # agent / goal / obstacle

    @property
    def edge_dim(self) -> int:
        """Edge feature dimension: [Δpx, Δpy, Δvx, Δvy, Δθ, min_dist, closest_dx, closest_dy]."""
        return 8

    @property
    def comm_radius(self) -> float:
        return self.params["comm_radius"]

    @property
    def g_x_matrix(self) -> np.ndarray:
        """
        Continuous-time control input matrix g(x).

        Maps u = [a_x, a_y, α] into 6D state derivative.
        """
        m = self.params.get("mass", 0.1)
        I = self.params.get("inertia", 0.01)
        return np.array(
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [1.0 / m, 0.0, 0.0],
             [0.0, 1.0 / m, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0 / I]],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Swarm geometry
    # ------------------------------------------------------------------
    def get_drone_positions(
        self, agent_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute global positions of all drones in all swarms.

        Parameters
        ----------
        agent_states : (n_agents, 6) or None — uses internal state if None.

        Returns
        -------
        drone_pos : (n_agents, 3, 2)
        """
        if agent_states is None:
            agent_states = self._agent_states
        device = agent_states.device
        offsets = get_equilateral_offsets(self.params["R_form"], device)
        return _compute_drone_positions(
            agent_states[:, :2], agent_states[:, 4], offsets
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> GraphsTuple:
        """
        Reset the environment and return the initial graph.

        Randomly places swarm CoMs, goals, and rectangular obstacles.
        """
        rng = np.random.default_rng(seed)
        self._step_count = 0

        n = self.num_agents
        area = self.area_size
        R_form = self.params["R_form"]
        drone_r = self.params["drone_radius"]
        # Buffer: CoM must be at least R_form + drone_r from boundary
        margin = R_form + drone_r
        n_obs = self.params["n_obs"]
        obs_lo, obs_hi = self.params["obs_len_range"]

        # ---- Generate obstacles ----
        self._obstacles = []
        obs_positions = []
        for _ in range(n_obs):
            cx, cy = rng.uniform(margin, area - margin, size=2)
            hw = rng.uniform(obs_lo, obs_hi) / 2.0
            hh = rng.uniform(obs_lo, obs_hi) / 2.0
            obs = Obstacle(
                center=torch.tensor([cx, cy], dtype=torch.float32),
                half_size=torch.tensor([hw, hh], dtype=torch.float32),
            )
            self._obstacles.append(obs)
            obs_positions.append([cx, cy, 0.0, 0.0, 0.0, 0.0])

        if n_obs > 0:
            self._obstacle_states = torch.tensor(obs_positions, dtype=torch.float32)
        else:
            self._obstacle_states = None

        # ---- Generate agent and goal positions (collision-free) ----
        def _sample_free_positions(count: int) -> np.ndarray:
            """Sample *count* CoM positions that don't overlap obstacles."""
            positions = np.empty((0, 2))
            while positions.shape[0] < count:
                candidates = rng.uniform(margin, area - margin, size=(count * 4, 2))
                free = np.ones(candidates.shape[0], dtype=bool)
                for obs in self._obstacles:
                    c = obs.center.numpy()
                    hs = obs.half_size.numpy() + margin
                    inside = (np.abs(candidates[:, 0] - c[0]) < hs[0]) & (
                        np.abs(candidates[:, 1] - c[1]) < hs[1]
                    )
                    free &= ~inside
                candidates = candidates[free]
                positions = np.concatenate([positions, candidates], axis=0)
            return positions[:count]

        start_pos = _sample_free_positions(n)
        goal_pos = _sample_free_positions(n)

        # Random initial orientations
        start_theta = rng.uniform(-math.pi, math.pi, size=(n, 1))

        self._agent_states = torch.tensor(
            np.concatenate([
                start_pos,
                np.zeros((n, 2)),     # zero velocity
                start_theta,
                np.zeros((n, 1)),     # zero angular velocity
            ], axis=1),
            dtype=torch.float32,
        )
        self._goal_states = torch.tensor(
            np.concatenate([
                goal_pos,
                np.zeros((n, 2)),     # zero velocity at goal
                np.zeros((n, 1)),     # target θ = 0
                np.zeros((n, 1)),     # target ω = 0
            ], axis=1),
            dtype=torch.float32,
        )

        return self._get_graph()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> Tuple[GraphsTuple, Dict[str, Any]]:
        """
        Apply *action* (n_agents, 3) and advance one time-step.

        Returns
        -------
        graph : GraphsTuple — the new observation.
        info  : dict        — contains ``done``, ``step``.
        """
        assert action.shape == (self.num_agents, self.action_dim)

        u_max = self.params.get("u_max")
        alpha_max = self.params.get("alpha_max")

        # Clamp translational and angular separately
        if u_max is not None:
            action = torch.cat([
                torch.clamp(action[:, :2], -u_max, u_max),
                action[:, 2:3],
            ], dim=1)
        if alpha_max is not None:
            action = torch.cat([
                action[:, :2],
                torch.clamp(action[:, 2:3], -alpha_max, alpha_max),
            ], dim=1)

        x = self._agent_states
        m = self.params["mass"]
        I = self.params["inertia"]
        dt = self.dt

        # Translation
        accel = action[:, :2] / m
        new_pos = x[:, :2] + x[:, 2:4] * dt + 0.5 * accel * dt ** 2
        new_vel = x[:, 2:4] + accel * dt

        # Rotation
        angular_accel = action[:, 2:3] / I
        new_theta = x[:, 4:5] + x[:, 5:6] * dt + 0.5 * angular_accel * dt ** 2
        new_omega = x[:, 5:6] + angular_accel * dt

        # Clamp velocities
        v_max = self.params.get("v_max")
        omega_max = self.params.get("omega_max")
        if v_max is not None:
            new_vel = torch.clamp(new_vel, -v_max, v_max)
        if omega_max is not None:
            new_omega = torch.clamp(new_omega, -omega_max, omega_max)

        # Wrap theta to [-π, π]
        new_theta = _wrap_angle(new_theta)

        self._agent_states = torch.cat([new_pos, new_vel, new_theta, new_omega], dim=1)
        self._step_count += 1

        done = self._step_count >= self.max_steps
        info = {"done": done, "step": self._step_count}

        return self._get_graph(), info

    # ------------------------------------------------------------------
    # Nominal controller (PD)
    # ------------------------------------------------------------------
    def nominal_controller(self) -> torch.Tensor:
        """
        Compute the nominal control u_nom = [a_x, a_y, α].

        Translation: LQR on [px,py,vx,vy] subsystem → (a_x, a_y).
        Rotation: PD to drive θ → θ_goal and ω → 0.

        Returns
        -------
        u_nom : (n_agents, 3)
        """
        x = self._agent_states  # (n, 6)
        g = self._goal_states   # (n, 6)

        # Translational LQR
        error_trans = x[:, :4] - g[:, :4]  # (n, 4)
        u_trans = -error_trans @ self._K_trans.T  # (n, 2)

        # Angular PD
        theta_err = _wrap_angle(x[:, 4] - g[:, 4])  # (n,)
        omega = x[:, 5]                               # (n,)
        u_alpha = -(self._Kp_theta * theta_err + self._Kd_theta * omega)  # (n,)

        u_nom = torch.cat([u_trans, u_alpha.unsqueeze(-1)], dim=1)  # (n, 3)

        # Clamp
        u_max = self.params.get("u_max")
        alpha_max = self.params.get("alpha_max")
        if u_max is not None:
            u_nom = torch.cat([
                torch.clamp(u_nom[:, :2], -u_max, u_max),
                u_nom[:, 2:3],
            ], dim=1)
        if alpha_max is not None:
            u_nom = torch.cat([
                u_nom[:, :2],
                torch.clamp(u_nom[:, 2:3], -alpha_max, alpha_max),
            ], dim=1)

        return u_nom

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _get_graph(self) -> GraphsTuple:
        """Build the ``GraphsTuple`` from the current state."""
        return build_swarm_graph_from_states(
            agent_states=self._agent_states,
            goal_states=self._goal_states,
            obstacle_positions=self._obstacle_states,
            comm_radius=self.comm_radius,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            R_form=self.params["R_form"],
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @property
    def agent_states(self) -> torch.Tensor:
        """Current agent states (n_agents, 6)."""
        return self._agent_states

    @property
    def goal_states(self) -> torch.Tensor:
        """Current goal states (n_agents, 6)."""
        return self._goal_states

    def to(self, device: torch.device) -> "SwarmIntegrator":
        """
        Move all internal tensors to *device* (e.g. ``torch.device('cuda')``).

        Returns ``self`` for chaining.
        """
        self._K_trans = self._K_trans.to(device)
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
        True for each swarm that is in collision.

        Collision is detected using 3×3=9 pairwise distances between
        constituent drones.  If ANY distance < 2*drone_radius, both
        swarms are marked unsafe.

        Returns : (n_agents,) bool tensor.
        """
        n = self.num_agents
        drone_r = self.params["drone_radius"]
        device = self._agent_states.device

        # Get all drone positions: (n, 3, 2)
        drone_pos = self.get_drone_positions()

        # Agent–agent collision via drone-to-drone distances
        agent_collision = torch.zeros(n, dtype=torch.bool, device=device)

        if n > 1:
            # (n, 1, 3, 1, 2) - (1, n, 1, 3, 2) → (n, n, 3, 3, 2)
            diff = (
                drone_pos[:, None, :, None, :]
                - drone_pos[None, :, None, :, :]
            )
            dist = torch.norm(diff, dim=-1)  # (n, n, 3, 3)

            # Min over the 9 pairs → (n, n)
            dist_min = dist.reshape(n, n, -1).min(dim=-1).values

            # Ignore self (set diagonal to large value)
            dist_min = dist_min + torch.eye(n, device=device) * 1e6

            # Collision if any pair < 2 * drone_radius
            collision_matrix = dist_min < 2 * drone_r
            agent_collision = collision_matrix.any(dim=1)

        # Agent–obstacle collision (any drone inside obstacle)
        obs_collision = torch.zeros(n, dtype=torch.bool, device=device)
        for obs in self._obstacles:
            padded_hs = obs.half_size + drone_r
            # Check each drone: drone_pos (n, 3, 2)
            for k in range(3):
                diff = torch.abs(drone_pos[:, k, :] - obs.center.unsqueeze(0))
                inside = (diff[:, 0] < padded_hs[0]) & (diff[:, 1] < padded_hs[1])
                obs_collision = obs_collision | inside

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
        agent_states : (n_agents, 6)  — may have requires_grad=True
        action       : (n_agents, 3)

        Returns
        -------
        next_states : (n_agents, 6)
        """
        m = self.params["mass"]
        I = self.params["inertia"]
        dt = self.dt

        u_max = self.params.get("u_max")
        alpha_max = self.params.get("alpha_max")
        if u_max is not None:
            action = torch.cat([
                torch.clamp(action[:, :2], -u_max, u_max),
                action[:, 2:3],
            ], dim=1)
        if alpha_max is not None:
            action = torch.cat([
                action[:, :2],
                torch.clamp(action[:, 2:3], -alpha_max, alpha_max),
            ], dim=1)

        # Translation
        accel = action[:, :2] / m
        new_pos = agent_states[:, :2] + agent_states[:, 2:4] * dt + 0.5 * accel * dt ** 2
        new_vel = agent_states[:, 2:4] + accel * dt

        # Rotation
        angular_accel = action[:, 2:3] / I
        new_theta = agent_states[:, 4:5] + agent_states[:, 5:6] * dt + 0.5 * angular_accel * dt ** 2
        new_omega = agent_states[:, 5:6] + angular_accel * dt

        # Clamp velocities
        v_max = self.params.get("v_max")
        omega_max = self.params.get("omega_max")
        if v_max is not None:
            new_vel = torch.clamp(new_vel, -v_max, v_max)
        if omega_max is not None:
            new_omega = torch.clamp(new_omega, -omega_max, omega_max)

        return torch.cat([new_pos, new_vel, new_theta, new_omega], dim=1)

    def build_graph_differentiable(
        self, agent_states: torch.Tensor
    ) -> GraphsTuple:
        """
        Build a ``GraphsTuple`` where edge features are differentiable w.r.t.
        *agent_states*.

        Parameters
        ----------
        agent_states : (n_agents, 6) — may have requires_grad = True.

        Returns
        -------
        GraphsTuple whose ``.edges`` carry gradient information.
        """
        return build_swarm_graph_from_states(
            agent_states=agent_states,
            goal_states=self._goal_states,
            obstacle_positions=self._obstacle_states,
            comm_radius=self.comm_radius,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            R_form=self.params["R_form"],
        )

    def state_dot(
        self, agent_states: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the continuous-time derivative  ẋ = f(x) + g(x)u.

        For the swarm integrator:
            ẋ = [v_x, v_y, a_x/m, a_y/m, ω, α/I]

        Parameters
        ----------
        agent_states : (n, 6)
        action       : (n, 3)

        Returns
        -------
        x_dot : (n, 6)
        """
        m = self.params["mass"]
        I = self.params["inertia"]

        u_max = self.params.get("u_max")
        alpha_max = self.params.get("alpha_max")
        if u_max is not None:
            action = torch.cat([
                torch.clamp(action[:, :2], -u_max, u_max),
                action[:, 2:3],
            ], dim=1)
        if alpha_max is not None:
            action = torch.cat([
                action[:, :2],
                torch.clamp(action[:, 2:3], -alpha_max, alpha_max),
            ], dim=1)

        vel = agent_states[:, 2:4]             # (n, 2)
        accel = action[:, :2] / m              # (n, 2)
        omega = agent_states[:, 5:6]           # (n, 1)
        angular_accel = action[:, 2:3] / I     # (n, 1)
        return torch.cat([vel, accel, omega, angular_accel], dim=1)  # (n, 6)
