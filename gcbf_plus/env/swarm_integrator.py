"""
Swarm environment — 4D Bounding Circle + Affine Transform + Payload swing.

Each swarm is modeled as a point mass with a dynamic bounding circle
whose radius scales with the formation scale factor s.
A payload is attached via a cable and swings under platform acceleration.

Agent State: [px, py, vx, vy]          (4D)
Scale State: [s, s_dot]                (2D per swarm)
Payload:     [γ_x, γ_y, γ̇_x, γ̇_y]    (4D, side-channel)
Control:     [a_cx, a_cy, a_s]         (3D — affine: translation + scale accel)
Dynamics: Double integrator (translation) + scale integrator (s̈ = a_s).
Safety:  Dynamic bounding circle collision (CoM distance < r_swarm_i + r_swarm_j).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..utils.graph import GraphsTuple
from ..algo.nominal_controller import NominalController
from ..utils.swarm_graph import build_swarm_graph_from_states


# ── Obstacle dataclass ───────────────────────────────────────────────

@dataclass
class Obstacle:
    center: torch.Tensor   # (2,)
    radius: float          # circle radius (m)

    def ray_intersection(self, start: torch.Tensor, end: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute the intersection of a ray (start -> end) with this circle.
        Returns the closest hit point (tensor(2,)) or None if no intersection in [0, 1].
        """
        d = end - start          # direction vector
        f = start - self.center  # vector from center to ray start

        a = (d * d).sum()
        b = 2.0 * (f * d).sum()
        c = (f * f).sum() - self.radius ** 2

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = discriminant.sqrt()
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Pick smallest t in [0, 1]
        for t in (t1, t2):
            if 0.0 <= t.item() <= 1.0:
                return start + t * d
        return None



class SwarmIntegrator:
    """
    4D bounding-circle swarm environment with affine transform control.

    The GNN outputs [Δa_cx, Δa_cy, a_s] per swarm. The affine distribution
    maps these to per-drone accelerations. The bounding circle radius
    scales dynamically with the formation scale factor s.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "R_form": 0.5,           # base formation radius (m)
        "r_margin": 0.2,         # bounding circle margin (m)
        "comm_radius": 3.0,
        "n_obs": 2,
        "obs_len_range": (0.4, 1.0),
        "mass": 0.1,
        "u_max": 1.0,
        "v_max": 1.0,
        # Scale limits
        "s_min": 0.4,            # minimum scale (drone collision prevention)
        "s_max": 1.5,            # maximum scale (wire tension limit)
        "s_dot_max": 1.0,        # maximum scale rate
        # Hierarchical velocity-command gains
        "K_pos": 0.5,            # proportional gain: goal_err → target velocity
        "K_v": 2.0,             # PD gain: velocity error → acceleration (translation)
        "K_s": 2.0,              # PD gain: velocity error → acceleration (scale)
        # Payload parameters
        "cable_length": 1.0,     # l (m)
        "gravity": 9.81,         # g (m/s^2)
        "gamma_min": 0.2,        # γ_max at s=s_min (strict swing limit)
        "gamma_max_full": 0.75,  # γ_max at s=s_max (relaxed swing limit)
        "payload_damping": 0.03, # small damping for numerical stability (1/s)
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

        # Nominal controller (PD, shared across eval and training)
        self._nominal_ctrl = NominalController(
            comm_radius=self.params["comm_radius"],
            u_max=self.params["u_max"],
            u_max_scale=self.params.get("u_max_scale", self.params["u_max"] * 0.3),
            K_s_pos=self.params.get("K_s_pos", 1.0),
            K_s=self.params.get("K_s", 2.0),
        )

        self._obstacles: List[Obstacle] = []
        self._obstacle_states: Optional[torch.Tensor] = None
        self.agent_states: Optional[torch.Tensor] = None
        self.goal_states: Optional[torch.Tensor] = None
        self.scale_states: Optional[torch.Tensor] = None   # (n, 2): [s, s_dot]
        self.payload_states: Optional[torch.Tensor] = None  # (n, 4)
        self._step_count = 0

    # ── Properties ────────────────────────────────────────────────────
    @property
    def state_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 3  # [a_cx, a_cy, a_s]

    @property
    def node_dim(self) -> int:
        # 3D one-hot + 4D payload state when payload is enabled
        return 7 if self.params.get("use_payload", False) else 3

    @property
    def edge_dim(self) -> int:
        return 4

    @property
    def comm_radius(self) -> float:
        """Base communication radius (before scale multiplication)."""
        return self.params["comm_radius"]

    @property
    def g_x_matrix(self) -> np.ndarray:
        """Control input matrix for translation only (4x2)."""
        m = self.params["mass"]
        return np.array(
            [[0, 0], [0, 0], [1/m, 0], [0, 1/m]],
            dtype=np.float32,
        )

    def r_swarm(self, s: torch.Tensor) -> torch.Tensor:
        """Dynamic bounding circle radius: R_form * s + margin."""
        R_form = self.params["R_form"]
        margin = self.params["r_margin"]
        return R_form * s + margin

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        n = self.num_agents
        area = self.area_size
        # Use r_swarm at s=1.0 for initial spacing
        r_init = self.params["R_form"] * 1.0 + self.params["r_margin"]
        margin = r_init + 0.1
        n_obs = self.params["n_obs"]

        # Obstacles (circular)
        self._obstacles = []
        obs_lo, obs_hi = self.params.get("obs_len_range", (0.4, 1.0))
        for _ in range(n_obs):
            cx = rng.uniform(margin, area - margin)
            cy = rng.uniform(margin, area - margin)
            r = float(rng.uniform(obs_lo, obs_hi) / 2.0)
            self._obstacles.append(Obstacle(
                center=torch.tensor([cx, cy], dtype=torch.float32),
                radius=r,
            ))

        obs_states = []
        for obs in self._obstacles:
            s = torch.zeros(4)
            s[:2] = obs.center
            obs_states.append(s)
        self._obstacle_states = torch.stack(obs_states) if obs_states else None
        
        # LiDAR cache
        self._last_lidar_hits: List[torch.Tensor] = [] # List of (N_hits, 4) per agent

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
        # Scale state: [s, s_dot] initialized to [1.0, 0.0]
        self.scale_states = torch.tensor(
            [[1.0, 0.0]] * n, dtype=torch.float32,
        )
        self.payload_states = torch.zeros(n, 4, dtype=torch.float32)
        self._step_count = 0

    def _sample_free_positions(self, rng, count, margin):
        """Sample positions that avoid obstacles AND are >= 2*r_swarm(s=1) apart."""
        area = self.area_size
        r_init = self.params["R_form"] * 1.0 + self.params["r_margin"]
        min_dist = 2 * r_init
        positions = []
        for _ in range(100_000):
            if len(positions) >= count:
                break
            p = rng.uniform(margin, area - margin, size=2).astype(np.float32)
            ok = True
            for obs in self._obstacles:
                oc = obs.center.numpy()
                if np.linalg.norm(p - oc) < obs.radius + margin:
                    ok = False
                    break
            if not ok:
                continue
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
        self.scale_states = self.scale_states.to(device)
        if self._obstacle_states is not None:
            self._obstacle_states = self._obstacle_states.to(device)
        return self

    # ── Step ──────────────────────────────────────────────────────────
    def step(self, action: torch.Tensor):
        """
        action: (n, 3) — [a_cx, a_cy, a_s] per swarm (affine parameters).

        Applies affine distribution to compute per-swarm CoM acceleration,
        integrates scale dynamics, and updates payload swing.
        """
        dt = self.dt
        u_max = self.params.get("u_max")
        s_dot_max = self.params.get("s_dot_max", 1.0)

        # Split affine parameters
        a_cx = action[:, 0]  # (n,)
        a_cy = action[:, 1]  # (n,)
        a_s = action[:, 2]   # (n,)

        # Clamp translational acceleration (norm-based to preserve direction)
        a_trans = torch.stack([a_cx, a_cy], dim=-1)  # (n, 2)
        if u_max is not None:
            a_norm = a_trans.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            a_trans = a_trans * (a_norm.clamp(max=u_max) / a_norm)

        # ── Translation dynamics (CoM) ──
        accel = a_trans  # action is already acceleration (m/s²), consistent with vectorized_swarm
        new_pos = self.agent_states[:, :2] + self.agent_states[:, 2:4] * dt + 0.5 * accel * dt**2
        new_vel = self.agent_states[:, 2:4] + accel * dt
        self.agent_states = torch.cat([new_pos, new_vel], dim=-1)

        # ── Scale dynamics ──
        s = self.scale_states[:, 0]      # (n,)
        s_dot = self.scale_states[:, 1]  # (n,)
        new_s_dot = s_dot + a_s * dt
        new_s_dot = torch.clamp(new_s_dot, -s_dot_max, s_dot_max)
        new_s = s + new_s_dot * dt
        # Hard clamp scale to physical limits
        s_min = self.params["s_min"]
        s_max = self.params["s_max"]
        new_s = torch.clamp(new_s, s_min, s_max)
        # Zero out velocity when hitting limits
        new_s_dot = torch.where(
            (new_s <= s_min) & (new_s_dot < 0), torch.zeros_like(new_s_dot), new_s_dot
        )
        new_s_dot = torch.where(
            (new_s >= s_max) & (new_s_dot > 0), torch.zeros_like(new_s_dot), new_s_dot
        )
        self.scale_states = torch.stack([new_s, new_s_dot], dim=-1)

        # ── Payload swing dynamics (Semi-implicit / Symplectic Euler) ──
        cable_length = self.params["cable_length"]
        R_form = self.params["R_form"]
        s_cur = self.scale_states[:, 0]  # (n,)
        l = torch.sqrt(torch.clamp(cable_length**2 - (R_form * s_cur)**2, min=1e-4))  # l_eff(s)
        g = self.params["gravity"]
        c = self.params["payload_damping"]
        ps = self.payload_states
        gx     = ps[:, 0]
        gy     = ps[:, 1]
        gx_dot = ps[:, 2]
        gy_dot = ps[:, 3]

        ax_plat = accel[:, 0]
        ay_plat = accel[:, 1]

        gx_ddot = -(g / l) * torch.sin(gx) - (ax_plat / l) * torch.cos(gx) - c * gx_dot
        gy_ddot = -(g / l) * torch.sin(gy) - (ay_plat / l) * torch.cos(gy) - c * gy_dot

        new_gx_dot = gx_dot + gx_ddot * dt
        new_gy_dot = gy_dot + gy_ddot * dt
        new_gx     = gx + new_gx_dot * dt
        new_gy     = gy + new_gy_dot * dt
        self.payload_states = torch.stack([new_gx, new_gy, new_gx_dot, new_gy_dot], dim=-1)

        self._step_count += 1
        collision = self.unsafe_mask().any().item()
        done = self._step_count >= self.max_steps
        return self.agent_states, {"done": done, "collision": collision}

    # ── Nominal controller ────────────────────────────────────────────────
    def nominal_controller(self) -> torch.Tensor:
        """Returns (n, 3): [a_cx_nom, a_cy_nom, a_s_nom] via LQR nominal controller."""
        return self._nominal_ctrl(
            self.agent_states,
            self.goal_states,
            self.scale_states,
        )

    # ── Unsafe mask (dynamic bounding circle) ─────────────────────────
    def unsafe_mask(self) -> torch.Tensor:
        n = self.num_agents
        device = self.agent_states.device

        pos = self.agent_states[:, :2]
        s = self.scale_states[:, 0]  # (n,)
        r = self.r_swarm(s)          # (n,) — per-agent dynamic radius

        agent_collision = torch.zeros(n, dtype=torch.bool, device=device)
        if n > 1:
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (n, n, 2)
            dist = torch.norm(diff, dim=-1)                # (n, n)
            dist = dist + torch.eye(n, device=device) * 1e6
            # Dynamic collision threshold: r_i + r_j
            thresh = r.unsqueeze(0) + r.unsqueeze(1)      # (n, n)
            collision_matrix = dist < thresh
            agent_collision = collision_matrix.any(dim=1)

        # Obstacle collision (circle: dist < r_swarm + r_obs)
        obs_collision = torch.zeros(n, dtype=torch.bool, device=device)
        for obs in self._obstacles:
            c = obs.center.to(device)
            dist = torch.norm(pos - c, dim=-1)  # (n,)
            inside = dist < (r + obs.radius)
            obs_collision = obs_collision | inside

        return agent_collision | obs_collision

    def get_lidar_points(self, num_beams: int = 32) -> List[torch.Tensor]:
        """
        Generate LiDAR hit points for each agent.
        Returns a list of length num_agents, where each element is a (M_i, 4) tensor
        of [px, py, vx, vy] hit points (velocity is always 0 for obstacles).
        """
        n = self.num_agents
        s_vals = self.scale_states[:, 0]
        R_form = self.params.get("R_form", 0.5)
        sensing_radii = R_form * s_vals + (self.comm_radius - R_form)
        agent_pos = self.agent_states[:, :2]

        all_agent_hits = []
        
        # Angles for radial beams
        angles = torch.linspace(0, 2 * math.pi, num_beams + 1, device=agent_pos.device)[:-1]
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) # (N_beams, 2)
        
        for i in range(n):
            start = agent_pos[i]
            radius = sensing_radii[i]
            agent_hits = []
            
            # Ends of the beams
            beam_ends = start + radius * directions # (N_beams, 2)
            
            for b_idx in range(num_beams):
                end = beam_ends[b_idx]
                min_t = 1.1 # initialized to > 1
                hit_pos = None
                
                for obs in self._obstacles:
                    # Intersection check with rectangle boundary
                    p_hit = obs.ray_intersection(start, end)
                    if p_hit is not None:
                        t = torch.norm(p_hit - start) / radius
                        if t < min_t:
                            min_t = t
                            hit_pos = p_hit
                
                if hit_pos is not None:
                    # hit_pos is (2,)
                    hit_state = torch.zeros(4, device=agent_pos.device)
                    hit_state[:2] = hit_pos
                    # vx, vy = 0
                    agent_hits.append(hit_state)
            
            if agent_hits:
                all_agent_hits.append(torch.stack(agent_hits))
            else:
                all_agent_hits.append(torch.zeros((0, 4), device=agent_pos.device))
                
        self._last_lidar_hits = all_agent_hits
        return all_agent_hits

    def get_lidar_hits(self, num_beams: int = 32) -> torch.Tensor:
        """
        Fixed-size LiDAR hit points for each agent.
        Returns: (num_agents, num_beams, 4) tensor.
        Non-hitting beams are filled with 1e6 (far away, ignored in QP).
        Matches VectorizedSwarmEnv.get_lidar_hits() interface.
        """
        n = self.num_agents
        device = self.agent_states.device
        s_vals = self.scale_states[:, 0]
        R_form = self.params.get("R_form", 0.5)
        sensing_radii = R_form * s_vals + (self.comm_radius - R_form)
        agent_pos = self.agent_states[:, :2]

        angles = torch.linspace(0, 2 * math.pi, num_beams + 1, device=device)[:-1]
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (nb, 2)

        result = torch.full((n, num_beams, 4), 1e6, device=device)

        for i in range(n):
            start = agent_pos[i]
            radius = sensing_radii[i]
            beam_ends = start + radius * directions  # (nb, 2)

            for b_idx in range(num_beams):
                end = beam_ends[b_idx]
                min_t = 1.1
                hit_pos = None

                for obs in self._obstacles:
                    p_hit = obs.ray_intersection(start, end)
                    if p_hit is not None:
                        t = torch.norm(p_hit - start) / radius
                        if t < min_t:
                            min_t = t
                            hit_pos = p_hit

                if hit_pos is not None:
                    result[i, b_idx, :2] = hit_pos
                    result[i, b_idx, 2:] = 0.0  # vx, vy = 0

        return result

    # ── Graph builder ─────────────────────────────────────────────────
    def _get_graph(self) -> GraphsTuple:
        s = self.scale_states[:, 0]
        R_form = self.params.get("R_form", 0.5)
        dyn_cr = R_form * s + (self.comm_radius - R_form)

        obs = self._obstacle_states  # (n_obs, 4) or None

        use_payload = self.params.get("use_payload", False)
        return build_swarm_graph_from_states(
            agent_states=self.agent_states,
            goal_states=self.goal_states,
            obstacle_positions=obs,
            comm_radius=dyn_cr,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            payload_states=self.payload_states if use_payload else None,
        )
