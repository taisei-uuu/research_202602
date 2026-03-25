"""
Vectorized (batched) swarm environment — 4D Bounding Circle + Affine Transform + Payload.

All B environments run in parallel on GPU using (B, n_agents, 4) state
tensors.  Dynamic bounding circle safety with scale factor s.

Agent State: [px, py, vx, vy]          (4D)
Scale State: [s, s_dot]                (2D per swarm)
Payload:     [γ_x, γ_y, γ̇_x, γ̇_y]    (4D, side-channel — not in GNN)
Control:     [a_cx, a_cy, a_s]         (3D — affine)
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
    Vectorized swarm env — 4D point mass with affine transform control.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "R_form": 0.5,           # base formation radius (m)
        "r_margin": 0.2,         # bounding circle margin (m)
        "comm_radius": 3.0,
        "n_obs": 2,
        "obs_len_range": (0.4, 1.0),
        "mass": 0.1,
        "u_max": 0.3,
        "v_max": 1.0,
        # Scale limits
        "s_min": 0.4,
        "s_max": 1.5,
        "s_dot_max": 1.0,
        # Hierarchical velocity-command gains
        "K_pos": 0.5,          # proportional gain: goal_err → target velocity
        "K_v": 2.0,            # PD gain: velocity error → acceleration (translation)
        "K_s": 2.0,            # PD gain: velocity error → acceleration (scale)
        # Payload parameters
        "cable_length": 1.0,
        "gravity": 9.81,
        "gamma_min": 0.2,        # γ_max at s=s_min (strict)
        "gamma_max_full": 0.75,  # γ_max at s=s_max (relaxed)
        "payload_damping": 0.03,
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

        # LQR gain (4D → 2D, translation only)
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
        self._agent_states: Optional[torch.Tensor] = None   # (B, n, 4)
        self._goal_states: Optional[torch.Tensor] = None    # (B, n, 4)
        self._scale_states: Optional[torch.Tensor] = None   # (B, n, 2): [s, s_dot]
        self._obstacle_centers: Optional[torch.Tensor] = None
        self._obstacle_half_sizes: Optional[torch.Tensor] = None
        self._obstacle_states: Optional[torch.Tensor] = None
        self._payload_states: Optional[torch.Tensor] = None  # (B, n, 4)
        self._last_lidar_hits: Optional[torch.Tensor] = None # (B, n, n_beams, 4) [px, py, vx, vy]
        self._step_counts: Optional[torch.Tensor] = None    # (B,) step count per batch

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 3  # [a_cx, a_cy, a_s]

    @property
    def node_dim(self) -> int:
        # 3D one-hot + 4D payload state when payload is enabled
        return 7 if self.params.get("use_payload", True) else 3

    @property
    def edge_dim(self) -> int:
        return 4

    @property
    def comm_radius(self) -> float:
        """Base communication radius (before scale multiplication)."""
        return self.params["comm_radius"]

    def dynamic_comm_radius(self) -> torch.Tensor:
        """Dynamic comm_radius: comm_base * s per agent.  Returns (B, n)."""
        s = self._scale_states[:, :, 0]  # (B, n)
        return self.comm_radius * s

    @property
    def n_obs(self) -> int:
        return self.params["n_obs"]

    @property
    def g_x_matrix(self) -> np.ndarray:
        """Control input matrix for translation only (4x2)."""
        m = self.params["mass"]
        return np.array([[0, 0], [0, 0], [1/m, 0], [0, 1/m]], dtype=np.float32)

    @property
    def payload_states(self) -> torch.Tensor:
        """(B, n, 4): [γ_x, γ_y, γ̇_x, γ̇_y]."""
        return self._payload_states

    @property
    def scale_states(self) -> torch.Tensor:
        """(B, n, 2): [s, s_dot]."""
        return self._scale_states

    def r_swarm(self, s: torch.Tensor) -> torch.Tensor:
        """Dynamic bounding circle radius: R_form * s + margin."""
        R_form = self.params["R_form"]
        margin = self.params["r_margin"]
        return R_form * s + margin

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, device: torch.device, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        B = self.batch_size
        n = self.num_agents
        area = self.area_size
        r_init = self.params["R_form"] * 1.0 + self.params["r_margin"]
        margin = r_init + 0.1
        n_obs = self.params["n_obs"]
        obs_lo, obs_hi = self.params.get("obs_len_range", (0.4, 1.0))
        self._step_count = 0

        # Obstacles (circular)
        if n_obs > 0:
            obs_cx = rng.uniform(margin, area - margin, size=(B, n_obs))
            obs_cy = rng.uniform(margin, area - margin, size=(B, n_obs))
            obs_r  = rng.uniform(obs_lo, obs_hi, size=(B, n_obs)) / 2.0
            self._obstacle_centers = torch.tensor(
                np.stack([obs_cx, obs_cy], axis=-1), dtype=torch.float32, device=device
            )
            self._obstacle_radii = torch.tensor(obs_r, dtype=torch.float32, device=device)
            obs_4d = np.zeros((B, n_obs, 4), dtype=np.float32)
            obs_4d[:, :, 0] = obs_cx
            obs_4d[:, :, 1] = obs_cy
            obs_4d[:, :, 2] = obs_r
            self._obstacle_states = torch.tensor(obs_4d, dtype=torch.float32, device=device)
        else:
            self._obstacle_centers = None
            self._obstacle_radii = None
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
        # Scale states: [s=1.0, s_dot=0.0] for all agents in all batches
        self._scale_states = torch.zeros(B, n, 2, dtype=torch.float32, device=device)
        self._scale_states[:, :, 0] = 1.0  # s = 1.0
        self._payload_states = torch.zeros(B, n, 4, dtype=torch.float32, device=device)
        self._step_counts = torch.zeros(B, dtype=torch.long, device=device)
        self._K = self._K.to(device)

    def _sample_free_pos(self, rng, count, margin, batch_idx):
        """Sample positions avoiding obstacles AND >= 2*r_swarm(s=1) apart."""
        area = self.area_size
        r_init = self.params["R_form"] * 1.0 + self.params["r_margin"]
        min_dist = 2 * r_init
        positions = []
        for _ in range(100_000):
            if len(positions) >= count:
                break
            p = rng.uniform(margin, area - margin, size=2).astype(np.float32)
            ok = True
            if self._obstacle_centers is not None:
                oc = self._obstacle_centers[batch_idx].cpu().numpy()
                or_ = self._obstacle_radii[batch_idx].cpu().numpy()
                for j in range(oc.shape[0]):
                    if np.linalg.norm(p - oc[j]) < or_[j] + margin:
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

    def reset_at_indices(self, indices: torch.Tensor, seed: Optional[int] = None):
        """Reset only specific batches (indices) within the vectorized environment."""
        if len(indices) == 0:
            return
        
        device = self._agent_states.device
        B_sub = len(indices)
        n = self.num_agents
        area = self.area_size
        r_init = self.params["R_form"] * 1.0 + self.params["r_margin"]
        margin = r_init + 0.1
        n_obs = self.params["n_obs"]
        obs_lo, obs_hi = self.params.get("obs_len_range", (0.4, 1.0))
        
        # Use a new RNG for these sub-batches
        rng = np.random.default_rng(seed)

        # 1. Update obstacles for these indices
        if n_obs > 0:
            sub_cx = rng.uniform(margin, area - margin, size=(B_sub, n_obs))
            sub_cy = rng.uniform(margin, area - margin, size=(B_sub, n_obs))
            sub_r  = rng.uniform(obs_lo, obs_hi, size=(B_sub, n_obs)) / 2.0

            self._obstacle_centers[indices] = torch.tensor(
                np.stack([sub_cx, sub_cy], axis=-1), dtype=torch.float32, device=device
            )
            self._obstacle_radii[indices] = torch.tensor(sub_r, dtype=torch.float32, device=device)
            obs_4d = np.zeros((B_sub, n_obs, 4), dtype=np.float32)
            obs_4d[:, :, 0] = sub_cx
            obs_4d[:, :, 1] = sub_cy
            obs_4d[:, :, 2] = sub_r
            self._obstacle_states[indices] = torch.tensor(obs_4d, dtype=torch.float32, device=device)

        # 2. Update agent and goal positions for these indices
        sub_start = np.empty((B_sub, n, 2), dtype=np.float32)
        sub_goal = np.empty((B_sub, n, 2), dtype=np.float32)
        for i, b_idx in enumerate(indices.tolist()):
            sub_start[i] = self._sample_free_pos(rng, n, margin, b_idx)
            sub_goal[i] = self._sample_free_pos(rng, n, margin, b_idx)

        agent_4d = np.concatenate([sub_start, np.zeros((B_sub, n, 2), dtype=np.float32)], axis=-1)
        goal_4d = np.concatenate([sub_goal, np.zeros((B_sub, n, 2), dtype=np.float32)], axis=-1)

        self._agent_states[indices] = torch.tensor(agent_4d, dtype=torch.float32, device=device)
        self._goal_states[indices] = torch.tensor(goal_4d, dtype=torch.float32, device=device)
        
        # 3. Reset scale and payload states for these indices
        self._scale_states[indices] = torch.zeros(B_sub, n, 2, dtype=torch.float32, device=device)
        self._scale_states[indices, :, 0] = 1.0  # s = 1.0
        self._payload_states[indices] = torch.zeros(B_sub, n, 4, dtype=torch.float32, device=device)
        
        # 4. Reset step counts for these indices
        self._step_counts[indices] = 0

    # ── Step ──────────────────────────────────────────────────────────
    def step(self, action: torch.Tensor):
        """action: (B, n, 3) — [a_cx, a_cy, a_s] per swarm."""
        m = self.params["mass"]
        dt = self.dt
        u_max = self.params.get("u_max")
        v_max = self.params.get("v_max")
        s_dot_max = self.params.get("s_dot_max", 1.0)

        # Split affine parameters
        a_trans = action[:, :, :2]   # (B, n, 2) translation
        a_s = action[:, :, 2]        # (B, n)    scale acceleration

        if u_max is not None:
            a_trans = torch.clamp(a_trans, -u_max, u_max)

        # ── Translation dynamics (CoM) ──
        accel = a_trans
        x = self._agent_states
        new_pos = x[:, :, :2] + x[:, :, 2:4] * dt + 0.5 * accel * dt**2
        new_vel = x[:, :, 2:4] + accel * dt
        if v_max is not None:
            new_vel = torch.clamp(new_vel, -v_max, v_max)
        self._agent_states = torch.cat([new_pos, new_vel], dim=-1)

        # ── Scale dynamics ──
        s = self._scale_states[:, :, 0]       # (B, n)
        s_dot = self._scale_states[:, :, 1]   # (B, n)
        new_s_dot = s_dot + a_s * dt
        new_s_dot = torch.clamp(new_s_dot, -s_dot_max, s_dot_max)
        new_s = s + new_s_dot * dt
        s_min = self.params["s_min"]
        s_max = self.params["s_max"]
        new_s = torch.clamp(new_s, s_min, s_max)
        # Zero out velocity at limits
        new_s_dot = torch.where(
            (new_s <= s_min) & (new_s_dot < 0), torch.zeros_like(new_s_dot), new_s_dot
        )
        new_s_dot = torch.where(
            (new_s >= s_max) & (new_s_dot > 0), torch.zeros_like(new_s_dot), new_s_dot
        )
        self._scale_states = torch.stack([new_s, new_s_dot], dim=-1)

        # ── Payload swing dynamics (Semi-implicit Euler) ──
        l = self.params["cable_length"]
        g = self.params["gravity"]
        c = self.params["payload_damping"]
        ps = self._payload_states
        gx     = ps[:, :, 0]
        gy     = ps[:, :, 1]
        gx_dot = ps[:, :, 2]
        gy_dot = ps[:, :, 3]

        ax = accel[:, :, 0]
        ay = accel[:, :, 1]

        gx_ddot = -(g / l) * torch.sin(gx) - (ax / l) * torch.cos(gx) - c * gx_dot
        gy_ddot = -(g / l) * torch.sin(gy) - (ay / l) * torch.cos(gy) - c * gy_dot

        new_gx_dot = gx_dot + gx_ddot * dt
        new_gy_dot = gy_dot + gy_ddot * dt
        new_gx     = gx + new_gx_dot * dt
        new_gy     = gy + new_gy_dot * dt
        self._payload_states = torch.stack([new_gx, new_gy, new_gx_dot, new_gy_dot], dim=-1)

        self._step_counts += 1

    # ── Nominal controller (Level 1+2: velocity-command + PD tracking) ──
    def nominal_controller(self, v_target=None, s_dot_target=None):
        """Returns (B, n, 3): [a_cx_nom, a_cy_nom, a_s_nom].

        Level 1: Compute target velocities.
            v_ref = K_pos * (goal_pos - pos)   (proportional goal tracking)
            v_target = v_ref + v_gnn_offset     (if provided externally)
            s_dot_target = s_dot_gnn            (if provided externally)

        Level 2: PD controller (velocity → acceleration).
            a_trans_nom = K_v * (v_target - v_current)
            a_s_nom     = K_s * (s_dot_target - s_dot_current)

        Parameters
        ----------
        v_target : (B, n, 2) or None
            Target translation velocity. If None, uses LQR-like v_ref.
        s_dot_target : (B, n) or None
            Target scale velocity. If None, uses 0.
        """
        K_pos = self.params.get("K_pos", 0.5)
        K_v = self.params.get("K_v", 2.0)
        K_s = self.params.get("K_s", 2.0)
        v_max = self.params.get("v_max", 1.0)

        # Level 1: target velocity from proportional goal tracking
        if v_target is None:
            pos = self._agent_states[:, :, :2]      # (B, n, 2)
            goal_pos = self._goal_states[:, :, :2]   # (B, n, 2)
            v_ref = K_pos * (goal_pos - pos)          # (B, n, 2)
            # Clamp to physical velocity limits
            v_ref = torch.clamp(v_ref, -v_max, v_max)
            v_target = v_ref

        if s_dot_target is None:
            s_dot_target = torch.zeros_like(self._scale_states[:, :, 0])

        # Level 2: PD controller
        v_current = self._agent_states[:, :, 2:4]   # (B, n, 2)
        s_dot_current = self._scale_states[:, :, 1]  # (B, n)

        a_trans_nom = K_v * (v_target - v_current)    # (B, n, 2)
        a_s_nom = K_s * (s_dot_target - s_dot_current)  # (B, n)

        u = torch.cat([a_trans_nom, a_s_nom.unsqueeze(-1)], dim=-1)
        return u  # (B, n, 3)

    # ── Unsafe mask (dynamic bounding circle) ─────────────────────────
    def unsafe_mask(self) -> torch.Tensor:
        """Returns (B, n) bool."""
        B = self.batch_size
        n = self.num_agents
        device = self._agent_states.device

        pos = self._agent_states[:, :, :2]  # (B, n, 2)
        s = self._scale_states[:, :, 0]     # (B, n)
        r = self.r_swarm(s)                 # (B, n) — per-agent dynamic radius

        agent_collision = torch.zeros(B, n, dtype=torch.bool, device=device)
        if n > 1:
            diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, n, n, 2)
            dist = torch.norm(diff, dim=-1)              # (B, n, n)
            dist = dist + torch.eye(n, device=device).unsqueeze(0) * 1e6
            # Dynamic collision threshold: r_i + r_j
            thresh = r.unsqueeze(2) + r.unsqueeze(1)    # (B, n, n)
            collision_matrix = dist < thresh
            agent_collision = collision_matrix.any(dim=2)

        obs_collision = torch.zeros(B, n, dtype=torch.bool, device=device)
        if self._obstacle_centers is not None and self._obstacle_centers.shape[1] > 0:
            dp = pos.unsqueeze(2) - self._obstacle_centers.unsqueeze(1) # (B, n, n_obs, 2)
            dist = torch.norm(dp, dim=-1)                                # (B, n, n_obs)
            r_exp = r.unsqueeze(2)                                       # (B, n, 1)
            r_obs = self._obstacle_radii.unsqueeze(1)                    # (B, 1, n_obs)
            inside = dist < (r_exp + r_obs)
            obs_collision = inside.any(dim=-1)  # (B, n)

        return agent_collision | obs_collision

    def get_lidar_hits(self, num_beams: int = 32) -> torch.Tensor:
        """
        Vectorized LiDAR sensing against circular obstacles.
        Returns: (B, n, num_beams, 4) [px, py, vx, vy] hit points.
        Points that don't hit anything are set to a very far distance.
        """
        B = self.batch_size
        n = self.num_agents
        device = self._agent_states.device

        pos = self._agent_states[:, :, :2]  # (B, n, 2)
        s = self._scale_states[:, :, 0]     # (B, n)
        sensing_radius = (self.comm_radius * s).unsqueeze(-1)  # (B, n, 1)

        # 1. Setup beams
        angles = torch.linspace(0, 2 * math.pi, num_beams + 1, device=device)[:-1]
        dir_ = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (nb, 2)

        # Ray: p0 (B, n, 1, 2), p1 (B, n, nb, 2), D = p1 - p0
        p0 = pos.unsqueeze(2)  # (B, n, 1, 2)
        p1 = p0 + sensing_radius.unsqueeze(-1) * dir_.view(1, 1, num_beams, 2)
        D = p1 - p0  # (B, n, nb, 2)

        if self._obstacle_centers is None or self._obstacle_centers.shape[1] == 0:
            hits = torch.full((B, n, num_beams, 4), 1e6, device=device)
            self._last_lidar_hits = hits
            return hits

        n_obs = self._obstacle_centers.shape[1]

        # 2. Ray-Circle Intersection (vectorized over B, n, beams, n_obs)
        # f = p0 - center: (B, n, 1, n_obs, 2)
        oc = self._obstacle_centers.view(B, 1, 1, n_obs, 2)   # (B, 1, 1, n_obs, 2)
        or_ = self._obstacle_radii.view(B, 1, 1, n_obs)        # (B, 1, 1, n_obs)
        f = p0.unsqueeze(3) - oc                               # (B, n, 1, n_obs, 2)
        D_exp = D.unsqueeze(3)                                 # (B, n, nb, 1, 2)

        a = (D_exp * D_exp).sum(dim=-1)      # (B, n, nb, n_obs)
        b = 2.0 * (f * D_exp).sum(dim=-1)   # (B, n, nb, n_obs)  — f broadcast over nb
        # f needs broadcasting: (B, n, 1, n_obs, 2) with D_exp (B, n, nb, 1, 2)
        f_exp = p0.unsqueeze(3).expand(B, n, num_beams, n_obs, 2) - oc.expand(B, n, num_beams, n_obs, 2)
        b = 2.0 * (f_exp * D_exp).sum(dim=-1)
        c = (f_exp * f_exp).sum(dim=-1) - or_ ** 2            # (B, n, nb, n_obs)

        disc = b * b - 4.0 * a * c                            # (B, n, nb, n_obs)
        valid_disc = disc >= 0.0

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a + 1e-8)              # nearest intersection
        t2 = (-b + sqrt_disc) / (2.0 * a + 1e-8)

        # Use t1 (nearest), fall back to t2 if t1 < 0, mark invalid if neither in [0,1]
        t = torch.where(t1 >= 0.0, t1, t2)
        valid = valid_disc & (t >= 0.0) & (t <= 1.0)
        t = torch.where(valid, t, torch.full_like(t, 1.1))

        # Closest obstacle per beam
        min_t, _ = torch.min(t, dim=-1)     # (B, n, nb)
        hit_mask = min_t <= 1.0

        # Hit positions
        hit_pos = p0 + min_t.unsqueeze(-1) * D  # (B, n, nb, 2)

        res = torch.full((B, n, num_beams, 4), 1e6, device=device)
        res[hit_mask, :2] = hit_pos[hit_mask]

        self._last_lidar_hits = res
        return res

    def get_done_masks(self) -> torch.Tensor:
        """
        Check terminal conditions for each batch.
        Returns: (B,) boolean tensor.
        """
        B = self.batch_size
        n = self.num_agents
        
        # 1. Collision Check
        # unsafe_mask() gives (B, n), if ANY agent is unsafe, the whole batch is 'done' (failed)
        collision = self.unsafe_mask().any(dim=1) # (B,)
        
        # 2. Goal Check
        # Check if all agents in the batch are close to their goals
        pos = self._agent_states[:, :, :2]
        goal = self._goal_states[:, :, :2]
        dist_to_goal = torch.norm(pos - goal, dim=-1) # (B, n)
        is_goal = dist_to_goal < 0.2
        all_goal = is_goal.all(dim=1) # (B,)
        
        # 3. Time Limit Check (動画に合わせて512step)
        timeout = self._step_counts >= self.max_steps
        
        return collision | all_goal | timeout

    # ── Graph builder ─────────────────────────────────────────────────
    def build_batch_graph(self, agent_states=None, scale_states=None) -> GraphsTuple:
        if agent_states is None:
            agent_states = self._agent_states
        if scale_states is None:
            scale_states = self._scale_states
            
        # 1. Update LiDAR hits for this step
        hits = self.get_lidar_hits(num_beams=16) # (B, n, n_beams, 4)
        
        # 2. Build graph using hit points instead of centers
        # Flatten hits for graph builder: (B, n * n_beams, 4)
        B, n, nb, _ = hits.shape
        flat_hits = hits.view(B, n * nb, 4)
        
        s = scale_states[:, :, 0]
        dyn_cr = self.comm_radius * s
        
        use_payload = self.params.get("use_payload", True)
        return build_vectorized_swarm_graph(
            agent_states=agent_states,
            goal_states=self._goal_states,
            obstacle_states=flat_hits,
            comm_radius=dyn_cr,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            payload_states=self._payload_states if use_payload else None,
        )
