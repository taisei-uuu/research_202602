#!/usr/bin/env python3
"""
Visualization: Animate multi-agent trajectories as an MP4 video.

Supports:
  1. DoubleIntegrator (single agent dots)
  2. SwarmIntegrator  (3-drone triangle formations + dynamic bounding circles)

Usage (Colab):
    # LQR mode (single agent)
    !python visualize.py --num_agents 4 --seed 0

    # Swarm LQR (no training)
    !python visualize.py --swarm_lqr --num_agents 3 --seed 0

    # Trained affine-transform policy
    !python visualize.py --checkpoint affine_swarm_checkpoint.pt --seed 0
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import numpy as np
import torch

import json

try:
    import quadprog
    _QUADPROG_AVAILABLE = True
except ImportError:
    _QUADPROG_AVAILABLE = False

from gcbf_plus.env import DoubleIntegrator, SwarmIntegrator
from gcbf_plus.env.swarm_integrator import Obstacle
from gcbf_plus.nn import PolicyNetwork
from gcbf_plus.utils.swarm_graph import build_swarm_graph_from_states
from gcbf_plus.algo.affine_qp_solver import solve_affine_qp
from gcbf_plus.train_swarm import extract_agent_outputs


AGENT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000",
]


def compute_triangle_vertices(com_x, com_y, theta, R_form):
    """Compute 3 vertices of equilateral triangle at (com_x, com_y) with scaled R_form."""
    s32 = math.sqrt(3.0) / 2.0
    local = np.array([
        [R_form, 0.0],
        [-R_form / 2.0,  R_form * s32],
        [-R_form / 2.0, -R_form * s32],
    ])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = local @ R.T + np.array([com_x, com_y])
    return verts


def _apply_scenario(env: SwarmIntegrator, scenario_path: str) -> Optional[float]:
    """Override env state with hardcoded positions from a JSON scenario file.

    Called after env.reset() so random state is replaced by the scenario.
    JSON format:
        {
          "_arena":    "7.5x7.5m",                 # optional: overrides area_size
          "agents":    [[x0,y0], [x1,y1], ...],   # start positions
          "goals":     [[x0,y0], [x1,y1], ...],   # goal positions
          "obstacles": [{"center":[cx,cy], "half_size":[hw,hh]}, ...]
        }

    Returns the area_size parsed from "_arena" if present, else None.
    """
    with open(scenario_path) as f:
        sc = json.load(f)

    # Parse _arena field (e.g. "7.5x7.5m" → 7.5)
    scenario_area: Optional[float] = None
    if "_arena" in sc:
        try:
            scenario_area = float(sc["_arena"].split("x")[0])
            env.area_size = scenario_area
        except (ValueError, IndexError):
            pass

    n = env.num_agents

    if "agents" in sc:
        pos = torch.tensor(sc["agents"], dtype=torch.float32)  # (n, 2)
        if pos.shape[0] != n:
            print(f"  [scenario] WARNING: JSON has {pos.shape[0]} agents but env expects {n}. Using JSON count.")
        env.agent_states = torch.zeros(pos.shape[0], 4, dtype=torch.float32)
        env.agent_states[:, :2] = pos
        # Sync num_agents and resize dependent states to match new agent count
        new_n = pos.shape[0]
        env.num_agents = new_n
        s_init = float(sc.get("scale_init", 1.0))
        env.scale_states = torch.tensor([[s_init, 0.0]] * new_n, dtype=torch.float32)
        env.payload_states = torch.zeros(new_n, 4, dtype=torch.float32)

    if "goals" in sc:
        pos = torch.tensor(sc["goals"], dtype=torch.float32)   # (n, 2)
        env.goal_states = torch.zeros(pos.shape[0], 4, dtype=torch.float32)
        env.goal_states[:, :2] = pos

    if "obstacles" in sc:
        env._obstacles = []
        obs_states = []
        for obs in sc["obstacles"]:
            center = torch.tensor(obs["center"], dtype=torch.float32)
            radius = float(obs["radius"])
            env._obstacles.append(Obstacle(center=center, radius=radius))
            s4 = torch.zeros(4)
            s4[:2] = center
            s4[2] = radius
            obs_states.append(s4)
        env._obstacle_states = torch.stack(obs_states) if obs_states else None

    print(f"  [scenario] Loaded from: {scenario_path}")
    print(f"    agents={len(sc.get('agents',[]))}  goals={len(sc.get('goals',[]))}  obstacles={len(sc.get('obstacles',[]))}")
    if scenario_area is not None:
        print(f"    area_size overridden → {scenario_area}")
    return scenario_area


def load_trained_policy(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    policy_net = PolicyNetwork(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        action_dim=cfg["action_dim"],
        n_agents=cfg["num_agents"],
    )
    policy_net.load_state_dict(ckpt["policy_net"])
    policy_net.eval()

    print(f"  Loaded trained policy from: {checkpoint_path}")
    print(f"    agents={cfg['num_agents']}  area={cfg['area_size']}  "
          f"n_obs={cfg['n_obs']}  dt={cfg['dt']}  comm_radius={cfg['comm_radius']}")
    arch = cfg.get("architecture", "unknown")
    use_payload = cfg.get("use_payload", True)
    print(f"    architecture={arch}  R_form={cfg.get('R_form', 'N/A')}"
          f"  s_min={cfg.get('s_min', 'N/A')}  s_max={cfg.get('s_max', 'N/A')}"
          f"  use_payload={use_payload}")
    return policy_net, cfg


def _solve_qp_exact_single(
    u_nom_i,       # (3,) numpy float64
    s_i, sd_i,     # float
    obs_hits_i,    # (K, 2) numpy — valid (filtered) obstacle hit points, or None
    agent_pos_i,   # (2,) numpy
    agent_vel_i,   # (2,) numpy
    other_pos,     # (M, 2) numpy or None
    other_vel,     # (M, 2) numpy or None
    other_s,       # (M,) numpy or None
    other_sd,      # (M,) numpy or None
    payload_i,     # (4,) numpy or None — [gx, gy, gx_dot, gy_dot]
    R_form, r_margin, s_min, s_max,
    alpha1_scale, alpha2_scale,
    alpha1_obs, alpha2_obs,
    hocbf_alpha1, hocbf_alpha2,
    cable_length, gravity, payload_damping,
    slack_weight, u_max,
) -> np.ndarray:
    """Solve per-agent QP exactly using quadprog.

    Variable: X5 = [a_cx, a_cy, a_s, delta_x, delta_y]  (5D)
    Objective: min 1/2 X5^T G X5 - a^T X5
    where G = diag([1, 1, 1, p, p]),  a = [u_nom; 0; 0]

    Returns: (3,) numpy array [a_cx, a_cy, a_s]
    """
    p = slack_weight
    G = np.diag([1.0, 1.0, 1.0, p, p]) + np.eye(5) * 1e-8  # ensure positive definite
    a_vec = np.array([u_nom_i[0], u_nom_i[1], u_nom_i[2], 0.0, 0.0], dtype=np.float64)

    # Constraint rows: C^T @ X5 >= b  (each row is a constraint normal)
    C_rows = []
    b_vals = []

    def add(row, b):
        C_rows.append(np.array(row, dtype=np.float64))
        b_vals.append(float(b))

    # ── Scale CBF (2nd order HOCBF) ──────────────────────────────────
    alpha_sum_s = alpha1_scale + alpha2_scale
    alpha_prod_s = alpha1_scale * alpha2_scale
    lb = -alpha_sum_s * sd_i - alpha_prod_s * (s_i - s_min)
    ub = -alpha_sum_s * sd_i + alpha_prod_s * (s_max - s_i)
    add([0, 0,  1, 0, 0],  lb)   # a_s >= lb
    add([0, 0, -1, 0, 0], -ub)   # a_s <= ub

    # ── Obstacle CBF (2nd order HOCBF) ───────────────────────────────
    if obs_hits_i is not None and len(obs_hits_i) > 0:
        a1, a2 = alpha1_obs, alpha2_obs
        r_sw = R_form * s_i + r_margin
        r_dot = R_form * sd_i
        for obs_pt in obs_hits_i:
            dp = agent_pos_i - obs_pt
            dist_sq = float(np.dot(dp, dp))
            h = dist_sq - r_sw ** 2
            h_dot = 2.0 * float(np.dot(dp, agent_vel_i)) - 2.0 * r_sw * r_dot
            h_ddot_drift = 2.0 * float(np.dot(agent_vel_i, agent_vel_i)) - 2.0 * r_dot ** 2
            A_cx = 2.0 * dp[0]
            A_cy = 2.0 * dp[1]
            A_as = -2.0 * r_sw * R_form
            rhs = h_ddot_drift + (a1 + a2) * h_dot + (a1 * a2) * h
            add([A_cx, A_cy, A_as, 0, 0], -rhs)

    # ── Agent-Agent CBF (2nd order HOCBF + Reciprocal CA) ────────────
    if other_pos is not None and len(other_pos) > 0:
        a1, a2 = alpha1_obs, alpha2_obs
        r_sw = R_form * s_i + r_margin
        for j in range(len(other_pos)):
            r_other = R_form * float(other_s[j]) + r_margin
            safe_dist = r_sw + r_other
            dp = agent_pos_i - other_pos[j]
            dv = agent_vel_i - other_vel[j]
            dist_sq = float(np.dot(dp, dp))
            h = dist_sq - safe_dist ** 2
            r_dot_total = R_form * (sd_i + float(other_sd[j]))
            h_dot = 2.0 * float(np.dot(dp, dv)) - 2.0 * safe_dist * r_dot_total
            h_ddot_drift = 2.0 * float(np.dot(dv, dv)) - 2.0 * r_dot_total ** 2
            A_cx = 2.0 * dp[0]
            A_cy = 2.0 * dp[1]
            A_as = -2.0 * safe_dist * R_form
            rhs = (h_ddot_drift + (a1 + a2) * h_dot + (a1 * a2) * h) * 0.5  # RCA 1/2
            add([A_cx, A_cy, A_as, 0, 0], -rhs)

    # ── Payload HOCBF (soft, with slack delta) ────────────────────────
    if payload_i is not None:
        gx, gy, gx_dot, gy_dot = payload_i
        g_val, c_damp = gravity, payload_damping
        # l_eff(s): effective vertical pendulum length
        l = np.sqrt(max(cable_length**2 - (R_form * s_i)**2, 1e-4))
        a1, a2 = hocbf_alpha1, hocbf_alpha2
        alpha_sum_h = a1 + a2
        alpha_prod_h = a1 * a2

        ratio = np.clip(R_form * s_i / cable_length, 0.0, 0.95)  # angle bound uses full cable
        gamma_dyn = np.arcsin(ratio)

        # dγ_max/ds and time derivatives (γ_max is time-varying via s(t))
        safe_denom = np.sqrt(max(1.0 - ratio**2, 1e-6))
        dgamma_ds   = (R_form / cable_length) / safe_denom
        d2gamma_ds2 = (R_form / cable_length)**2 * ratio / safe_denom**3
        gamma_dyn_dot = dgamma_ds * sd_i                 # dγ_max/dt
        C_s_h = 2.0 * gamma_dyn * dgamma_ds              # coupling to a_s
        extra_drift = (2.0 * gamma_dyn_dot**2
                       + 2.0 * gamma_dyn * d2gamma_ds2 * sd_i**2)

        # X-axis: C_x * a_cx + C_s * a_s + rhs_x + delta_x >= 0
        h_x = gamma_dyn ** 2 - gx ** 2
        h_dot_x = 2.0 * gamma_dyn * gamma_dyn_dot - 2.0 * gx * gx_dot
        h_ddot_drift_x = (extra_drift
                          - 2.0 * gx_dot ** 2
                          + 2.0 * gx * g_val / l * np.sin(gx)
                          + 2.0 * gx * c_damp * gx_dot)
        C_x = (2.0 * gx * np.cos(gx)) / l
        rhs_x = h_ddot_drift_x + alpha_sum_h * h_dot_x + alpha_prod_h * h_x
        add([C_x, 0, C_s_h, 1, 0], -rhs_x)

        # Y-axis: C_y * a_cy + C_s * a_s + rhs_y + delta_y >= 0
        h_y = gamma_dyn ** 2 - gy ** 2
        h_dot_y = 2.0 * gamma_dyn * gamma_dyn_dot - 2.0 * gy * gy_dot
        h_ddot_drift_y = (extra_drift
                          - 2.0 * gy_dot ** 2
                          + 2.0 * gy * g_val / l * np.sin(gy)
                          + 2.0 * gy * c_damp * gy_dot)
        C_y = (2.0 * gy * np.cos(gy)) / l
        rhs_y = h_ddot_drift_y + alpha_sum_h * h_dot_y + alpha_prod_h * h_y
        add([0, C_y, C_s_h, 0, 1], -rhs_y)

        # delta >= 0
        add([0, 0, 0, 1, 0], 0.0)
        add([0, 0, 0, 0, 1], 0.0)

    # ── Box constraint on translation ─────────────────────────────────
    if u_max is not None:
        add([ 1, 0, 0, 0, 0], -u_max)
        add([-1, 0, 0, 0, 0], -u_max)
        add([ 0, 1, 0, 0, 0], -u_max)
        add([ 0,-1, 0, 0, 0], -u_max)

    # ── Solve ──────────────────────────────────────────────────────────
    C_mat = np.array(C_rows, dtype=np.float64).T  # (5, n_constraints)
    b_vec = np.array(b_vals, dtype=np.float64)

    try:
        sol = quadprog.solve_qp(G, a_vec, C_mat, b_vec)
        return sol[0][:3].astype(np.float32)
    except Exception:
        # infeasible or numerical failure → return nominal (already safe enough for viz)
        return u_nom_i[:3].astype(np.float32)


def solve_affine_qp_exact(
    u_nom,
    obs_hits, agent_pos, agent_vel, s, s_dot,
    other_agent_pos, other_agent_vel, other_agent_s, other_agent_s_dot,
    payload_states,
    R_form, r_margin, s_min, s_max,
    cable_length, gravity, payload_damping,
    slack_weight=100.0, u_max=None,
    alpha1_scale=2.0, alpha2_scale=2.0,
    alpha1_obs=0.8, alpha2_obs=0.8,
    hocbf_alpha1=2.0, hocbf_alpha2=2.0,
) -> torch.Tensor:
    """Exact QP solver (quadprog) for visualization.  Loops over agents."""
    N = u_nom.shape[0]
    results = []

    for i in range(N):
        u_nom_i = u_nom[i].numpy().astype(np.float64)
        s_i = float(s[i])
        sd_i = float(s_dot[i])
        pos_i = agent_pos[i].numpy().astype(np.float64)
        vel_i = agent_vel[i].numpy().astype(np.float64)

        # Filter valid obstacle hits (non-hits stored as 1e6)
        if obs_hits is not None:
            hits_i = obs_hits[i].numpy()  # (nb, 2)
            valid = ~np.any(hits_i > 1e5, axis=-1)
            obs_i = hits_i[valid].astype(np.float64) if valid.any() else None
        else:
            obs_i = None

        # Other agents
        op = other_agent_pos[i].numpy().astype(np.float64) if other_agent_pos is not None else None
        ov = other_agent_vel[i].numpy().astype(np.float64) if other_agent_vel is not None else None
        os_ = other_agent_s[i].numpy().astype(np.float64) if other_agent_s is not None else None
        osd = other_agent_s_dot[i].numpy().astype(np.float64) if other_agent_s_dot is not None else None

        pay_i = payload_states[i].numpy().astype(np.float64) if payload_states is not None else None

        sol_i = _solve_qp_exact_single(
            u_nom_i, s_i, sd_i,
            obs_i, pos_i, vel_i,
            op, ov, os_, osd,
            pay_i,
            R_form, r_margin, s_min, s_max,
            alpha1_scale, alpha2_scale,
            alpha1_obs, alpha2_obs,
            hocbf_alpha1, hocbf_alpha2,
            cable_length, gravity, payload_damping,
            slack_weight, u_max,
        )
        results.append(sol_i)

    return torch.tensor(np.stack(results), dtype=torch.float32)


def run_simulation(
    num_agents: int = 4,
    area_size: float = 2.0,
    max_steps: int = 512,
    dt: float = 0.03,
    n_obs: Optional[int] = None,
    seed: int = 0,
    checkpoint_path: Optional[str] = None,
    force_lqr: bool = False,
    swarm_lqr: bool = False,
    scenario_path: Optional[str] = None,
    use_exact_qp: bool = False,
    no_scale: bool = False,
    s_max_one: bool = False,
    method: str = "affine_policy",
):
    policy_net = None
    mode = "lqr"
    is_swarm = swarm_lqr
    R_form = 0.5
    r_margin = 0.2
    comm_radius = 3.0 if swarm_lqr else 1.5
    s_min = 0.4
    s_max = 1.5
    cfg = None

    if n_obs is None:
        n_obs = 2  # default when no checkpoint

    if swarm_lqr:
        area_size = max(area_size, 15.0)

    if checkpoint_path is not None:
        policy_net, cfg = load_trained_policy(checkpoint_path)
        num_agents = cfg["num_agents"]
        area_size = cfg["area_size"]
        n_obs_cfg = cfg["n_obs"]
        if n_obs is None:
            n_obs = n_obs_cfg
        else:
            print(f"  [n_obs] Overriding training config ({n_obs_cfg}) → {n_obs}")
        dt = cfg["dt"]
        comm_radius = cfg["comm_radius"]
        mode = "trained_policy"
        is_swarm = cfg.get("architecture") in ("affine_transform", "hierarchical_velocity_command") or "R_form" in cfg
        R_form = cfg.get("R_form", 0.5)
        r_margin = cfg.get("r_margin", 0.2)
        s_min = cfg.get("s_min", 0.4)
        s_max = cfg.get("s_max", 1.5)
        if force_lqr or method != "affine_policy":
            policy_net = None
            if force_lqr:
                print("  [force_lqr] Ignoring trained policy — using LQR only")

    # Set mode label based on method
    if method == "hocbf_lqr":
        mode = "hocbf_lqr"
    elif method == "lqr_only":
        mode = "lqr"

    if no_scale:
        print("  [no_scale] Formation scale fixed at s=1.0")
    if s_max_one:
        print("  [s_max_one] Scale capped at s_max=1.0 (shrink allowed)")

    if is_swarm:
        # Use full cfg to ensure mass, gravity, obs_len_range, etc. exactly match training
        env_params = cfg.copy() if cfg else {}
        env_params.update({"n_obs": n_obs, "comm_radius": comm_radius})
        if no_scale:
            env_params["s_min"] = 1.0
            env_params["s_max"] = 1.0
        elif s_max_one:
            env_params["s_max"] = 1.0
        env = SwarmIntegrator(
            num_agents=num_agents, area_size=area_size, dt=dt, max_steps=max_steps,
            params=env_params,
        )
    else:
        env = DoubleIntegrator(
            num_agents=num_agents, area_size=area_size, dt=dt, max_steps=max_steps,
            params={"n_obs": n_obs, "comm_radius": comm_radius},
        )

    env.reset(seed=seed)

    if scenario_path is not None:
        sc_area = _apply_scenario(env, scenario_path)
        if sc_area is not None:
            area_size = sc_area
        num_agents = env.num_agents  # update in case scenario changed agent count

    trajectories: List[np.ndarray] = []
    payload_trajectories: List[np.ndarray] = []
    scale_trajectories: List[np.ndarray] = []
    edge_trajectories: List[np.ndarray] = []
    lidar_trajectories: List[List[np.ndarray]] = []

    _use_payload = cfg.get("use_payload", True) if cfg else True
    trajectories.append(env.agent_states.detach().numpy().copy())
    if _use_payload and hasattr(env, 'payload_states') and env.payload_states is not None:
        payload_trajectories.append(env.payload_states.detach().numpy().copy())
    if hasattr(env, 'scale_states') and env.scale_states is not None:
        scale_trajectories.append(env.scale_states.detach().numpy().copy())
    
    # Store initial graph edges
    init_graph = env._get_graph()
    if hasattr(init_graph, 'senders') and init_graph.senders is not None:
        edges = torch.stack([init_graph.senders, init_graph.receivers], dim=0)
        edge_trajectories.append(edges.detach().numpy().copy())
    else:
        edge_trajectories.append(np.empty((2, 0)))
    
    # Store initial LiDAR hits (via _get_graph call above)
    lidar_trajectories.append([h.detach().numpy().copy() for h in env._last_lidar_hits])

    goals = env.goal_states.detach().numpy().copy()
    obstacle_info = [(obs.center.numpy().copy(), float(obs.radius))
                     for obs in env._obstacles]

    # Control config
    v_max_cfg = cfg.get("v_max", 1.0) if cfg else 1.0
    K_pos_cfg = cfg.get("K_pos", 0.5) if cfg else 0.5
    K_v_cfg = cfg.get("K_v", 2.0) if cfg else 2.0
    K_s_cfg = cfg.get("K_s", 2.0) if cfg else 2.0
    # GNN acceleration output scale (must match train_swarm.py)
    a_max_gnn = 1.0    # m/s²
    a_max_gnn_s = 0.5  # s⁻²

    for step_idx in range(max_steps):
        if method == "lqr_only":
            u = env.nominal_controller()
        else:
            # affine_policy or hocbf_lqr — both use QP; differ only in u_nom source
            with torch.no_grad():
                pos = env.agent_states[:, :2]
                goal_pos = env.goal_states[:, :2]
                v_current = env.agent_states[:, 2:4]
                sc = env.scale_states[:, 0]
                sd = env.scale_states[:, 1]

                # PD nominal: translation toward goal, scale toward s_max
                v_ref = K_pos_cfg * (goal_pos - pos)
                v_ref = torch.clamp(v_ref, -v_max_cfg, v_max_cfg)
                s_dot_ref = 1.0 * (s_max - sc)
                a_trans = K_v_cfg * (v_ref - v_current)
                a_s = K_s_cfg * (s_dot_ref - sd)
                u_nom = torch.cat([a_trans, a_s.unsqueeze(-1)], dim=-1)

                if method == "affine_policy" and policy_net is not None:
                    # GNN → tanh → scale to physical acceleration offset
                    graph = env._get_graph()
                    pi_tanh = policy_net(graph)  # (n, 3) already tanh'd
                    pi_scaled = pi_tanh.clone()
                    pi_scaled[:, :2] *= a_max_gnn
                    pi_scaled[:, 2] *= a_max_gnn_s
                    if no_scale:
                        pi_scaled[:, 2] = 0.0
                    u_nom = u_nom + pi_scaled

                if no_scale:
                    u_nom[:, 2] = 0.0

                # Pre-clamp to physically feasible range
                _u_max = env.params.get("u_max")
                if _u_max is not None:
                    a_max_t = _u_max * 0.7
                    a_max_s = _u_max * 0.3
                    u_nom[:, :2] = u_nom[:, :2].clamp(-a_max_t, a_max_t)
                    u_nom[:, 2]  = u_nom[:, 2].clamp(-a_max_s, a_max_s)

                # GNN only: skip QP
                if method == "gnn_only":
                    u = u_nom
                    next_obs, info = env.step(u)
                    trajectories.append(env.agent_states.detach().numpy().copy())
                    if _use_payload and hasattr(env, 'payload_states') and env.payload_states is not None:
                        payload_trajectories.append(env.payload_states.detach().numpy().copy())
                    if hasattr(env, 'scale_states') and env.scale_states is not None:
                        scale_trajectories.append(env.scale_states.detach().numpy().copy())
                    curr_graph = env._get_graph()
                    edges = torch.stack([curr_graph.senders, curr_graph.receivers], dim=0)
                    edge_trajectories.append(edges.detach().numpy().copy())
                    lidar_trajectories.append([h.detach().numpy().copy() for h in env._last_lidar_hits])
                    if info["done"]:
                        break
                    continue

                ps = env.payload_states if _use_payload else None

                # Obstacle hits from LiDAR — always returns (num_agents, 32, 4)
                nb = 32
                lidar_hits_4d = env.get_lidar_hits(num_beams=nb)  # (num_agents, nb, 4)
                lidar_hits = lidar_hits_4d[..., :2]               # (num_agents, nb, 2)

                # Agent-Agent info
                if num_agents > 1:
                    dev = pos.device
                    agent_idx = torch.arange(num_agents, device=dev)
                    mask = agent_idx.unsqueeze(0) != agent_idx.unsqueeze(1)
                    pos_other = pos.unsqueeze(0).expand(num_agents, num_agents, 2)
                    other_pos_flat = pos_other[mask].reshape(num_agents, num_agents - 1, 2)
                    vel_other = v_current.unsqueeze(0).expand(num_agents, num_agents, 2)
                    other_vel_flat = vel_other[mask].reshape(num_agents, num_agents - 1, 2)
                    s_other = sc.unsqueeze(0).expand(num_agents, num_agents)
                    other_s_flat = s_other[mask].reshape(num_agents, num_agents - 1)
                    sd_other = sd.unsqueeze(0).expand(num_agents, num_agents)
                    other_sd_flat = sd_other[mask].reshape(num_agents, num_agents - 1)
                else:
                    other_pos_flat = other_vel_flat = other_s_flat = other_sd_flat = None

                _qp_u_max = env.params.get("u_max")
                if use_exact_qp and _QUADPROG_AVAILABLE:
                    u = solve_affine_qp_exact(
                        u_nom=u_nom,
                        obs_hits=lidar_hits,
                        agent_pos=pos,
                        agent_vel=v_current,
                        s=sc,
                        s_dot=sd,
                        other_agent_pos=other_pos_flat,
                        other_agent_vel=other_vel_flat,
                        other_agent_s=other_s_flat,
                        other_agent_s_dot=other_sd_flat,
                        payload_states=ps,
                        R_form=R_form,
                        r_margin=r_margin,
                        s_min=s_min,
                        s_max=s_max,
                        cable_length=env.params["cable_length"],
                        gravity=env.params["gravity"],
                        payload_damping=env.params["payload_damping"],
                        u_max=_qp_u_max,
                    )
                else:
                    if use_exact_qp and not _QUADPROG_AVAILABLE:
                        print("  [WARNING] quadprog not installed — falling back to Dykstra QP")
                    u = solve_affine_qp(
                        u_nom=u_nom,
                        obs_hits=lidar_hits,
                        agent_pos=pos,
                        agent_vel=v_current,
                        s=sc,
                        s_dot=sd,
                        other_agent_pos=other_pos_flat,
                        other_agent_vel=other_vel_flat,
                        other_agent_s=other_s_flat,
                        other_agent_s_dot=other_sd_flat,
                        R_form=R_form,
                        r_margin=r_margin,
                        mass=env.params["mass"],
                        s_min=s_min,
                        s_max=s_max,
                        payload_states=ps,
                        cable_length=env.params["cable_length"],
                        gravity=env.params["gravity"],
                        gamma_min=env.params["gamma_min"],
                        gamma_max_full=env.params["gamma_max_full"],
                        payload_damping=env.params["payload_damping"],
                        u_max=_qp_u_max,
                    )

        next_obs, info = env.step(u)
        trajectories.append(env.agent_states.detach().numpy().copy())
        if _use_payload and hasattr(env, 'payload_states') and env.payload_states is not None:
            payload_trajectories.append(env.payload_states.detach().numpy().copy())
        if hasattr(env, 'scale_states') and env.scale_states is not None:
            scale_trajectories.append(env.scale_states.detach().numpy().copy())
        
        # Save edge index
        curr_graph = env._get_graph()
        if hasattr(curr_graph, 'senders') and curr_graph.senders is not None:
            # FIX: Use curr_graph instead of init_graph
            edges = torch.stack([curr_graph.senders, curr_graph.receivers], dim=0)
            edge_trajectories.append(edges.detach().numpy().copy())
        else:
            edge_trajectories.append(np.empty((2, 0)))
        
        # Save LiDAR hits
        lidar_trajectories.append([h.detach().numpy().copy() for h in env._last_lidar_hits])
        if step_idx == 0 or step_idx == max_steps - 1 or (step_idx % 100 == 0):
            print(f"  [DEBUG] Sim step {step_idx}: Extracted {edge_trajectories[-1].shape[1]} edges")

        if info["done"]:
            break

    trajectories = np.array(trajectories)
    payload_traj = np.array(payload_trajectories) if len(payload_trajectories) > 0 else None
    scale_traj = np.array(scale_trajectories) if len(scale_trajectories) > 0 else None
    displacement = np.linalg.norm(trajectories[-1, :, :2] - trajectories[0, :, :2], axis=1)
    print(f"  Agent displacements: {displacement}")

    cable_length = env.params.get("cable_length", 1.0) if hasattr(env, 'params') else 1.0
    return (trajectories, goals, obstacle_info, area_size, comm_radius, mode,
            is_swarm, R_form, r_margin, payload_traj, cable_length, scale_traj, edge_trajectories, lidar_trajectories)


# ═══════════════════════════════════════════════════════════════════════
# Video creation
# ═══════════════════════════════════════════════════════════════════════

def create_video(
    trajectories, goals, obstacle_info, area_size,
    save_path="trajectories.mp4", fps=30, skip=1,
    mode="lqr", comm_radius=1.5,
    is_swarm=False, R_form=0.3, r_margin=0.2,
    payload_traj=None, cable_length=1.0, scale_traj=None,
    edge_traj=None, lidar_traj=None,
):
    n_agents = trajectories.shape[1]
    total_frames = trajectories.shape[0]
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)]

    frame_indices = list(range(0, total_frames, skip))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)
    n_frames = len(frame_indices)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(-0.3, area_size + 0.3)
    ax.set_ylim(-0.3, area_size + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#adb5bd")

    # Obstacles
    for center, radius in obstacle_info:
        circle = Circle(
            (center[0], center[1]), radius,
            facecolor="#adb5bd", edgecolor="#6c757d",
            linewidth=1.2, alpha=0.65, zorder=5,
        )
        ax.add_patch(circle)

    # Start markers
    starts = trajectories[0]
    for i in range(n_agents):
        if is_swarm:
            verts = compute_triangle_vertices(starts[i, 0], starts[i, 1], 0.0, R_form)
            tri = Polygon(verts, closed=True, facecolor=colors[i],
                          edgecolor="white", linewidth=1.5, alpha=0.35, zorder=6)
            ax.add_patch(tri)
        ax.plot(starts[i, 0], starts[i, 1], marker="o", markersize=10 if is_swarm else 12,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=2, zorder=7)

    # Goal markers
    for i in range(n_agents):
        ax.plot(goals[i, 0], goals[i, 1], marker="*", markersize=18,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=1.2, zorder=7)

    # ── Dynamic elements ─────────────────────────────────────────────
    trail_lines = []
    for i in range(n_agents):
        (line,) = ax.plot([], [], color=colors[i], linewidth=1.8, alpha=0.75, zorder=4)
        trail_lines.append(line)

    swarm_triangles = []
    bounding_circles = []
    current_dots = []

    if is_swarm:
        for i in range(n_agents):
            tri = Polygon(np.zeros((3, 2)), closed=True,
                          facecolor=colors[i], edgecolor="white",
                          linewidth=1.5, alpha=0.6, zorder=8)
            ax.add_patch(tri)
            swarm_triangles.append(tri)
            for k in range(3):
                (d,) = ax.plot([], [], marker="o", markersize=5,
                               markerfacecolor="white",
                               markeredgecolor=colors[i],
                               markeredgewidth=1.2, zorder=9)
                current_dots.append(d)
            # Dynamic bounding circle (radius updated per frame)
            bc = plt.Circle((0, 0), R_form + r_margin, fill=True,
                            facecolor=colors[i], edgecolor=colors[i],
                            linewidth=1.5, alpha=0.12, linestyle="-", zorder=6)
            ax.add_patch(bc)
            bounding_circles.append(bc)
    else:
        for i in range(n_agents):
            (dot,) = ax.plot([], [], marker="o", markersize=8,
                             markerfacecolor=colors[i],
                             markeredgecolor="white", markeredgewidth=1.5,
                             zorder=8)
            current_dots.append(dot)

    # Sensing radius
    sensing_circles = []
    for i in range(n_agents):
        sc = plt.Circle((0, 0), comm_radius, fill=False, linestyle="--",
                         linewidth=1.0, edgecolor=colors[i], alpha=0.25, zorder=3)
        ax.add_patch(sc)
        sensing_circles.append(sc)

    # Payload rendering
    payload_dots = []
    payload_cables = []
    if is_swarm and payload_traj is not None:
        for i in range(n_agents):
            (pdot,) = ax.plot([], [], marker="o", markersize=6,
                              markerfacecolor="#343a40", markeredgecolor="white",
                              markeredgewidth=1.0, zorder=10)
            payload_dots.append(pdot)
            (cable,) = ax.plot([], [], color="#6c757d", linewidth=1.2,
                               linestyle="-", alpha=0.7, zorder=9)
            payload_cables.append(cable)

    entity_name = "swarms" if is_swarm else "agents"
    _mode_labels = {
        "trained_policy": "Trained Policy π(x)",
        "hocbf_lqr": "HOCBF+LQR",
        "lqr": "LQR Controller",
    }
    mode_label = _mode_labels.get(mode, mode)
    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=11, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=15,
    )

    ax.set_title(
        f"{'Swarm' if is_swarm else 'Agent'} Trajectories\n"
        f"({n_agents} {entity_name} · {mode_label})",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)

    # Legend
    from matplotlib.lines import Line2D
    handles = []
    for i in range(n_agents):
        handles.append(Line2D([0], [0], color=colors[i], linewidth=2,
                              label=f"Swarm {i}" if is_swarm else f"Agent {i}"))
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                          markeredgecolor="white", markersize=14, label="Goal (★)"))
    handles.append(mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                                  alpha=0.65, label="Obstacle"))
    if is_swarm:
        handles.append(mpatches.Patch(facecolor="gray", edgecolor="white",
                                      alpha=0.5, label=f"Formation △ (R×s)"))
        handles.append(mpatches.Patch(facecolor="gray", edgecolor="gray",
                                      alpha=0.2, label=f"Safety ○ (R×s+m)"))
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#dee2e6")

    # Graph edges visualization
    from matplotlib.collections import LineCollection
    # Increase zorder to 15 to ensure visibility, set colors dynamically in update loop
    edge_lines = LineCollection([], linewidths=1.2, alpha=0.5, zorder=15)
    ax.add_collection(edge_lines)
    
    print(f"  [DEBUG] Creating video with {len(edge_traj) if edge_traj else 0} edge trajectory frames")

    def init():
        for line in trail_lines:
            line.set_data([], [])
        for dot in current_dots:
            dot.set_data([], [])
        edge_lines.set_segments([])
        return []

    def update(frame_idx):
        sim_step = frame_indices[frame_idx]

        for i in range(n_agents):
            trail = trajectories[:sim_step + 1, i, :]
            trail_lines[i].set_data(trail[:, 0], trail[:, 1])
            cx = trajectories[sim_step, i, 0]
            cy = trajectories[sim_step, i, 1]
            sensing_circles[i].center = (cx, cy)
            # Dynamic sensing radius: R_form * s + (comm_radius - R_form)
            if scale_traj is not None and sim_step < scale_traj.shape[0]:
                s_sense = float(scale_traj[sim_step, i, 0])
                sensing_circles[i].set_radius(R_form * s_sense + (comm_radius - R_form))

            if is_swarm:
                # Get current scale
                s_i = 1.0
                if scale_traj is not None and sim_step < scale_traj.shape[0]:
                    s_i = float(scale_traj[sim_step, i, 0])

                scaled_R = R_form * s_i
                verts = compute_triangle_vertices(cx, cy, 0.0, scaled_R)
                swarm_triangles[i].set_xy(verts)
                for k in range(3):
                    current_dots[i * 3 + k].set_data([verts[k, 0]], [verts[k, 1]])
                bounding_circles[i].center = (cx, cy)
                bounding_circles[i].set_radius(R_form * s_i + r_margin)
            else:
                current_dots[i].set_data([cx], [cy])

        # Payload
        if is_swarm and payload_traj is not None and sim_step < payload_traj.shape[0]:
            for i in range(n_agents):
                cx = trajectories[sim_step, i, 0]
                cy = trajectories[sim_step, i, 1]
                gx = payload_traj[sim_step, i, 0]
                gy = payload_traj[sim_step, i, 1]
                px_pay = cx + cable_length * np.sin(gx)
                py_pay = cy + cable_length * np.sin(gy)
                payload_dots[i].set_data([px_pay], [py_pay])
                payload_cables[i].set_data([cx, px_pay], [cy, py_pay])

        # Draw actual PyTorch Geometric GNN graph edges!
        segments = []
        edge_colors = []
        if edge_traj is not None and sim_step < len(edge_traj):
            edges = edge_traj[sim_step] # shape (2, num_edges)
            # The edge indices might refer to 0..N-1 (agents), N..N+M-1 (obs), N+M..2N+M-1 (goals)
            # We need to map these node IDs back to coordinates.
            # In swarm_graph.py `build_swarm_graph_from_states` (used by DoubleIntegrator):
            # agents (0..n-1), obstacles (n..n+n_obs-1), goals (n+n_obs..2n+n_obs-1)
            n_obs_nodes = len(obstacle_info)
            # Total nodes = agents (N) + goals (N) + total lidar hits
            n_lidar_hits = sum(h.shape[0] for h in lidar_traj[sim_step]) if lidar_traj else 0
            n_per = 2 * n_agents + n_lidar_hits
            
            for e_idx in range(edges.shape[1]):
                # Use modulo just in case, though unbatched shouldn't strictly need it
                src = int(edges[0, e_idx]) % n_per
                dst = int(edges[1, e_idx]) % n_per
                
                def get_pos(node_id):
                    # Agent states (0 to n-1)
                    if node_id < n_agents:
                        return (trajectories[sim_step, node_id, 0], trajectories[sim_step, node_id, 1])
                    
                    # Goal states (n to 2n - 1)
                    elif node_id < 2 * n_agents:
                        goal_idx = node_id - n_agents
                        if goal_idx < len(goals):
                            return (goals[goal_idx][0], goals[goal_idx][1])
                        return None
                    
                    # Obstacle / Hit Point states (2n onward)
                    else:
                        hit_idx = node_id - 2 * n_agents
                        if lidar_traj is not None and sim_step < len(lidar_traj):
                            # We need to find which agent this hit point belongs to
                            current_total = 0
                            for a_idx in range(n_agents):
                                hits = lidar_traj[sim_step][a_idx]
                                n_h = hits.shape[0]
                                if current_total <= hit_idx < current_total + n_h:
                                    local_idx = hit_idx - current_total
                                    return (hits[local_idx, 0], hits[local_idx, 1])
                                current_total += n_h
                                
                        # Fallback for old global centers (if any)
                        if n_obs_nodes > 0:
                            obs_idx = node_id - 2 * n_agents
                            if obs_idx < n_obs_nodes:
                                return (obstacle_info[obs_idx][0][0], obstacle_info[obs_idx][0][1])
                        return None

                p1 = get_pos(src)
                p2 = get_pos(dst)
                if p1 is not None and p2 is not None:
                    segments.append([p1, p2])
                    # Color based on receiver's color
                    edge_colors.append(colors[dst])
            
            if sim_step == 0:
                print(f"  [DEBUG] Animation frame 0: {len(segments)} valid segments to draw")
                if len(segments) > 0:
                    print(f"  [DEBUG] Node mapping check (Edge 0): src={int(edges[0,0])}->{p1}, dst={int(edges[1,0])}->{p2}, color={edge_colors[0]}")
                
        edge_lines.set_segments(segments)
        edge_lines.set_edgecolor(edge_colors)

        # Step text with scale info
        scale_info = ""
        if is_swarm and scale_traj is not None and sim_step < scale_traj.shape[0]:
            scales = scale_traj[sim_step, :, 0]
            scale_info = f"  s=[{', '.join(f'{s:.2f}' for s in scales)}]"
        step_text.set_text(f"Step {sim_step}/{total_frames - 1}  [{mode_label}]{scale_info}")
        return []

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_frames,
        interval=1000 // fps, blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Video saved to: {save_path}  ({n_frames} frames @ {fps} fps)")
    return save_path


# ═══════════════════════════════════════════════════════════════════════
# Static plot
# ═══════════════════════════════════════════════════════════════════════

def plot_trajectories(
    trajectories, goals, obstacle_info, area_size,
    save_path="trajectories.png", mode="lqr", comm_radius=1.5,
    is_swarm=False, R_form=0.3, r_margin=0.2,
    payload_traj=None, cable_length=1.0, scale_traj=None,
):
    n_agents = trajectories.shape[1]
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)]
    entity_name = "swarms" if is_swarm else "agents"
    _mode_labels = {
        "trained_policy": "Trained Policy π(x)",
        "hocbf_lqr": "HOCBF+LQR",
        "lqr": "LQR Controller",
    }
    mode_label = _mode_labels.get(mode, mode)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(-0.3, area_size + 0.3)
    ax.set_ylim(-0.3, area_size + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#adb5bd")

    # Obstacles
    for center, radius in obstacle_info:
        circle = Circle(
            (center[0], center[1]), radius,
            facecolor="#adb5bd", edgecolor="#6c757d",
            linewidth=1.2, alpha=0.65, zorder=1,
        )
        ax.add_patch(circle)

    # Trajectories
    for i in range(n_agents):
        path = trajectories[:, i, :]
        ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=1.8, alpha=0.75,
                zorder=2, label=f"Swarm {i}" if is_swarm else f"Agent {i}")

    # Start positions
    starts = trajectories[0]
    for i in range(n_agents):
        if is_swarm:
            verts = compute_triangle_vertices(starts[i, 0], starts[i, 1], 0.0, R_form)
            tri = Polygon(verts, closed=True, facecolor=colors[i],
                          edgecolor="white", linewidth=1.5, alpha=0.35, zorder=4)
            ax.add_patch(tri)
        ax.plot(starts[i, 0], starts[i, 1], marker="o", markersize=10,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=2, zorder=5)

    # Goals
    for i in range(n_agents):
        ax.plot(goals[i, 0], goals[i, 1], marker="*", markersize=18,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)

    # Final positions with dynamic scale
    finals = trajectories[-1]
    for i in range(n_agents):
        if is_swarm:
            s_i = 1.0
            if scale_traj is not None:
                s_i = float(scale_traj[-1, i, 0])
            scaled_R = R_form * s_i
            verts = compute_triangle_vertices(finals[i, 0], finals[i, 1], 0.0, scaled_R)
            tri = Polygon(verts, closed=True, facecolor=colors[i],
                          edgecolor="white", linewidth=1.5, alpha=0.7, zorder=6)
            ax.add_patch(tri)
            for k in range(3):
                ax.plot(verts[k, 0], verts[k, 1], marker="o", markersize=5,
                        markerfacecolor="white", markeredgecolor=colors[i],
                        markeredgewidth=1.2, zorder=7)
            bc = plt.Circle((finals[i, 0], finals[i, 1]), R_form * s_i + r_margin,
                            fill=True, facecolor=colors[i], edgecolor=colors[i],
                            linewidth=1.5, alpha=0.12, zorder=3)
            ax.add_patch(bc)

    # Ghost formations
    if is_swarm:
        n_ghosts = min(8, trajectories.shape[0] // 10)
        if n_ghosts > 0:
            ghost_indices = np.linspace(0, trajectories.shape[0] - 1,
                                        n_ghosts, dtype=int)[1:-1]
            for idx in ghost_indices:
                for i in range(n_agents):
                    s_i = 1.0
                    if scale_traj is not None and idx < scale_traj.shape[0]:
                        s_i = float(scale_traj[idx, i, 0])
                    scaled_R = R_form * s_i
                    verts = compute_triangle_vertices(
                        trajectories[idx, i, 0], trajectories[idx, i, 1], 0.0, scaled_R
                    )
                    tri = Polygon(verts, closed=True, facecolor=colors[i],
                                  edgecolor=colors[i], linewidth=0.5,
                                  alpha=0.12, zorder=3)
                    ax.add_patch(tri)

    # Sensing radius at final
    for i in range(n_agents):
        s_final_i = 1.0
        if scale_traj is not None:
            s_final_i = float(scale_traj[-1, i, 0])
        sc = plt.Circle((finals[i, 0], finals[i, 1]),
                        R_form * s_final_i + (comm_radius - R_form),
                        fill=False, linestyle="--", linewidth=0.8,
                        edgecolor=colors[i], alpha=0.25, zorder=3)
        ax.add_patch(sc)

    ax.set_title(
        f"{'Swarm' if is_swarm else 'Agent'} Trajectories\n"
        f"({n_agents} {entity_name}, {trajectories.shape[0]-1} steps · {mode_label})",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)

    from matplotlib.lines import Line2D
    handles = []
    for i in range(n_agents):
        handles.append(Line2D([0], [0], color=colors[i], linewidth=2,
                              label=f"Swarm {i}" if is_swarm else f"Agent {i}"))
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                          markeredgecolor="white", markersize=14, label="Goal (★)"))
    handles.append(mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                                  alpha=0.65, label="Obstacle"))
    if is_swarm:
        handles.append(mpatches.Patch(facecolor="gray", edgecolor="white",
                                      alpha=0.5, label=f"Formation △ (R×s)"))
        handles.append(mpatches.Patch(facecolor="gray", edgecolor="gray",
                                      alpha=0.2, label=f"Safety ○ (R×s+m)"))
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#dee2e6")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-agent trajectories (LQR or trained policy)"
    )
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--area_size", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--n_obs", type=int, default=None,
                        help="Override number of obstacles (default: 2 or training config)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="trajectories.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--swarm_lqr", action="store_true",
                        help="Run SwarmIntegrator with LQR only")
    parser.add_argument("--force_lqr", action="store_true",
                        help="Ignore trained policy, use LQR only")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Path to scenario JSON (hardcoded agent/goal/obstacle positions)")
    parser.add_argument("--exact_qp", action="store_true",
                        help="Use exact QP solver (quadprog) instead of Dykstra projection")
    parser.add_argument("--no_scale", action="store_true", default=False,
                        help="Fix scale at s=1.0 (ablation: no formation deformation)")
    parser.add_argument("--s_max_one", action="store_true", default=False,
                        help="Cap scale at s_max=1.0 (no expansion, but shrink allowed)")
    parser.add_argument("--method", type=str, default="affine_policy",
                        choices=["affine_policy", "gnn_only", "hocbf_lqr", "lqr_only"],
                        help="Control method to visualize (default: affine_policy)")

    args = parser.parse_args()

    print("Running simulation...")
    # The following arguments are not defined in the provided snippet,
    # assuming they are defined elsewhere in the actual file or are placeholders.
    # For the purpose of this edit, we'll use dummy values or assume they exist.
    # policy_net, cfg, is_swarm are not defined in the provided context.
    # We will use the original run_simulation call structure and add edge_traj.
    if args.exact_qp:
        if _QUADPROG_AVAILABLE:
            print("  [QP] Using exact solver (quadprog)")
        else:
            print("  [QP] quadprog not installed — pip install quadprog")
    result = run_simulation(
        num_agents=args.num_agents, area_size=args.area_size,
        max_steps=args.max_steps, dt=args.dt,
        n_obs=args.n_obs,  # None → use training config; int → override
        seed=args.seed, checkpoint_path=args.checkpoint,
        force_lqr=args.force_lqr, swarm_lqr=args.swarm_lqr,
        scenario_path=args.scenario,
        use_exact_qp=args.exact_qp,
        no_scale=args.no_scale,
        s_max_one=args.s_max_one,
        method=args.method,
    )
    (trajectories, goals, obstacle_info, area, comm_r, mode,
     is_swarm, R_form, r_margin, payload_traj, cable_length, scale_traj, edge_traj, lidar_traj) = result
    entity = "swarms" if is_swarm else "agents"
    print(f"  Recorded {trajectories.shape[0]} frames for "
          f"{trajectories.shape[1]} {entity}.  Mode: {mode}")

    # The 'common' dict is no longer used for create_video, but might be for plot_trajectories
    common = dict(
        trajectories=trajectories, goals=goals, obstacle_info=obstacle_info,
        area_size=area, mode=mode, comm_radius=comm_r,
        is_swarm=is_swarm, R_form=R_form, r_margin=r_margin,
        payload_traj=payload_traj, cable_length=cable_length,
        scale_traj=scale_traj,
        edge_traj=edge_traj, # Add edge_traj to common for plot_trajectories if needed
    )

    if args.save.endswith(".mp4"):
        print("Creating video...")
        create_video(
            trajectories, goals, obstacle_info, area,
            save_path=args.save, fps=args.fps, skip=args.skip,
            mode=mode, comm_radius=comm_r,
            is_swarm=is_swarm, R_form=R_form, r_margin=r_margin,
            payload_traj=payload_traj, cable_length=cable_length, scale_traj=scale_traj,
            edge_traj=edge_traj, lidar_traj=lidar_traj
        )
    else:
        print("Plotting static image...")
        plot_trajectories(**common, save_path=args.save)

    if args.png:
        png_path = args.save.replace(".mp4", ".png") if args.save.endswith(".mp4") \
            else "trajectories.png"
        plot_trajectories(**common, save_path=png_path)

    print("Done!")


if __name__ == "__main__":
    main()
