#!/usr/bin/env python3
"""
Evaluation benchmark — compare swarm control methods.

Usage:
    python -m evaluate --checkpoint affine_swarm_checkpoint.pt
    python -m evaluate --checkpoint affine_swarm_checkpoint.pt --methods affine_policy,lqr_only

Methods:
  - affine_policy: Trained affine-transform policy + analytical QP
  - hocbf_lqr:    LQR + HOCBF filter (no learned policy)
  - lqr_only:     Pure LQR baseline (no safety filter)
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod

import numpy as np
import torch

from gcbf_plus.env import SwarmIntegrator
from gcbf_plus.nn import PolicyNetwork
from gcbf_plus.algo.affine_qp_solver import solve_affine_qp

try:
    import quadprog
    _QUADPROG_AVAILABLE = True
except ImportError:
    _QUADPROG_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# Exact QP solver (quadprog) — ported from visualize.py
# ═══════════════════════════════════════════════════════════════════════

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
    G = np.diag([1.0, 1.0, 1.0, p, p]) + np.eye(5) * 1e-8
    a_vec = np.array([u_nom_i[0], u_nom_i[1], u_nom_i[2], 0.0, 0.0], dtype=np.float64)

    C_rows = []
    b_vals = []

    def add(row, b):
        C_rows.append(np.array(row, dtype=np.float64))
        b_vals.append(float(b))

    # Scale CBF (2nd order HOCBF)
    alpha_sum_s = alpha1_scale + alpha2_scale
    alpha_prod_s = alpha1_scale * alpha2_scale
    lb = -alpha_sum_s * sd_i - alpha_prod_s * (s_i - s_min)
    ub = -alpha_sum_s * sd_i + alpha_prod_s * (s_max - s_i)
    add([0, 0,  1, 0, 0],  lb)
    add([0, 0, -1, 0, 0], -ub)

    # Obstacle CBF (2nd order HOCBF)
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

    # Agent-Agent CBF (2nd order HOCBF + Reciprocal CA)
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
            rhs = (h_ddot_drift + (a1 + a2) * h_dot + (a1 * a2) * h) * 0.5
            add([A_cx, A_cy, A_as, 0, 0], -rhs)

    # Payload HOCBF (soft, with slack delta)
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

        h_x = gamma_dyn ** 2 - gx ** 2
        h_dot_x = 2.0 * gamma_dyn * gamma_dyn_dot - 2.0 * gx * gx_dot
        h_ddot_drift_x = (extra_drift
                          - 2.0 * gx_dot ** 2
                          + 2.0 * gx * g_val / l * np.sin(gx)
                          + 2.0 * gx * c_damp * gx_dot)
        C_x = (2.0 * gx * np.cos(gx)) / l
        rhs_x = h_ddot_drift_x + alpha_sum_h * h_dot_x + alpha_prod_h * h_x
        add([C_x, 0, C_s_h, 1, 0], -rhs_x)

        h_y = gamma_dyn ** 2 - gy ** 2
        h_dot_y = 2.0 * gamma_dyn * gamma_dyn_dot - 2.0 * gy * gy_dot
        h_ddot_drift_y = (extra_drift
                          - 2.0 * gy_dot ** 2
                          + 2.0 * gy * g_val / l * np.sin(gy)
                          + 2.0 * gy * c_damp * gy_dot)
        C_y = (2.0 * gy * np.cos(gy)) / l
        rhs_y = h_ddot_drift_y + alpha_sum_h * h_dot_y + alpha_prod_h * h_y
        add([0, C_y, C_s_h, 0, 1], -rhs_y)

        add([0, 0, 0, 1, 0], 0.0)
        add([0, 0, 0, 0, 1], 0.0)

    # Box constraint on translation
    if u_max is not None:
        add([ 1, 0, 0, 0, 0], -u_max)
        add([-1, 0, 0, 0, 0], -u_max)
        add([ 0, 1, 0, 0, 0], -u_max)
        add([ 0,-1, 0, 0, 0], -u_max)

    C_mat = np.array(C_rows, dtype=np.float64).T  # (5, n_constraints)
    b_vec = np.array(b_vals, dtype=np.float64)

    try:
        sol = quadprog.solve_qp(G, a_vec, C_mat, b_vec)
        return sol[0][:3].astype(np.float32), False
    except Exception:
        return u_nom_i[:3].astype(np.float32), True


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
) -> tuple:
    """Exact QP solver (quadprog). Loops over agents.
    Returns: (u_tensor, n_infeasible) where n_infeasible is the number of agents
    for which the QP was infeasible and fell back to u_nom."""
    N = u_nom.shape[0]
    results = []
    n_infeasible = 0

    for i in range(N):
        u_nom_i = u_nom[i].numpy().astype(np.float64)
        s_i = float(s[i])
        sd_i = float(s_dot[i])
        pos_i = agent_pos[i].numpy().astype(np.float64)
        vel_i = agent_vel[i].numpy().astype(np.float64)

        if obs_hits is not None:
            hits_i = obs_hits[i].numpy()
            valid = ~np.any(hits_i > 1e5, axis=-1)
            obs_i = hits_i[valid].astype(np.float64) if valid.any() else None
        else:
            obs_i = None

        op  = other_agent_pos[i].numpy().astype(np.float64)  if other_agent_pos  is not None else None
        ov  = other_agent_vel[i].numpy().astype(np.float64)  if other_agent_vel  is not None else None
        os_ = other_agent_s[i].numpy().astype(np.float64)    if other_agent_s    is not None else None
        osd = other_agent_s_dot[i].numpy().astype(np.float64) if other_agent_s_dot is not None else None
        pay_i = payload_states[i].numpy().astype(np.float64) if payload_states is not None else None

        sol_i, infeasible = _solve_qp_exact_single(
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
        if infeasible:
            n_infeasible += 1

    return torch.tensor(np.stack(results), dtype=torch.float32), n_infeasible


# ═══════════════════════════════════════════════════════════════════════
# Method registry
# ═══════════════════════════════════════════════════════════════════════

METHOD_REGISTRY: Dict[str, Type["MethodController"]] = {}

def register_method(cls: Type["MethodController"]):
    METHOD_REGISTRY[cls.name] = cls
    return cls


class MethodController(ABC):
    """Interface every evaluation method must implement."""
    name: str = "unnamed"

    def reset(self, env: SwarmIntegrator, **kwargs: Any):
        self.infeasible_count: int = 0
        self.total_qp_calls: int = 0

    @abstractmethod
    def select_action(self, env: SwarmIntegrator) -> torch.Tensor:
        ...


# ═══════════════════════════════════════════════════════════════════════
# Built-in methods
# ═══════════════════════════════════════════════════════════════════════

@register_method
class AffinePolicy(MethodController):
    """Trained hierarchical velocity-command policy with analytical QP."""
    name = "affine_policy"

    def __init__(self, policy_net, cfg, no_scale: bool = False, use_exact_qp: bool = False):
        self.policy_net = policy_net
        self.cfg = cfg
        self.no_scale = no_scale
        self.use_exact_qp = use_exact_qp

    def select_action(self, env):
        with torch.no_grad():
            graph = env._get_graph()
            cfg = self.cfg

            # GNN → tanh → scale to physical acceleration offset
            pi_tanh = self.policy_net(graph)  # (n, 3) already tanh'd
            a_max_gnn   = 1.0   # m/s²
            a_max_gnn_s = 0.5   # s⁻²
            pi_scaled = pi_tanh.clone()
            pi_scaled[:, :2] *= a_max_gnn
            pi_scaled[:, 2]  *= a_max_gnn_s
            if self.no_scale:
                pi_scaled[:, 2] = 0.0

            # PD nominal + GNN acceleration offset
            pos       = env.agent_states[:, :2]
            goal_pos  = env.goal_states[:, :2]
            v_current = env.agent_states[:, 2:4]
            s_current    = env.scale_states[:, 0]
            s_dot_current = env.scale_states[:, 1]

            v_max_cfg = cfg.get("v_max", 1.0)
            K_pos = cfg.get("K_pos", 0.5)
            K_v   = cfg.get("K_v",   2.0)
            K_s   = cfg.get("K_s",   2.0)
            s_max = env.params.get("s_max", 1.5)

            v_ref     = torch.clamp(K_pos * (goal_pos - pos), -v_max_cfg, v_max_cfg)
            s_dot_ref = 1.0 * (s_max - s_current)
            a_trans   = K_v * (v_ref - v_current) + pi_scaled[:, :2]
            a_s       = K_s * (s_dot_ref - s_dot_current) + pi_scaled[:, 2]
            u_nom     = torch.cat([a_trans, a_s.unsqueeze(-1)], dim=-1)

            _u_max = env.params.get("u_max")
            if _u_max is not None:
                a_max_t = _u_max * 0.7
                a_max_s = _u_max * 0.3
                u_nom[:, :2] = u_nom[:, :2].clamp(-a_max_t, a_max_t)
                u_nom[:, 2]  = u_nom[:,  2].clamp(-a_max_s, a_max_s)

            # QP — LiDAR hits + agent-agent info
            sc = env.scale_states[:, 0]
            sd = env.scale_states[:, 1]
            ps = env.payload_states
            n  = env.num_agents

            lidar_hits = env.get_lidar_hits(num_beams=32)[..., :2]  # (n, 32, 2)

            if n > 1:
                idx  = torch.arange(n)
                mask = idx.unsqueeze(0) != idx.unsqueeze(1)
                other_pos = pos.unsqueeze(0).expand(n, n, 2)[mask].reshape(n, n-1, 2)
                other_vel = v_current.unsqueeze(0).expand(n, n, 2)[mask].reshape(n, n-1, 2)
                other_s   = sc.unsqueeze(0).expand(n, n)[mask].reshape(n, n-1)
                other_sd  = sd.unsqueeze(0).expand(n, n)[mask].reshape(n, n-1)
            else:
                other_pos = other_vel = other_s = other_sd = None

            if self.use_exact_qp and _QUADPROG_AVAILABLE:
                u_qp, n_inf = solve_affine_qp_exact(
                    u_nom=u_nom,
                    obs_hits=lidar_hits,
                    agent_pos=pos, agent_vel=v_current,
                    s=sc, s_dot=sd,
                    other_agent_pos=other_pos, other_agent_vel=other_vel,
                    other_agent_s=other_s, other_agent_s_dot=other_sd,
                    payload_states=ps,
                    R_form=env.params["R_form"],
                    r_margin=env.params["r_margin"],
                    s_min=env.params["s_min"], s_max=env.params["s_max"],
                    cable_length=env.params["cable_length"],
                    gravity=env.params["gravity"],
                    payload_damping=env.params["payload_damping"],
                    u_max=_u_max,
                )
                self.infeasible_count += n_inf
                self.total_qp_calls += n
            else:
                if self.use_exact_qp and not _QUADPROG_AVAILABLE:
                    print("  [WARNING] quadprog not installed — falling back to Dykstra QP")
                u_qp = solve_affine_qp(
                    u_nom=u_nom,
                    obs_hits=lidar_hits,
                    agent_pos=pos, agent_vel=v_current,
                    s=sc, s_dot=sd,
                    other_agent_pos=other_pos, other_agent_vel=other_vel,
                    other_agent_s=other_s, other_agent_s_dot=other_sd,
                    R_form=env.params["R_form"],
                    r_margin=env.params["r_margin"],
                    mass=env.params["mass"],
                    s_min=env.params["s_min"], s_max=env.params["s_max"],
                    payload_states=ps,
                    cable_length=env.params["cable_length"],
                    gravity=env.params["gravity"],
                    gamma_min=env.params["gamma_min"],
                    gamma_max_full=env.params["gamma_max_full"],
                    payload_damping=env.params["payload_damping"],
                    u_max=_u_max,
                )
            return u_qp


@register_method
class HOCBFWithLQR(MethodController):
    """LQR velocity tracking + HOCBF filter (no learned policy)."""
    name = "hocbf_lqr"

    def __init__(self, cfg=None, no_scale: bool = False, use_exact_qp: bool = False):
        self.cfg = cfg or {}
        self.no_scale = no_scale
        self.use_exact_qp = use_exact_qp

    def select_action(self, env):
        s_max = env.params.get("s_max", 1.5)
        s_dot_target = 1.0 * (s_max - env.scale_states[:, 0])
        u_ref = env.nominal_controller(s_dot_target=s_dot_target)
        if self.no_scale:
            u_ref[:, 2] = 0.0
        with torch.no_grad():
            pos = env.agent_states[:, :2]
            vel = env.agent_states[:, 2:4]
            sc  = env.scale_states[:, 0]
            sd  = env.scale_states[:, 1]
            ps  = env.payload_states
            n   = env.num_agents

            lidar_hits = env.get_lidar_hits(num_beams=32)[..., :2]  # (n, 32, 2)

            if n > 1:
                idx  = torch.arange(n)
                mask = idx.unsqueeze(0) != idx.unsqueeze(1)
                other_pos = pos.unsqueeze(0).expand(n, n, 2)[mask].reshape(n, n-1, 2)
                other_vel = vel.unsqueeze(0).expand(n, n, 2)[mask].reshape(n, n-1, 2)
                other_s   = sc.unsqueeze(0).expand(n, n)[mask].reshape(n, n-1)
                other_sd  = sd.unsqueeze(0).expand(n, n)[mask].reshape(n, n-1)
            else:
                other_pos = other_vel = other_s = other_sd = None

            if self.use_exact_qp and _QUADPROG_AVAILABLE:
                u_qp, n_inf = solve_affine_qp_exact(
                    u_nom=u_ref,
                    obs_hits=lidar_hits,
                    agent_pos=pos, agent_vel=vel,
                    s=sc, s_dot=sd,
                    other_agent_pos=other_pos, other_agent_vel=other_vel,
                    other_agent_s=other_s, other_agent_s_dot=other_sd,
                    payload_states=ps,
                    R_form=env.params["R_form"],
                    r_margin=env.params["r_margin"],
                    s_min=env.params["s_min"], s_max=env.params["s_max"],
                    cable_length=env.params["cable_length"],
                    gravity=env.params["gravity"],
                    payload_damping=env.params["payload_damping"],
                    u_max=env.params.get("u_max"),
                )
                self.infeasible_count += n_inf
                self.total_qp_calls += n
            else:
                if self.use_exact_qp and not _QUADPROG_AVAILABLE:
                    print("  [WARNING] quadprog not installed — falling back to Dykstra QP")
                u_qp = solve_affine_qp(
                    u_nom=u_ref,
                    obs_hits=lidar_hits,
                    agent_pos=pos, agent_vel=vel,
                    s=sc, s_dot=sd,
                    other_agent_pos=other_pos, other_agent_vel=other_vel,
                    other_agent_s=other_s, other_agent_s_dot=other_sd,
                    R_form=env.params["R_form"],
                    r_margin=env.params["r_margin"],
                    mass=env.params["mass"],
                    s_min=env.params["s_min"], s_max=env.params["s_max"],
                    payload_states=ps,
                    cable_length=env.params["cable_length"],
                    gravity=env.params["gravity"],
                    gamma_min=env.params["gamma_min"],
                    gamma_max_full=env.params["gamma_max_full"],
                    payload_damping=env.params["payload_damping"],
                    u_max=env.params.get("u_max"),
                )
            return u_qp


@register_method
class LQROnly(MethodController):
    """Pure LQR baseline (no safety filter)."""
    name = "lqr_only"

    def select_action(self, env):
        return env.nominal_controller()


# ═══════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    method: MethodController,
    env: SwarmIntegrator,
    seed: int,
    max_steps: int = 512,
    goal_radius: float = 0.5,
):
    """Run one episode, return metrics dict."""
    env.reset(seed=seed)
    method.reset(env)

    n = env.num_agents
    total_control_effort = 0.0
    min_dist = float("inf")
    collision_count = 0
    goal_reached_step = None
    max_gamma = 0.0
    gamma_values = []
    scale_values = []

    for t in range(max_steps):
        action = method.select_action(env)
        _, info = env.step(action)

        # Metrics
        total_control_effort += action[:, :2].norm(dim=-1).sum().item()

        # Min inter-agent distance
        pos = env.agent_states[:, :2]
        if n > 1:
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.norm(diff, dim=-1)
            dist = dist + torch.eye(n) * 1e6
            min_dist = min(min_dist, dist.min().item())

        # Collision check
        if env.unsafe_mask().any():
            collision_count += 1

        # Scale tracking
        scale_values.append(env.scale_states[:, 0].clone())

        # Payload swing
        gamma = torch.sqrt(env.payload_states[:, 0]**2 + env.payload_states[:, 1]**2)
        max_gamma = max(max_gamma, gamma.max().item())
        gamma_values.append(gamma.mean().item())

        # Goal distance
        goal_dist = torch.norm(env.agent_states[:, :2] - env.goal_states[:, :2], dim=-1)
        if goal_dist.max() < goal_radius and goal_reached_step is None:
            goal_reached_step = t

        if info["done"]:
            break

    # Final goal distance
    final_goal_dist = torch.norm(env.agent_states[:, :2] - env.goal_states[:, :2], dim=-1)

    scale_stack = torch.stack(scale_values)  # (T, n)

    # Infeasibility (only populated when use_exact_qp=True)
    total_calls = getattr(method, "total_qp_calls", 0)
    infeasible   = getattr(method, "infeasible_count", 0)
    infeasibility_rate = infeasible / total_calls if total_calls > 0 else float("nan")

    return {
        "success": goal_reached_step is not None,
        "collision_count": collision_count,
        "safety_rate": 1.0 - (collision_count / (t + 1)),
        "min_distance": min_dist if min_dist < 1e5 else float("nan"),
        "control_effort": total_control_effort,
        "goal_time": goal_reached_step,
        "final_goal_dist": final_goal_dist.mean().item(),
        "max_gamma": max_gamma,
        "mean_gamma": float(np.mean(gamma_values)) if gamma_values else 0.0,
        "scale_mean": scale_stack.mean().item(),
        "scale_min": scale_stack.min().item(),
        "scale_max": scale_stack.max().item(),
        "qp_infeasible_count": infeasible,
        "qp_total_calls": total_calls,
        "qp_infeasibility_rate": infeasibility_rate,
    }


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint loader
# ═══════════════════════════════════════════════════════════════════════

def load_checkpoint(path: str):
    """Load trained policy and config from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    policy_net = PolicyNetwork(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        action_dim=cfg["action_dim"],
        n_agents=cfg["num_agents"],
    )
    policy_net.load_state_dict(ckpt["policy_net"])
    policy_net.eval()

    return policy_net, cfg


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate swarm control methods")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--methods", type=str, default="affine_policy,hocbf_lqr,lqr_only",
                        help="Comma-separated method names")
    parser.add_argument("--n_obs", type=int, default=None,
                        help="Override number of obstacles (default: use training config)")
    parser.add_argument("--exact_qp", action="store_true", default=False,
                        help="Use exact QP solver (quadprog) instead of Dykstra projection")
    parser.add_argument("--no_scale", action="store_true", default=False,
                        help="Fix scale at s=1.0 (ablation: no formation deformation)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--area_size", type=float, default=None,
                        help="Override arena size (default: use training config)")
    parser.add_argument("--seed_start", type=int, default=1000)
    parser.add_argument("--save_json", type=str, default="eval_results.json")
    args = parser.parse_args()

    if args.exact_qp:
        if _QUADPROG_AVAILABLE:
            print("  [QP] Using exact solver (quadprog)")
        else:
            print("  [QP] quadprog not installed — pip install quadprog")

    policy_net, cfg = load_checkpoint(args.checkpoint)
    method_names = [m.strip() for m in args.methods.split(",")]

    n_obs = args.n_obs if args.n_obs is not None else cfg["n_obs"]
    if args.n_obs is not None:
        print(f"  [n_obs] Overriding training config ({cfg['n_obs']}) → {n_obs}")
    if args.no_scale:
        print("  [no_scale] Formation scale fixed at s=1.0")

    area_size = args.area_size if args.area_size is not None else cfg["area_size"]
    if args.area_size is not None:
        print(f"  [area_size] Overriding training config ({cfg['area_size']}) → {area_size}")

    env = SwarmIntegrator(
        num_agents=cfg["num_agents"],
        area_size=area_size,
        dt=cfg.get("dt", 0.03),
        params={
            "n_obs": n_obs,
            "use_payload": cfg.get("use_payload", True),
            **({"s_min": 1.0, "s_max": 1.0} if args.no_scale else {}),
            "comm_radius": cfg["comm_radius"],
            "R_form": cfg.get("R_form", 0.5),
            "r_margin": cfg.get("r_margin", 0.2),
            "s_min": cfg.get("s_min", 0.4),
            "s_max": cfg.get("s_max", 1.5),
            "mass": cfg.get("mass", 0.1),
            "u_max": cfg.get("u_max", 0.3),
            "v_max": cfg.get("v_max", 1.0),
            "cable_length": cfg.get("cable_length", 1.0),
            "gravity": cfg.get("gravity", 9.81),
            "gamma_min": cfg.get("gamma_min", 0.2),
            "gamma_max_full": cfg.get("gamma_max_full", 0.75),
            "payload_damping": cfg.get("payload_damping", 0.03),
        },
    )

    results: Dict[str, Any] = {}

    for mname in method_names:
        if mname not in METHOD_REGISTRY:
            print(f"  [WARN] Unknown method: {mname}, skipping")
            continue

        cls = METHOD_REGISTRY[mname]
        if mname == "affine_policy":
            method = cls(policy_net=policy_net, cfg=cfg, no_scale=args.no_scale, use_exact_qp=args.exact_qp)
        elif mname == "hocbf_lqr":
            method = cls(cfg=cfg, no_scale=args.no_scale, use_exact_qp=args.exact_qp)
        else:
            method = cls()

        print(f"\n  Evaluating: {mname} ({args.episodes} episodes)")
        metrics_list = []
        for ep in range(args.episodes):
            seed = args.seed_start + ep
            m = evaluate_episode(method, env, seed=seed, max_steps=args.max_steps)
            metrics_list.append(m)

        # Aggregate
        agg = {}
        for key in metrics_list[0]:
            vals = [m[key] for m in metrics_list if m[key] is not None]
            if isinstance(vals[0], bool):
                agg[key] = sum(vals) / len(vals)
            elif isinstance(vals[0], (int, float)):
                vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                if vals:
                    agg[f"{key}_mean"] = np.mean(vals)
                    agg[f"{key}_std"] = np.std(vals)

        results[mname] = agg
        print(f"    Success: {agg.get('success', 0):.1%}")
        print(f"    Safety:  {agg.get('safety_rate_mean', 0):.3f}")
        print(f"    FinalGoalDist: {agg.get('final_goal_dist_mean', 0):.3f}")
        print(f"    MaxGamma: {agg.get('max_gamma_mean', 0):.3f}")
        print(f"    Scale: [{agg.get('scale_min_mean', 1):.2f}, {agg.get('scale_max_mean', 1):.2f}]")
        if args.exact_qp and _QUADPROG_AVAILABLE:
            inf_rate = agg.get("qp_infeasibility_rate_mean", float("nan"))
            inf_total = agg.get("qp_infeasible_count_mean", 0)
            print(f"    QP Infeasibility: {inf_rate:.1%}  (avg {inf_total:.1f} infeasible solves/episode)")

    # Save
    with open(args.save_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
