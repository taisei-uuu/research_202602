#!/usr/bin/env python3
"""
Evaluation benchmark — compare multiple swarm control methods.

Usage:
    python -m evaluate --checkpoint gcbf_swarm_checkpoint.pt
    python -m evaluate --checkpoint gcbf_swarm_checkpoint.pt --methods gcbf_hocbf,lqr_only
    python -m evaluate --checkpoint gcbf_swarm_checkpoint.pt --external my_method.py

All methods run on the SAME randomised environments (same seed per episode)
so that results are directly comparable.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch

from gcbf_plus.env.swarm_integrator import SwarmIntegrator
from gcbf_plus.nn import GCBFNetwork, PolicyNetwork
from gcbf_plus.algo.qp_solver_torch import solve_cbf_qp_batched


# ═══════════════════════════════════════════════════════════════════════
# Method registry
# ═══════════════════════════════════════════════════════════════════════

METHOD_REGISTRY: Dict[str, Type[MethodController]] = {}


def register_method(cls: Type[MethodController]) -> Type[MethodController]:
    """Decorator: add a MethodController subclass to the global registry."""
    METHOD_REGISTRY[cls.name] = cls
    return cls


class MethodController(ABC):
    """Interface every evaluation method must implement."""

    name: str = "unnamed"

    def reset(self, env: SwarmIntegrator, **kwargs: Any) -> None:
        """Called once at the start of each episode."""
        pass

    @abstractmethod
    def select_action(self, env: SwarmIntegrator) -> torch.Tensor:
        """Return control u of shape (n_agents, 2)."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Built-in methods
# ═══════════════════════════════════════════════════════════════════════

@register_method
class GCBFWithHOCBF(MethodController):
    """GCBF+ policy with QP (GNN-CBF hard + HOCBF soft)."""
    name = "gcbf_hocbf"

    def __init__(self, policy_net, gcbf_net, cfg):
        self.policy_net = policy_net
        self.gcbf_net = gcbf_net
        self.cfg = cfg

    def select_action(self, env):
        with torch.no_grad():
            graph = env._get_graph()
            u_ref = env.nominal_controller()
            pi_raw = self.policy_net(graph)
            u_cand = 2.0 * pi_raw + u_ref

            # GNN-CBF constraint via QP
            B_mat = torch.tensor(env.g_x_matrix, dtype=torch.float32)
            h_vals = self.gcbf_net(graph)
            dh_dx = _numerical_dh_dx(env, self.gcbf_net, h_vals)
            x_dot_f = torch.zeros_like(env.agent_states)
            x_dot_f[:, :2] = env.agent_states[:, 2:4]

            # HOCBF swing constraints
            A_swing, b_swing = _build_hocbf_constraints(env, self.cfg)

            u_qp = solve_cbf_qp_batched(
                u_nom=u_cand, h=h_vals, dh_dx=dh_dx,
                x_dot_f=x_dot_f, B_mat=B_mat,
                alpha=self.cfg.get("alpha", 1.0),
                u_max=env.params.get("u_max"),
                A_extra=A_swing, b_extra=b_swing,
            )
        return u_qp


@register_method
class GCBFOnly(MethodController):
    """GCBF+ policy with QP (GNN-CBF only, no HOCBF)."""
    name = "gcbf_only"

    def __init__(self, policy_net, gcbf_net, cfg):
        self.policy_net = policy_net
        self.gcbf_net = gcbf_net
        self.cfg = cfg

    def select_action(self, env):
        with torch.no_grad():
            graph = env._get_graph()
            u_ref = env.nominal_controller()
            pi_raw = self.policy_net(graph)
            u_cand = 2.0 * pi_raw + u_ref

            B_mat = torch.tensor(env.g_x_matrix, dtype=torch.float32)
            h_vals = self.gcbf_net(graph)
            dh_dx = _numerical_dh_dx(env, self.gcbf_net, h_vals)
            x_dot_f = torch.zeros_like(env.agent_states)
            x_dot_f[:, :2] = env.agent_states[:, 2:4]

            u_qp = solve_cbf_qp_batched(
                u_nom=u_cand, h=h_vals, dh_dx=dh_dx,
                x_dot_f=x_dot_f, B_mat=B_mat,
                alpha=self.cfg.get("alpha", 1.0),
                u_max=env.params.get("u_max"),
            )
        return u_qp


@register_method
class HOCBFWithLQR(MethodController):
    """LQR + HOCBF filter (no learned policy)."""
    name = "hocbf_lqr"

    def __init__(self, cfg):
        self.cfg = cfg

    def select_action(self, env):
        u = env.nominal_controller()
        u = _apply_hocbf_filter(env, u, self.cfg)
        return u


@register_method
class LQROnly(MethodController):
    """Pure LQR baseline (no safety filter)."""
    name = "lqr_only"

    def select_action(self, env):
        return env.nominal_controller()


# ═══════════════════════════════════════════════════════════════════════
# HOCBF helper functions
# ═══════════════════════════════════════════════════════════════════════

def _build_hocbf_constraints(env: SwarmIntegrator, cfg: dict):
    """Build A_swing (n,2,2) and b_swing (n,2) for QP."""
    ps = env.payload_states
    gx, gy = ps[:, 0], ps[:, 1]
    gx_dot, gy_dot = ps[:, 2], ps[:, 3]

    l = env.params["cable_length"]
    g_val = env.params["gravity"]
    c_damp = env.params.get("payload_damping", 0.03)
    gamma_max = env.params["gamma_max"]
    a1 = cfg.get("hocbf_alpha1", 2.0)
    a2 = cfg.get("hocbf_alpha2", 2.0)

    # X HOCBF
    h1_x = gamma_max**2 - gx**2
    h1_dot_x = -2 * gx * gx_dot
    h2_x = h1_dot_x + a1 * h1_x
    C_x = 2 * gx * torch.cos(gx) / l
    D_x = (2 * gx_dot**2
           - 2 * gx * (-(g_val / l) * torch.sin(gx) - c_damp * gx_dot)
           + a1 * (-2 * gx * gx_dot)
           - a2 * h2_x)

    # Y HOCBF
    h1_y = gamma_max**2 - gy**2
    h1_dot_y = -2 * gy * gy_dot
    h2_y = h1_dot_y + a1 * h1_y
    C_y = 2 * gy * torch.cos(gy) / l
    D_y = (2 * gy_dot**2
           - 2 * gy * (-(g_val / l) * torch.sin(gy) - c_damp * gy_dot)
           + a1 * (-2 * gy * gy_dot)
           - a2 * h2_y)

    n = gx.shape[0]
    A_swing = torch.zeros(n, 2, 2)
    A_swing[:, 0, 0] = -C_x
    A_swing[:, 1, 1] = -C_y
    b_swing = torch.stack([-D_x, -D_y], dim=-1)
    return A_swing, b_swing


def _apply_hocbf_filter(env: SwarmIntegrator, u: torch.Tensor, cfg: dict):
    """Lightweight per-axis HOCBF projection for LQR-based methods."""
    A_swing, b_swing = _build_hocbf_constraints(env, cfg)

    # Per-axis projection
    for axis in range(2):
        a_k = A_swing[:, axis, axis]  # -C
        b_k = b_swing[:, axis]        # -D
        # Constraint: a_k * u_axis <= b_k → -C * u <= -D → C * u >= D
        violation = a_k * u[:, axis] - b_k
        violate_mask = (violation > 0) & (a_k.abs() > 1e-6)
        u_fix = b_k / a_k  # = -D / -C = D/C
        u = u.clone()
        u[violate_mask, axis] = u_fix[violate_mask]
    return u


def _numerical_dh_dx(env: SwarmIntegrator, gcbf_net, h_vals):
    """Compute dh/dx via finite differences (no autograd needed)."""
    eps = 1e-3
    state = env.agent_states.clone()
    n, d = state.shape
    dh_dx = torch.zeros(n, d)
    for i in range(d):
        perturbed = state.clone()
        perturbed[:, i] += eps
        # Rebuild graph with perturbed state
        old_states = env.agent_states
        env.agent_states = perturbed
        graph_p = env._get_graph()
        h_p = gcbf_net(graph_p)
        env.agent_states = old_states
        dh_dx[:, i] = (h_p - h_vals) / eps
    return dh_dx


# ═══════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    method: MethodController,
    env: SwarmIntegrator,
    seed: int,
    max_steps: int = 512,
    goal_radius: float = 0.5,
) -> Dict[str, float]:
    """Run one episode, return metrics dict."""

    env.reset(seed=seed)
    method.reset(env)

    n = env.num_agents
    r_swarm = env.params["r_swarm"]
    gamma_max = env.params["gamma_max"]

    ever_collided = False
    all_gammas = []
    all_efforts = []
    min_dist = float("inf")
    goal_time = max_steps  # default if never reached

    for step in range(max_steps):
        u = method.select_action(env)
        _, info = env.step(u)

        # --- Collision check ---
        if env.unsafe_mask().any().item():
            ever_collided = True

        # --- Gamma ---
        ps = env.payload_states
        gamma_combined = torch.sqrt(ps[:, 0]**2 + ps[:, 1]**2)
        all_gammas.append(gamma_combined.detach().numpy().copy())

        # --- Control effort ---
        effort = torch.norm(u, dim=-1).mean().item()
        all_efforts.append(effort)

        # --- Min inter-agent distance ---
        if n > 1:
            pos = env.agent_states[:, :2]
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.norm(diff, dim=-1)
            dist = dist + torch.eye(n) * 1e6
            min_d = dist.min().item()
            min_dist = min(min_dist, min_d)

        # --- Goal check ---
        if goal_time == max_steps:
            goal_err = torch.norm(
                env.agent_states[:, :2] - env.goal_states[:, :2], dim=-1
            )
            if (goal_err < goal_radius).all():
                goal_time = step + 1

        if info["done"]:
            break

    # --- Success ---
    goal_err_final = torch.norm(
        env.agent_states[:, :2] - env.goal_states[:, :2], dim=-1
    )
    success = (goal_err_final < goal_radius).all().item()

    # --- Gamma stats ---
    all_gammas = np.concatenate(all_gammas)
    gamma_mean = float(np.mean(all_gammas))
    gamma_max_val = float(np.max(all_gammas))
    gamma_viol = float(np.mean(all_gammas > gamma_max))

    return {
        "success": float(success),
        "safety": float(not ever_collided),
        "gamma_mean": gamma_mean,
        "gamma_max": gamma_max_val,
        "gamma_viol": gamma_viol,
        "min_dist": min_dist if min_dist < 1e5 else 0.0,
        "effort": float(np.mean(all_efforts)),
        "goal_time": float(goal_time),
    }


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint loader
# ═══════════════════════════════════════════════════════════════════════

def load_checkpoint(path: str):
    """Load trained policy/cbf nets and config from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    gcbf_net = GCBFNetwork(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        action_dim=cfg["action_dim"],
        n_agents=cfg["num_agents"],
    )
    gcbf_net.load_state_dict(ckpt["gcbf_net"])
    gcbf_net.eval()

    policy_net = PolicyNetwork(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        action_dim=cfg["action_dim"],
        n_agents=cfg["num_agents"],
    )
    policy_net.load_state_dict(ckpt["policy_net"])
    policy_net.eval()

    return policy_net, gcbf_net, cfg


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate swarm control methods")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained GCBF+ checkpoint (.pt)")
    parser.add_argument("--n_episodes", type=int, default=50,
                        help="Number of random episodes per method")
    parser.add_argument("--max_steps", type=int, default=512,
                        help="Max steps per episode")
    parser.add_argument("--goal_radius", type=float, default=0.5,
                        help="Radius to consider goal reached")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method names (default: all)")
    parser.add_argument("--external", type=str, nargs="*", default=[],
                        help="External .py files with @register_method classes")
    args = parser.parse_args()

    # ── Load external methods ──
    for ext_path in args.external:
        spec = importlib.util.spec_from_file_location("ext_method", ext_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"  Loaded external methods from: {ext_path}")

    # ── Load checkpoint ──
    policy_net, gcbf_net, cfg = load_checkpoint(args.checkpoint)
    print(f"  Loaded checkpoint: {args.checkpoint}")
    print(f"  Config: agents={cfg['num_agents']} area={cfg['area_size']} "
          f"n_obs={cfg['n_obs']}")

    # ── Build environment template ──
    env = SwarmIntegrator(
        num_agents=cfg["num_agents"],
        area_size=cfg["area_size"],
        dt=cfg["dt"],
        max_steps=args.max_steps,
        params={
            "n_obs": cfg["n_obs"],
            "comm_radius": cfg["comm_radius"],
            "R_form": cfg.get("R_form", 0.5),
        },
    )

    # ── Instantiate methods ──
    available_methods: Dict[str, MethodController] = {}

    # Built-in methods that need checkpoint
    if "gcbf_hocbf" in METHOD_REGISTRY:
        available_methods["gcbf_hocbf"] = GCBFWithHOCBF(policy_net, gcbf_net, cfg)
    if "gcbf_only" in METHOD_REGISTRY:
        available_methods["gcbf_only"] = GCBFOnly(policy_net, gcbf_net, cfg)
    if "hocbf_lqr" in METHOD_REGISTRY:
        available_methods["hocbf_lqr"] = HOCBFWithLQR(cfg)
    if "lqr_only" in METHOD_REGISTRY:
        available_methods["lqr_only"] = LQROnly()

    # Add any external methods (they must have a no-arg constructor or
    # accept **kwargs)
    for name, cls in METHOD_REGISTRY.items():
        if name not in available_methods:
            try:
                available_methods[name] = cls()
            except TypeError:
                try:
                    available_methods[name] = cls(cfg=cfg)
                except TypeError:
                    print(f"  WARNING: Could not instantiate method '{name}', skipping")

    # Filter by --methods
    if args.methods:
        selected = [m.strip() for m in args.methods.split(",")]
        available_methods = {k: v for k, v in available_methods.items() if k in selected}

    if not available_methods:
        print("ERROR: No methods to evaluate!")
        return

    method_names = list(available_methods.keys())
    print(f"  Methods: {method_names}")
    print(f"  Episodes: {args.n_episodes}, Max steps: {args.max_steps}")
    print("=" * 100)

    # ── Run evaluation ──
    results: Dict[str, List[Dict[str, float]]] = {m: [] for m in method_names}
    seeds = list(range(args.seed, args.seed + args.n_episodes))

    t_start = time.time()
    for ep_idx, seed in enumerate(seeds):
        for m_name in method_names:
            metrics = evaluate_episode(
                method=available_methods[m_name],
                env=env,
                seed=seed,
                max_steps=args.max_steps,
                goal_radius=args.goal_radius,
            )
            results[m_name].append(metrics)

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            elapsed = time.time() - t_start
            print(f"  Episode {ep_idx + 1}/{args.n_episodes} done ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\n  Evaluation complete in {elapsed:.1f}s\n")

    # ── Aggregate & display ──
    metric_keys = ["success", "safety", "gamma_mean", "gamma_max",
                   "gamma_viol", "min_dist", "effort", "goal_time"]
    headers = ["Method", "Success↑", "Safety↑", "γ_mean↓", "γ_max↓",
               "γ_viol↓", "MinDist↑", "Effort↓", "GoalTime↓"]
    col_widths = [16, 10, 10, 10, 10, 10, 10, 10, 10]

    # Header
    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)

    for m_name in method_names:
        ep_results = results[m_name]
        agg = {}
        for k in metric_keys:
            vals = [r[k] for r in ep_results]
            agg[k] = np.mean(vals)

        row = [
            m_name.center(col_widths[0]),
            f"{agg['success']*100:.1f}%".center(col_widths[1]),
            f"{agg['safety']*100:.1f}%".center(col_widths[2]),
            f"{agg['gamma_mean']:.3f}".center(col_widths[3]),
            f"{agg['gamma_max']:.3f}".center(col_widths[4]),
            f"{agg['gamma_viol']*100:.1f}%".center(col_widths[5]),
            f"{agg['min_dist']:.3f}".center(col_widths[6]),
            f"{agg['effort']:.3f}".center(col_widths[7]),
            f"{agg['goal_time']:.0f}".center(col_widths[8]),
        ]
        print(" | ".join(row))

    print()


if __name__ == "__main__":
    main()
