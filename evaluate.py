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
        pass

    @abstractmethod
    def select_action(self, env: SwarmIntegrator) -> torch.Tensor:
        ...


# ═══════════════════════════════════════════════════════════════════════
# Built-in methods
# ═══════════════════════════════════════════════════════════════════════

@register_method
class AffinePolicy(MethodController):
    """Trained affine-transform policy with analytical QP."""
    name = "affine_policy"

    def __init__(self, policy_net, cfg):
        self.policy_net = policy_net
        self.cfg = cfg

    def select_action(self, env):
        with torch.no_grad():
            graph = env._get_graph()
            u_ref = env.nominal_controller()
            pi_raw = self.policy_net(graph)
            u_at = u_ref + pi_raw

            # QP filter
            pos = env.agent_states[:, :2]
            vel = env.agent_states[:, 2:4]
            sc = env.scale_states[:, 0]
            sd = env.scale_states[:, 1]
            ps = env.payload_states
            n = env.num_agents

            n_obs_env = len(env._obstacles)
            if n_obs_env > 0:
                oc = torch.stack([obs.center for obs in env._obstacles]).unsqueeze(0).expand(n, -1, -1)
                ohs = torch.stack([obs.half_size for obs in env._obstacles]).unsqueeze(0).expand(n, -1, -1)
            else:
                oc = None
                ohs = None

            u_qp = solve_affine_qp(
                u_at=u_at,
                obs_centers=oc, obs_half_sizes=ohs,
                agent_pos=pos, agent_vel=vel,
                s=sc, s_dot=sd,
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
class HOCBFWithLQR(MethodController):
    """LQR + HOCBF filter (no learned policy)."""
    name = "hocbf_lqr"

    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def select_action(self, env):
        u_ref = env.nominal_controller()
        with torch.no_grad():
            pos = env.agent_states[:, :2]
            vel = env.agent_states[:, 2:4]
            sc = env.scale_states[:, 0]
            sd = env.scale_states[:, 1]
            ps = env.payload_states
            n = env.num_agents

            n_obs_env = len(env._obstacles)
            if n_obs_env > 0:
                oc = torch.stack([obs.center for obs in env._obstacles]).unsqueeze(0).expand(n, -1, -1)
                ohs = torch.stack([obs.half_size for obs in env._obstacles]).unsqueeze(0).expand(n, -1, -1)
            else:
                oc = None
                ohs = None

            u_qp = solve_affine_qp(
                u_at=u_ref,
                obs_centers=oc, obs_half_sizes=ohs,
                agent_pos=pos, agent_vel=vel,
                s=sc, s_dot=sd,
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

        # Goal distance
        goal_dist = torch.norm(env.agent_states[:, :2] - env.goal_states[:, :2], dim=-1)
        if goal_dist.max() < goal_radius and goal_reached_step is None:
            goal_reached_step = t

        if info["done"]:
            break

    # Final goal distance
    final_goal_dist = torch.norm(env.agent_states[:, :2] - env.goal_states[:, :2], dim=-1)

    scale_stack = torch.stack(scale_values)  # (T, n)
    return {
        "success": goal_reached_step is not None,
        "collision_count": collision_count,
        "safety_rate": 1.0 - (collision_count / (t + 1)),
        "min_distance": min_dist if min_dist < 1e5 else float("nan"),
        "control_effort": total_control_effort,
        "goal_time": goal_reached_step,
        "final_goal_dist": final_goal_dist.mean().item(),
        "max_gamma": max_gamma,
        "scale_mean": scale_stack.mean().item(),
        "scale_min": scale_stack.min().item(),
        "scale_max": scale_stack.max().item(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint loader
# ═══════════════════════════════════════════════════════════════════════

def load_checkpoint(path: str):
    """Load trained policy and config from checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--seed_start", type=int, default=1000)
    parser.add_argument("--save_json", type=str, default="eval_results.json")
    args = parser.parse_args()

    policy_net, cfg = load_checkpoint(args.checkpoint)
    method_names = [m.strip() for m in args.methods.split(",")]

    env = SwarmIntegrator(
        num_agents=cfg["num_agents"],
        area_size=cfg["area_size"],
        params={
            "n_obs": cfg["n_obs"],
            "comm_radius": cfg["comm_radius"],
            "R_form": cfg.get("R_form", 0.5),
            "r_margin": cfg.get("r_margin", 0.2),
            "s_min": cfg.get("s_min", 0.4),
            "s_max": cfg.get("s_max", 1.5),
            "cable_length": cfg.get("cable_length", 1.0),
            "gamma_min": cfg.get("gamma_min", 0.2),
            "gamma_max_full": cfg.get("gamma_max_full", 0.75),
        },
    )

    results: Dict[str, Any] = {}

    for mname in method_names:
        if mname not in METHOD_REGISTRY:
            print(f"  [WARN] Unknown method: {mname}, skipping")
            continue

        cls = METHOD_REGISTRY[mname]
        if mname == "affine_policy":
            method = cls(policy_net=policy_net, cfg=cfg)
        elif mname == "hocbf_lqr":
            method = cls(cfg=cfg)
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

    # Save
    with open(args.save_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
