#!/usr/bin/env python3
"""
GCBF+ Training Loop.

Trains the GCBF network h(x) and Policy network π(x) jointly using:
    - Autograd Lie derivative for ḣ
    - CBF-QP (cvxpy) for policy target generation
    - Composite loss  L = L_CBF + L_ctrl  (Eqs. 19-22)

Usage:
    python -m gcbf_plus.train --num_agents 4 --num_steps 2000
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np
import torch

from gcbf_plus.env import DoubleIntegrator
from gcbf_plus.nn import GCBFNetwork, PolicyNetwork
from gcbf_plus.algo.loss import compute_lie_derivative, compute_loss
from gcbf_plus.algo.qp_solver import solve_cbf_qp


def train(
    num_agents: int = 4,
    area_size: float = 10.0,
    num_steps: int = 2000,
    batch_size: int = 8,
    lr_cbf: float = 3e-5,
    lr_actor: float = 3e-5,
    alpha: float = 1.0,
    eps: float = 0.02,
    coef_safe: float = 1.0,
    coef_unsafe: float = 1.0,
    coef_h_dot: float = 0.2,
    coef_action: float = 0.001,
    max_grad_norm: float = 2.0,
    log_interval: int = 50,
    seed: int = 0,
) -> Dict[str, list]:
    """
    Train the GCBF+ networks.

    Returns
    -------
    history : dict of lists with logged metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Environment ----
    env = DoubleIntegrator(
        num_agents=num_agents,
        area_size=area_size,
        params={"comm_radius": 1.0, "n_obs": 4},
    )

    # ---- Networks ----
    gcbf_net = GCBFNetwork(
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        n_agents=num_agents,
    )
    policy_net = PolicyNetwork(
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        action_dim=env.action_dim,
        n_agents=num_agents,
    )

    # ---- Optimizers ----
    optim_cbf = torch.optim.Adam(gcbf_net.parameters(), lr=lr_cbf)
    optim_actor = torch.optim.Adam(policy_net.parameters(), lr=lr_actor)

    # ---- Dynamics matrices (constant for Double Integrator) ----
    B_mat = env._B   # (state_dim, action_dim)

    # ---- Training history ----
    history: Dict[str, list] = {
        "step": [], "loss/total": [], "loss/safe": [], "loss/unsafe": [],
        "loss/h_dot": [], "loss/action": [], "acc/safe": [], "acc/unsafe": [],
        "acc/h_dot": [],
    }

    print("=" * 60)
    print(f" GCBF+ Training  |  agents={num_agents}  steps={num_steps}")
    print("=" * 60)
    t_start = time.time()

    for step in range(1, num_steps + 1):

        # ---- (1) Sample a batch of random initial configurations ----
        batch_loss = 0.0
        batch_info: Dict[str, float] = {}

        for _ in range(batch_size):
            env.reset(seed=None)  # fresh random seed each sample

            # ---- (2) Build differentiable graph ----
            agent_states = env.agent_states.clone().requires_grad_(True)
            graph = env.build_graph_differentiable(agent_states)

            # ---- (3) Forward pass: h and π ----
            h_all = gcbf_net(graph)           # (n_agents, 1)
            h = h_all.squeeze(-1)             # (n_agents,)
            pi_action = policy_net(graph)     # (n_agents, action_dim)

            # ---- (4) Lie derivative via autograd ----
            x_dot = env.state_dot(agent_states, pi_action)  # (n, 4)
            h_dot, dh_dx = compute_lie_derivative(h, agent_states, x_dot)

            # ---- (5) Safe / unsafe masks ----
            safe_m = env.safe_mask()           # (n_agents,) bool
            unsafe_m = env.unsafe_mask()       # (n_agents,) bool

            # ---- (6) QP target (non-differentiable) ----
            u_nom = env.nominal_controller()
            # Drift part of ẋ:  f(x) = [vx, vy, 0, 0]
            x_dot_f = torch.zeros_like(agent_states)
            x_dot_f[:, :2] = agent_states[:, 2:].detach()

            with torch.no_grad():
                u_qp = solve_cbf_qp(
                    u_nom=u_nom,
                    h=h.detach(),
                    dh_dx=dh_dx.detach(),
                    x_dot_f=x_dot_f,
                    B_mat=B_mat,
                    alpha=alpha,
                )

            # ---- (7) Compute loss ----
            loss, info = compute_loss(
                h=h,
                h_dot=h_dot,
                pi_action=pi_action,
                u_qp=u_qp,
                safe_mask=safe_m,
                unsafe_mask=unsafe_m,
                alpha=alpha,
                eps=eps,
                coef_safe=coef_safe,
                coef_unsafe=coef_unsafe,
                coef_h_dot=coef_h_dot,
                coef_action=coef_action,
            )

            batch_loss = batch_loss + loss / batch_size
            for k, v in info.items():
                batch_info[k] = batch_info.get(k, 0.0) + v / batch_size

        # ---- (8) Backprop & update ----
        optim_cbf.zero_grad()
        optim_actor.zero_grad()
        batch_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(gcbf_net.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)

        optim_cbf.step()
        optim_actor.step()

        # ---- (9) Logging ----
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t_start
            print(
                f"  step {step:5d}/{num_steps}"
                f"  |  loss {batch_info['loss/total']:.4f}"
                f"  safe {batch_info['loss/safe']:.4f}"
                f"  unsafe {batch_info['loss/unsafe']:.4f}"
                f"  hdot {batch_info['loss/h_dot']:.4f}"
                f"  action {batch_info['loss/action']:.4f}"
                f"  |  acc_s {batch_info['acc/safe']:.2f}"
                f"  acc_u {batch_info['acc/unsafe']:.2f}"
                f"  acc_h {batch_info['acc/h_dot']:.2f}"
                f"  |  {elapsed:.1f}s"
            )
            history["step"].append(step)
            for k in history:
                if k != "step" and k in batch_info:
                    history[k].append(batch_info[k])

    print("=" * 60)
    print(f" Training complete in {time.time() - t_start:.1f}s")
    print("=" * 60)
    return history


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GCBF+")
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--area_size", type=float, default=10.0)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_cbf", type=float, default=3e-5)
    parser.add_argument("--lr_actor", type=float, default=3e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
