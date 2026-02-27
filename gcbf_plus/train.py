#!/usr/bin/env python3
"""
GCBF+ Training Loop (corrected to match official algorithm).

Key changes from the previous version:
  1. Forward rollout for safe/unsafe labeling (horizon=32)
  2. Lie derivative ḣ = (∂h/∂x) · ẋ via PyTorch autograd (exact gradients)
  3. Residual policy: action = 2 * π(x) + u_ref

Trains the GCBF network h(x) and Policy network π(x) jointly using:
    - Forward rollout for safe control invariant labeling (D_C, D_A)
    - Discrete-time ḣ via forward simulation
    - CBF-QP (cvxpy) for policy target generation
    - Composite loss  L = L_CBF + L_ctrl  (Eqs. 19-22)

Usage:
    python -m gcbf_plus.train --num_agents 4 --num_steps 2000
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from gcbf_plus.env import DoubleIntegrator
from gcbf_plus.nn import GCBFNetwork, PolicyNetwork
from gcbf_plus.algo.loss import compute_lie_derivative, compute_loss
from gcbf_plus.algo.qp_solver_torch import solve_cbf_qp_batched


# ── Helper: compute the full policy action (residual formulation) ────
def policy_action(policy_net, graph, u_ref):
    """
    action = 2 * π(x) + u_ref

    The policy network outputs a *residual correction* on top of the
    nominal (LQR) controller. Factor of 2 lets it override u_ref if needed.
    """
    pi_raw = policy_net(graph)        # (n_agents, action_dim)
    return 2.0 * pi_raw + u_ref


# ── Helper: forward rollout for safe/unsafe labeling ─────────────────
def rollout_and_label(
    env: DoubleIntegrator,
    policy_net: PolicyNetwork,
    gcbf_net: GCBFNetwork,
    horizon: int,
    dev: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
           List[torch.Tensor], List[torch.Tensor]]:
    """
    Simulate the environment for `horizon` steps using the current policy.
    Collect (graph_states, safe_mask, unsafe_mask) at each step.

    Labeling rule (matching official code):
      - unsafe[t] = True if agent is in collision at time t
      - safe[t]   = True if agent is collision-free for ALL steps in
                     [max(0, t-horizon) .. t]

    Returns lists of per-step tensors.
    """
    n = env.num_agents

    # ---- Collect rollout ----
    all_graphs = []
    all_agent_states = []
    all_unsafe = []

    for t in range(horizon):
        # Build graph for current state
        agent_states_t = env.agent_states.clone()
        graph_t = env._get_graph()
        unsafe_t = env.unsafe_mask()  # (n,) bool

        all_graphs.append(graph_t)
        all_agent_states.append(agent_states_t)
        all_unsafe.append(unsafe_t)

        # Step the environment using current policy
        with torch.no_grad():
            u_ref = env.nominal_controller()
            u = policy_action(policy_net, graph_t, u_ref)
        env.step(u)

    # Stack unsafe masks: (horizon, n)
    unsafe_stack = torch.stack(all_unsafe, dim=0)  # (T, n)

    # ---- Compute safe labels using look-back window ----
    # safe[t] = True if agent was collision-free for all s in [max(0,t-H)..t]
    all_safe = []
    for t in range(horizon):
        start = max(0, t - horizon)
        # Agent is safe at t if it was never unsafe in [start..t]
        window_unsafe = unsafe_stack[start:t+1, :]  # (window, n)
        was_ever_unsafe = window_unsafe.any(dim=0)   # (n,)
        safe_t = ~was_ever_unsafe
        # Initial state (t=0) is always considered safe
        if t == 0:
            safe_t = torch.ones(n, dtype=torch.bool, device=dev)
        all_safe.append(safe_t)

    return all_graphs, all_agent_states, all_safe, all_unsafe, [env.nominal_controller() for _ in range(horizon)]


def train(
    num_agents: int = 4,
    area_size: float = 2.0,
    num_steps: int = 2000,
    batch_size: int = 8,
    horizon: int = 32,
    lr_cbf: float = 1e-4,
    lr_actor: float = 1e-4,
    alpha: float = 1.0,
    eps: float = 0.02,
    coef_safe: float = 1.0,
    coef_unsafe: float = 1.0,
    coef_h_dot: float = 0.2,
    coef_action: float = 1e-4,
    max_grad_norm: float = 2.0,
    log_interval: int = 50,
    seed: int = 0,
    checkpoint_path: str = "gcbf_plus_checkpoint.pt",
    device: str = "auto",
) -> Dict[str, list]:
    """
    Train the GCBF+ networks.

    Returns
    -------
    history : dict of lists with logged metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Device ----
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"  Device: {dev}")

    # ---- Environment ----
    env = DoubleIntegrator(
        num_agents=num_agents,
        area_size=area_size,
        params={"comm_radius": 1.5, "n_obs": 2},
    )

    # ---- Networks ----
    gcbf_net = GCBFNetwork(
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        n_agents=num_agents,
    ).to(dev)
    policy_net = PolicyNetwork(
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        action_dim=env.action_dim,
        n_agents=num_agents,
    ).to(dev)

    # ---- Optimizers ----
    optim_cbf = torch.optim.Adam(gcbf_net.parameters(), lr=lr_cbf)
    optim_actor = torch.optim.Adam(policy_net.parameters(), lr=lr_actor)

    # ---- Dynamics matrix as torch tensor (constant for Double Integrator) ----
    B_mat = torch.tensor(env.g_x_matrix, dtype=torch.float32, device=dev)

    # ---- Training history ----
    history: Dict[str, list] = {
        "step": [], "loss/total": [], "loss/safe": [], "loss/unsafe": [],
        "loss/h_dot": [], "loss/action": [], "acc/safe": [], "acc/unsafe": [],
        "acc/h_dot": [],
    }

    print("=" * 60)
    print(f"  GCBF+ Training  |  agents={num_agents}  steps={num_steps}"
          f"  horizon={horizon}")
    print("=" * 60)
    t_start = time.time()

    for step in range(1, num_steps + 1):

        # ---- (1) Collect rollouts and label safe/unsafe ----
        batch_loss = 0.0
        batch_info: Dict[str, float] = {}

        for _ in range(batch_size):
            env.reset(seed=None)
            env.to(dev)

            # ---- Forward rollout with current policy to get labels ----
            with torch.no_grad():
                graphs, agent_states_list, safe_masks, unsafe_masks, _ = \
                    rollout_and_label(env, policy_net, gcbf_net, horizon, dev)

            # ---- (2) Compute loss over all rollout steps ----
            rollout_loss = torch.tensor(0.0, device=dev)
            rollout_info: Dict[str, float] = {}
            n_valid_steps = 0

            for t in range(horizon - 1):
                # Re-build differentiable graph at time t
                states_t = agent_states_list[t].clone().requires_grad_(True)
                graph_t = env.build_graph_differentiable(states_t)

                safe_m = safe_masks[t]
                unsafe_m = unsafe_masks[t]

                # ---- (3) Forward pass: h and action ----
                h_t = gcbf_net(graph_t).squeeze(-1)         # (n_agents,)
                u_ref_t = torch.zeros(num_agents, env.action_dim, device=dev)
                # Compute u_ref from states (not from env internal state)
                with torch.no_grad():
                    error_t = states_t.detach() - env.goal_states
                    K = env._K.to(dev)
                    u_ref_t = -error_t @ K.T
                    u_max = env.params.get("u_max")
                    if u_max is not None:
                        u_ref_t = torch.clamp(u_ref_t, -u_max, u_max)

                action_t = policy_action(policy_net, graph_t, u_ref_t)

                # ---- (4) Lie derivative ḣ = (∂h/∂x) · ẋ via autograd ----
                x_dot = env.state_dot(states_t, action_t)  # (n, 4)
                h_dot, dh_dx = compute_lie_derivative(h_t, states_t, x_dot)

                # ---- (5) QP target (batched, on-device, no cvxpy) ----
                with torch.no_grad():
                    # Drift part of ẋ:  f(x) = [vx, vy, 0, 0]
                    x_dot_f = torch.zeros_like(states_t)
                    x_dot_f[:, :2] = states_t[:, 2:].detach()

                    u_qp = solve_cbf_qp_batched(
                        u_nom=u_ref_t,
                        h=h_t.detach(),
                        dh_dx=dh_dx.detach(),
                        x_dot_f=x_dot_f,
                        B_mat=B_mat,
                        alpha=alpha,
                        u_max=env.params.get("u_max"),
                    )

                # ---- (6) Compute loss ----
                loss_t, info_t = compute_loss(
                    h=h_t,
                    h_dot=h_dot,
                    pi_action=action_t,
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

                rollout_loss = rollout_loss + loss_t
                for k, v in info_t.items():
                    rollout_info[k] = rollout_info.get(k, 0.0) + v
                n_valid_steps += 1

            # Average over rollout steps
            if n_valid_steps > 0:
                rollout_loss = rollout_loss / n_valid_steps
                for k in rollout_info:
                    rollout_info[k] /= n_valid_steps

            batch_loss = batch_loss + rollout_loss / batch_size
            for k, v in rollout_info.items():
                batch_info[k] = batch_info.get(k, 0.0) + v / batch_size

        # ---- (7) Backprop & update ----
        optim_cbf.zero_grad()
        optim_actor.zero_grad()
        batch_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(gcbf_net.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)

        optim_cbf.step()
        optim_actor.step()

        # ---- (8) Logging ----
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t_start
            print(
                f"  step {step:5d}/{num_steps}"
                f"  |  loss {batch_info.get('loss/total', 0):.4f}"
                f"  safe {batch_info.get('loss/safe', 0):.4f}"
                f"  unsafe {batch_info.get('loss/unsafe', 0):.4f}"
                f"  hdot {batch_info.get('loss/h_dot', 0):.4f}"
                f"  action {batch_info.get('loss/action', 0):.4f}"
                f"  |  acc_s {batch_info.get('acc/safe', 0):.2f}"
                f"  acc_u {batch_info.get('acc/unsafe', 0):.2f}"
                f"  acc_h {batch_info.get('acc/h_dot', 0):.2f}"
                f"  |  {elapsed:.1f}s"
            )
            history["step"].append(step)
            for k in history:
                if k != "step" and k in batch_info:
                    history[k].append(batch_info[k])

    print("=" * 60)
    print(f"  Training complete in {time.time() - t_start:.1f}s")
    print("=" * 60)

    # ---- Save checkpoint ----
    ckpt = {
        "gcbf_net": gcbf_net.state_dict(),
        "policy_net": policy_net.state_dict(),
        "config": {
            "num_agents": num_agents,
            "area_size": area_size,
            "node_dim": env.node_dim,
            "edge_dim": env.edge_dim,
            "action_dim": env.action_dim,
            "comm_radius": env.comm_radius,
            "n_obs": env.params["n_obs"],
            "dt": env.dt,
        },
        "history": history,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"  Checkpoint saved to: {checkpoint_path}")

    return history


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GCBF+")
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--area_size", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--lr_cbf", type=float, default=1e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="gcbf_plus_checkpoint.pt",
                        help="Path to save the trained checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto' (detect GPU), 'cuda', or 'cpu'")
    args = parser.parse_args()
    train_args = vars(args)
    train_args["checkpoint_path"] = train_args.pop("checkpoint")
    train(**train_args)


if __name__ == "__main__":
    main()
