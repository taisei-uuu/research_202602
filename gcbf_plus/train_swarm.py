#!/usr/bin/env python3
"""
GCBF+ Training Loop — Swarm variant (3-drone rigid body per node).

Adapted from train.py for the SwarmIntegrator environment:
  - State dim 6:  [px, py, vx, vy, θ, ω]
  - Action dim 3: [ax, ay, α]
  - Edge dim 8:   [Δpx, Δpy, Δvx, Δvy, Δθ, min_dist, closest_dx, closest_dy]
  - g(x) matrix:  6×3  (includes translational + rotational dynamics)

Usage:
    python -m gcbf_plus.train_swarm --num_agents 4 --num_steps 2000
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from gcbf_plus.env import SwarmIntegrator
from gcbf_plus.nn import GCBFNetwork, PolicyNetwork
from gcbf_plus.algo.loss import compute_loss
from gcbf_plus.algo.qp_solver_torch import solve_cbf_qp_batched
from gcbf_plus.utils.graph import GraphsTuple
from gcbf_plus.utils.swarm_graph import build_swarm_graph_from_states


# ── Build a batched mega-graph from multiple independent samples ──────
def build_mega_graph(
    agent_states_list: List[torch.Tensor],
    goal_states_list: List[torch.Tensor],
    obstacle_states_list: List[torch.Tensor],
    comm_radius: float,
    node_dim: int,
    edge_dim: int,
    n_agents: int,
    R_form: float,
) -> GraphsTuple:
    """
    Merge S independent graph samples into ONE GraphsTuple.

    Uses build_swarm_graph_from_states for 8D edge features with
    drone-to-drone sensing.
    """
    S = len(agent_states_list)
    device = agent_states_list[0].device

    all_nodes = []
    all_edges = []
    all_senders = []
    all_receivers = []
    all_node_types = []
    node_offset = 0

    for s in range(S):
        g = build_swarm_graph_from_states(
            agent_states=agent_states_list[s],
            goal_states=goal_states_list[s],
            obstacle_positions=obstacle_states_list[s],
            comm_radius=comm_radius,
            node_dim=node_dim,
            edge_dim=edge_dim,
            R_form=R_form,
        )
        all_nodes.append(g.nodes)
        all_edges.append(g.edges)
        all_senders.append(g.senders + node_offset)
        all_receivers.append(g.receivers + node_offset)
        all_node_types.append(g.node_type)
        node_offset += g.n_node

    mega_graph = GraphsTuple(
        nodes=torch.cat(all_nodes, dim=0),
        edges=torch.cat(all_edges, dim=0) if any(e.numel() > 0 for e in all_edges) else torch.zeros(0, edge_dim, device=device),
        senders=torch.cat(all_senders, dim=0) if any(s.numel() > 0 for s in all_senders) else torch.zeros(0, dtype=torch.long, device=device),
        receivers=torch.cat(all_receivers, dim=0) if any(r.numel() > 0 for r in all_receivers) else torch.zeros(0, dtype=torch.long, device=device),
        n_node=node_offset,
        n_edge=sum(e.shape[0] for e in all_edges),
        node_type=torch.cat(all_node_types, dim=0),
    )
    return mega_graph


def extract_agent_outputs(
    full_output: torch.Tensor,
    n_agents: int,
    n_nodes_per_sample: int,
    n_samples: int,
) -> torch.Tensor:
    """
    Extract agent-only outputs from a mega-graph GNN output.

    For each sample, agents are the first n_agents nodes.
    """
    indices = []
    for s in range(n_samples):
        start = s * n_nodes_per_sample
        indices.append(torch.arange(start, start + n_agents, device=full_output.device))
    idx = torch.cat(indices)
    return full_output[idx]


# ── Forward rollout for safe/unsafe labeling ──────────────────────────
def collect_rollout_data(
    env: SwarmIntegrator,
    policy_net: PolicyNetwork,
    horizon: int,
    dev: torch.device,
    n_agents: int,
) -> Dict[str, List[torch.Tensor]]:
    """
    Run a single rollout (no_grad) and collect per-timestep data.
    """
    state_dim = env.state_dim  # 6

    all_agent_states = []
    all_goal_states = []
    all_obstacle_states = []
    all_unsafe = []

    for t in range(horizon):
        all_agent_states.append(env.agent_states.clone())
        all_goal_states.append(env.goal_states.clone())
        all_obstacle_states.append(
            env._obstacle_states.clone()
            if env._obstacle_states is not None
            else torch.zeros(0, state_dim, device=dev)
        )
        all_unsafe.append(env.unsafe_mask())

        # Step using current policy
        graph_t = env._get_graph()
        u_ref = env.nominal_controller()
        pi_raw = policy_net(graph_t)
        u = 2.0 * pi_raw + u_ref
        env.step(u)

    # Compute safe labels from unsafe masks
    unsafe_stack = torch.stack(all_unsafe, dim=0)  # (T, n_agents)
    all_safe = []
    for t in range(horizon):
        start = max(0, t - horizon)
        window_unsafe = unsafe_stack[start:t+1, :]
        was_ever_unsafe = window_unsafe.any(dim=0)
        safe_t = ~was_ever_unsafe
        if t == 0:
            safe_t = torch.ones(n_agents, dtype=torch.bool, device=dev)
        all_safe.append(safe_t)

    return {
        "agent_states": all_agent_states,
        "goal_states": all_goal_states,
        "obstacle_states": all_obstacle_states,
        "safe_masks": all_safe,
        "unsafe_masks": all_unsafe,
    }


def train(
    num_agents: int = 4,
    area_size: float = 4.0,
    num_steps: int = 2000,
    batch_size: int = 128,
    horizon: int = 32,
    lr_cbf: float = 1e-4,
    lr_actor: float = 1e-4,
    alpha: float = 1.0,
    eps: float = 0.02,
    coef_safe: float = 1.0,
    coef_unsafe: float = 2.0,
    coef_h_dot: float = 0.2,
    coef_action: float = 1e-4,
    max_grad_norm: float = 2.0,
    log_interval: int = 1,
    seed: int = 0,
    checkpoint_path: str = "gcbf_swarm_checkpoint.pt",
    device: str = "auto",
) -> Dict[str, list]:
    """Train the GCBF+ networks (swarm variant) with fully batched inner loop."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Device ----
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"  Device: {dev}")

    # ---- Environment ----
    env = SwarmIntegrator(
        num_agents=num_agents,
        area_size=area_size,
        params={"comm_radius": 2.0, "n_obs": 2},
    )

    # ---- Networks (dimensions adapted for swarm) ----
    gcbf_net = GCBFNetwork(
        node_dim=env.node_dim,    # 3
        edge_dim=env.edge_dim,    # 8
        n_agents=num_agents,
    ).to(dev)
    policy_net = PolicyNetwork(
        node_dim=env.node_dim,    # 3
        edge_dim=env.edge_dim,    # 8
        action_dim=env.action_dim,  # 3
        n_agents=num_agents,
    ).to(dev)

    # ---- Optimizers ----
    optim_cbf = torch.optim.Adam(gcbf_net.parameters(), lr=lr_cbf)
    optim_actor = torch.optim.Adam(policy_net.parameters(), lr=lr_actor)

    # ---- Constants ----
    B_mat = torch.tensor(env.g_x_matrix, dtype=torch.float32, device=dev)  # (6, 3)
    K_mat = env._K_trans.to(dev)  # (2, 4) — translational LQR gain
    mass = env.params["mass"]
    inertia = env.params["inertia"]
    u_max = env.params.get("u_max")
    alpha_max = env.params.get("alpha_max")
    R_form = env.params["R_form"]
    dt = env.dt
    n_obs_actual = env._obstacle_states.shape[0] if env._obstacle_states is not None else 0
    n_nodes_per_sample = num_agents * 2 + n_obs_actual

    # Total flattened samples per training step
    T_loss = horizon - 1

    # ---- Training history ----
    history: Dict[str, list] = {
        "step": [], "loss/total": [], "loss/safe": [], "loss/unsafe": [],
        "loss/h_dot": [], "loss/action": [], "acc/safe": [], "acc/unsafe": [],
        "acc/h_dot": [],
    }

    print("=" * 60)
    print(f"  GCBF+ Swarm Training (batched)  |  swarms={num_agents}"
          f"  batch={batch_size}  horizon={horizon}")
    print(f"  Samples per step: {batch_size} × {T_loss} = "
          f"{batch_size * T_loss} graphs in ONE forward pass")
    print(f"  State dim={env.state_dim}  Action dim={env.action_dim}"
          f"  Edge dim={env.edge_dim}")
    print("=" * 60)
    t_start = time.time()

    for step in range(1, num_steps + 1):

        # ============================================================
        # PHASE 1: Data Collection (no_grad, sequential over batch)
        # ============================================================
        all_agent_states = []
        all_goal_states = []
        all_obs_states = []
        all_safe_masks = []
        all_unsafe_masks = []

        with torch.no_grad():
            for b in range(batch_size):
                env.reset(seed=None)
                env.to(dev)

                rollout_data = collect_rollout_data(
                    env, policy_net, horizon, dev, num_agents
                )

                for t in range(T_loss):
                    all_agent_states.append(rollout_data["agent_states"][t])
                    all_goal_states.append(rollout_data["goal_states"][t])
                    all_obs_states.append(rollout_data["obstacle_states"][t])
                    all_safe_masks.append(rollout_data["safe_masks"][t])
                    all_unsafe_masks.append(rollout_data["unsafe_masks"][t])

        S = len(all_agent_states)  # batch_size * T_loss

        # Stack masks: (S, n_agents) → flatten to (S * n_agents,)
        safe_mask_flat = torch.stack(all_safe_masks).reshape(-1)
        unsafe_mask_flat = torch.stack(all_unsafe_masks).reshape(-1)

        # ============================================================
        # PHASE 2: Build ONE mega-graph (with gradient tracking)
        # ============================================================
        agent_states_grad = torch.stack(all_agent_states).detach().requires_grad_(True)
        # Shape: (S, n_agents, 6)

        agent_states_list = [agent_states_grad[s] for s in range(S)]

        mega_graph = build_mega_graph(
            agent_states_list=agent_states_list,
            goal_states_list=all_goal_states,
            obstacle_states_list=all_obs_states,
            comm_radius=env.comm_radius,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            n_agents=num_agents,
            R_form=R_form,
        )

        # ============================================================
        # PHASE 3: ONE forward pass through GNN
        # ============================================================
        h_all_nodes = gcbf_net.gnn_layers[0](mega_graph)      # (total_nodes, 1)
        pi_all_nodes = policy_net.gnn_layers[0](mega_graph)    # (total_nodes, 3)

        h_agents = extract_agent_outputs(h_all_nodes, num_agents, n_nodes_per_sample, S).squeeze(-1)
        pi_agents = extract_agent_outputs(pi_all_nodes, num_agents, n_nodes_per_sample, S)

        # ============================================================
        # PHASE 4: Compute u_ref, action, Lie derivative (all batched)
        # ============================================================
        states_flat = agent_states_grad.reshape(-1, 6)  # (S*n, 6)
        goals_flat = torch.stack(all_goal_states).reshape(-1, 6)  # (S*n, 6)

        # u_ref: translational LQR + angular PD
        error_trans = states_flat[:, :4].detach() - goals_flat[:, :4]
        u_trans = -error_trans @ K_mat.T  # (S*n, 2)
        if u_max is not None:
            u_trans = torch.clamp(u_trans, -u_max, u_max)

        from gcbf_plus.utils.swarm_graph import _wrap_angle
        theta_err = _wrap_angle(states_flat[:, 4].detach() - goals_flat[:, 4])
        omega = states_flat[:, 5].detach()
        u_alpha = -(env._Kp_theta * theta_err + env._Kd_theta * omega)
        if alpha_max is not None:
            u_alpha = torch.clamp(u_alpha, -alpha_max, alpha_max)

        u_ref_flat = torch.cat([u_trans, u_alpha.unsqueeze(-1)], dim=1)  # (S*n, 3)

        # Residual policy action
        action_flat = 2.0 * pi_agents + u_ref_flat  # (S*n, 3)

        # State derivative: ẋ = [v, a/m, ω, α/I]
        vel = states_flat[:, 2:4]  # (S*n, 2)
        action_clamped = action_flat.clone()
        if u_max is not None:
            action_clamped = torch.cat([
                torch.clamp(action_clamped[:, :2], -u_max, u_max),
                action_clamped[:, 2:3],
            ], dim=1)
        if alpha_max is not None:
            action_clamped = torch.cat([
                action_clamped[:, :2],
                torch.clamp(action_clamped[:, 2:3], -alpha_max, alpha_max),
            ], dim=1)

        accel = action_clamped[:, :2] / mass        # (S*n, 2)
        omega_state = states_flat[:, 5:6]            # (S*n, 1)
        angular_accel = action_clamped[:, 2:3] / inertia  # (S*n, 1)
        x_dot = torch.cat([vel, accel, omega_state, angular_accel], dim=1)  # (S*n, 6)

        # ONE Lie derivative computation
        dh_dx_3d = torch.autograd.grad(
            outputs=h_agents.sum(),
            inputs=agent_states_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # (S, n_agents, 6)
        dh_dx = dh_dx_3d.reshape(-1, 6)  # (S*n, 6)

        h_dot = (dh_dx * x_dot).sum(dim=-1)  # (S*n,)

        # ============================================================
        # PHASE 5: ONE batched QP solve
        # ============================================================
        with torch.no_grad():
            x_dot_f = torch.zeros_like(states_flat)
            x_dot_f[:, :2] = states_flat[:, 2:4].detach()   # ṗ = v
            x_dot_f[:, 4:5] = states_flat[:, 5:6].detach()  # θ̇ = ω

            u_qp = solve_cbf_qp_batched(
                u_nom=u_ref_flat,
                h=h_agents.detach(),
                dh_dx=dh_dx.detach(),
                x_dot_f=x_dot_f,
                B_mat=B_mat,
                alpha=alpha,
                u_max=u_max,  # Note: this is approximate for angular; works in practice
            )

        # ============================================================
        # PHASE 6: ONE batched loss computation
        # ============================================================
        loss, info = compute_loss(
            h=h_agents,
            h_dot=h_dot,
            pi_action=action_flat,
            u_qp=u_qp,
            safe_mask=safe_mask_flat,
            unsafe_mask=unsafe_mask_flat,
            alpha=alpha,
            eps=eps,
            coef_safe=coef_safe,
            coef_unsafe=coef_unsafe,
            coef_h_dot=coef_h_dot,
            coef_action=coef_action,
        )

        # ============================================================
        # PHASE 7: Backprop & update
        # ============================================================
        optim_cbf.zero_grad()
        optim_actor.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(gcbf_net.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)

        optim_cbf.step()
        optim_actor.step()

        # ============================================================
        # Logging
        # ============================================================
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t_start
            print(
                f"  step {step:5d}/{num_steps}"
                f"  |  loss {info.get('loss/total', 0):.4f}"
                f"  safe {info.get('loss/safe', 0):.4f}"
                f"  unsafe {info.get('loss/unsafe', 0):.4f}"
                f"  hdot {info.get('loss/h_dot', 0):.4f}"
                f"  action {info.get('loss/action', 0):.4f}"
                f"  |  acc_s {info.get('acc/safe', 0):.2f}"
                f"  acc_u {info.get('acc/unsafe', 0):.2f}"
                f"  acc_h {info.get('acc/h_dot', 0):.2f}"
                f"  |  {elapsed:.1f}s"
            )
            history["step"].append(step)
            for k in history:
                if k != "step" and k in info:
                    history[k].append(info[k])

    print("=" * 60)
    print(f"  Swarm training complete in {time.time() - t_start:.1f}s")
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
            "state_dim": env.state_dim,
            "comm_radius": env.comm_radius,
            "R_form": R_form,
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
    parser = argparse.ArgumentParser(description="Train GCBF+ (Swarm)")
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--area_size", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr_cbf", type=float, default=1e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="gcbf_swarm_checkpoint.pt",
                        help="Path to save the trained checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto' (detect GPU), 'cuda', or 'cpu'")
    args = parser.parse_args()
    train_args = vars(args)
    train_args["checkpoint_path"] = train_args.pop("checkpoint")
    train(**train_args)


if __name__ == "__main__":
    main()
