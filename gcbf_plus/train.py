#!/usr/bin/env python3
"""
GCBF+ Training Loop — Option C: Fully Batched Inner Loop.

Key optimizations vs. the sequential version:
  - Data collection (rollout + labeling) runs under torch.no_grad()
  - ALL samples (batch × timesteps) are flattened into ONE mega-graph
  - ONE GNN forward pass, ONE Lie derivative, ONE QP solve, ONE loss
  - GPU utilisation is maximised; no Python loops during gradient computation

Algorithm (matching official paper):
  1. Forward rollout for safe/unsafe labeling (horizon=32)
  2. Lie derivative ḣ = (∂h/∂x) · ẋ via PyTorch autograd
  3. Residual policy: action = 2 * π(x) + u_ref
  4. Analytical KKT QP solver (pure PyTorch, on-device)

Usage:
    python -m gcbf_plus.train --num_agents 4 --num_steps 2000
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from gcbf_plus.env import DoubleIntegrator
from gcbf_plus.nn import GCBFNetwork, PolicyNetwork
from gcbf_plus.algo.loss import compute_lie_derivative, compute_loss
from gcbf_plus.algo.qp_solver_torch import solve_cbf_qp_batched
from gcbf_plus.utils.graph import GraphsTuple, build_graph_from_states


# ── Build a batched mega-graph from multiple independent samples ──────
def build_mega_graph(
    agent_states_list: List[torch.Tensor],
    goal_states_list: List[torch.Tensor],
    obstacle_states_list: List[torch.Tensor],
    comm_radius: float,
    node_dim: int,
    edge_dim: int,
    n_agents: int,
) -> GraphsTuple:
    """
    Merge S independent graph samples into ONE GraphsTuple.

    Each sample has its own agents, goals, obstacles. Node indices are
    offset so that edges never cross sample boundaries.

    Parameters
    ----------
    agent_states_list : list of (n_agents, 4) tensors — S samples
    goal_states_list  : list of (n_agents, 4) tensors
    obstacle_states_list : list of (n_obs, 4) tensors
    comm_radius, node_dim, edge_dim, n_agents : env params

    Returns
    -------
    mega_graph : GraphsTuple with S*N_per_sample total nodes
    agent_mask : (total_nodes,) bool — True for agent nodes
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
        g = build_graph_from_states(
            agent_states=agent_states_list[s],
            goal_states=goal_states_list[s],
            obstacle_positions=obstacle_states_list[s],
            comm_radius=comm_radius,
            node_dim=node_dim,
            edge_dim=edge_dim,
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

    The mega-graph has nodes laid out as:
        [agents_0, goals_0, obs_0, agents_1, goals_1, obs_1, ...]

    For each sample, agents are the first n_agents nodes.

    Parameters
    ----------
    full_output : (total_nodes, out_dim)
    n_agents : agents per sample
    n_nodes_per_sample : total nodes per sample (agents + goals + obstacles)
    n_samples : number of samples

    Returns
    -------
    (n_samples * n_agents, out_dim)
    """
    # Build index: for each sample s, take indices [s*N .. s*N + n_agents)
    indices = []
    for s in range(n_samples):
        start = s * n_nodes_per_sample
        indices.append(torch.arange(start, start + n_agents, device=full_output.device))
    idx = torch.cat(indices)  # (n_samples * n_agents,)
    return full_output[idx]


# ── Forward rollout for safe/unsafe labeling ──────────────────────────
def collect_rollout_data(
    env: DoubleIntegrator,
    policy_net: PolicyNetwork,
    horizon: int,
    dev: torch.device,
    n_agents: int,
) -> Dict[str, List[torch.Tensor]]:
    """
    Run a single rollout (no_grad) and collect per-timestep data.

    Returns dict with lists of tensors, one per timestep.
    """
    all_agent_states = []
    all_goal_states = []
    all_obstacle_states = []
    all_unsafe = []

    for t in range(horizon):
        all_agent_states.append(env.agent_states.clone())
        all_goal_states.append(env.goal_states.clone())
        all_obstacle_states.append(env._obstacle_states.clone() if env._obstacle_states is not None else torch.zeros(0, 4, device=dev))
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
    area_size: float = 2.0,
    num_steps: int = 2000,
    batch_size: int = 256,
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
    log_interval: int = 1,
    seed: int = 0,
    checkpoint_path: str = "gcbf_plus_checkpoint.pt",
    device: str = "auto",
) -> Dict[str, list]:
    """Train the GCBF+ networks with fully batched inner loop."""
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

    # ---- Constants ----
    B_mat = torch.tensor(env.g_x_matrix, dtype=torch.float32, device=dev)
    K_mat = env._K.to(dev)
    mass = env.params["mass"]
    u_max = env.params.get("u_max")
    dt = env.dt
    n_obs_actual = env._obstacle_states.shape[0] if env._obstacle_states is not None else 0
    n_nodes_per_sample = num_agents * 2 + n_obs_actual  # agents + goals + obstacles

    # Total flattened samples per training step
    T_loss = horizon - 1  # timesteps used for loss (we don't use last one)

    # ---- Training history ----
    history: Dict[str, list] = {
        "step": [], "loss/total": [], "loss/safe": [], "loss/unsafe": [],
        "loss/h_dot": [], "loss/action": [], "acc/safe": [], "acc/unsafe": [],
        "acc/h_dot": [],
    }

    print("=" * 60)
    print(f"  GCBF+ Training (batched)  |  agents={num_agents}"
          f"  batch={batch_size}  horizon={horizon}")
    print(f"  Samples per step: {batch_size} × {T_loss} = "
          f"{batch_size * T_loss} graphs in ONE forward pass")
    print("=" * 60)
    t_start = time.time()

    for step in range(1, num_steps + 1):

        # ============================================================
        # PHASE 1: Data Collection (no_grad, sequential over batch)
        # ============================================================
        all_agent_states = []   # will be (S, n_agents, 4)
        all_goal_states = []    # will be (S, n_agents, 4)
        all_obs_states = []     # will be (S, n_obs, 4)
        all_safe_masks = []     # will be (S, n_agents) bool
        all_unsafe_masks = []   # will be (S, n_agents) bool

        with torch.no_grad():
            for b in range(batch_size):
                env.reset(seed=None)
                env.to(dev)

                rollout_data = collect_rollout_data(
                    env, policy_net, horizon, dev, num_agents
                )

                # Collect timesteps 0..T_loss-1 (skip last step of rollout)
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
        # PHASE 2: Build ONE mega-graph (with gradient tracking on edges)
        # ============================================================
        # Re-create agent_states with requires_grad for Lie derivative
        agent_states_grad = torch.stack(all_agent_states).detach().requires_grad_(True)
        # Shape: (S, n_agents, 4)

        # Split back into list for graph building
        agent_states_list = [agent_states_grad[s] for s in range(S)]

        mega_graph = build_mega_graph(
            agent_states_list=agent_states_list,
            goal_states_list=all_goal_states,
            obstacle_states_list=all_obs_states,
            comm_radius=env.comm_radius,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            n_agents=num_agents,
        )

        # ============================================================
        # PHASE 3: ONE forward pass through GNN
        # ============================================================
        # Get raw GNN output for all nodes
        h_all_nodes = gcbf_net.gnn_layers[0](mega_graph)  # (total_nodes, 1)
        pi_all_nodes = policy_net.gnn_layers[0](mega_graph)  # (total_nodes, action_dim)

        # Extract agent-only outputs: (S * n_agents, ...)
        h_agents = extract_agent_outputs(h_all_nodes, num_agents, n_nodes_per_sample, S).squeeze(-1)
        pi_agents = extract_agent_outputs(pi_all_nodes, num_agents, n_nodes_per_sample, S)

        # ============================================================
        # PHASE 4: Compute u_ref, action, Lie derivative (all batched)
        # ============================================================
        # Flatten agent states for vectorized ops
        states_flat = agent_states_grad.reshape(-1, 4)  # (S*n, 4)  — view of leaf
        goals_flat = torch.stack(all_goal_states).reshape(-1, 4)  # (S*n, 4)

        # u_ref = -K (x - x_goal), clamped
        error = states_flat.detach() - goals_flat
        u_ref_flat = -error @ K_mat.T  # (S*n, 2)
        if u_max is not None:
            u_ref_flat = torch.clamp(u_ref_flat, -u_max, u_max)

        # Residual policy action
        action_flat = 2.0 * pi_agents + u_ref_flat  # (S*n, action_dim)

        # State derivative: ẋ = [v, a/m]
        vel = states_flat[:, 2:]  # (S*n, 2)
        action_clamped = action_flat
        if u_max is not None:
            action_clamped = torch.clamp(action_flat, -u_max, u_max)
        accel = action_clamped / mass  # (S*n, 2)
        x_dot = torch.cat([vel, accel], dim=1)  # (S*n, 4)

        # ONE Lie derivative computation
        # IMPORTANT: differentiate w.r.t. agent_states_grad (the leaf tensor
        # that was used to build the graph), NOT states_flat (a post-hoc view)
        dh_dx_3d = torch.autograd.grad(
            outputs=h_agents.sum(),
            inputs=agent_states_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # (S, n_agents, 4)
        dh_dx = dh_dx_3d.reshape(-1, 4)  # (S*n, 4)

        # ḣ = (∂h/∂x) · ẋ
        h_dot = (dh_dx * x_dot).sum(dim=-1)  # (S*n,)

        # ============================================================
        # PHASE 5: ONE batched QP solve
        # ============================================================
        with torch.no_grad():
            x_dot_f = torch.zeros_like(states_flat)
            x_dot_f[:, :2] = states_flat[:, 2:].detach()

            u_qp = solve_cbf_qp_batched(
                u_nom=u_ref_flat,
                h=h_agents.detach(),
                dh_dx=dh_dx.detach(),
                x_dot_f=x_dot_f,
                B_mat=B_mat,
                alpha=alpha,
                u_max=u_max,
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
        # PHASE 7: Backprop & update (ONE backward pass)
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
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr_cbf", type=float, default=1e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--log_interval", type=int, default=1)
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
