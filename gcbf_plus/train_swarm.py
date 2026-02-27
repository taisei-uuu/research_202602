#!/usr/bin/env python3
"""
GCBF+ Training Loop — Swarm variant (VECTORIZED).

Uses VectorizedSwarmEnv to run all B environments in parallel on GPU.
Data collection: B environments × H timesteps → only H graph builds + GNN
forward passes (instead of B×H in the non-vectorized version).

Usage:
    python -m gcbf_plus.train_swarm --num_agents 3 --num_steps 2000 --batch_size 256
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np
import torch

from gcbf_plus.env.vectorized_swarm import VectorizedSwarmEnv
from gcbf_plus.nn import GCBFNetwork, PolicyNetwork
from gcbf_plus.algo.loss import compute_loss
from gcbf_plus.algo.qp_solver_torch import solve_cbf_qp_batched
from gcbf_plus.utils.graph import GraphsTuple
from gcbf_plus.utils.swarm_graph import (
    _wrap_angle,
    get_equilateral_offsets,
    build_vectorized_swarm_graph,
)


def extract_agent_outputs(
    full_output: torch.Tensor,
    n_agents: int,
    n_nodes_per_sample: int,
    n_samples: int,
) -> torch.Tensor:
    """
    Extract agent-only rows from a mega-graph GNN output.

    For each of n_samples graphs, agents are the first n_agents nodes.
    """
    offsets = torch.arange(n_samples, device=full_output.device) * n_nodes_per_sample
    agent_offsets = torch.arange(n_agents, device=full_output.device)
    # (n_samples, n_agents)
    idx = offsets.unsqueeze(1) + agent_offsets.unsqueeze(0)
    return full_output[idx.reshape(-1)]


def train(
    num_agents: int = 3,
    area_size: float = 4.0,
    num_steps: int = 2000,
    batch_size: int = 256,
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
    log_interval: int = 100,
    seed: int = 0,
    checkpoint_path: str = "gcbf_swarm_checkpoint.pt",
    device: str = "auto",
) -> Dict[str, list]:
    """Train GCBF+ swarm networks with vectorized data collection."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Device ----
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"  Device: {dev}")

    # ---- Vectorized environment ----
    n_obs = 2
    vec_env = VectorizedSwarmEnv(
        num_agents=num_agents,
        batch_size=batch_size,
        area_size=area_size,
        params={"comm_radius": 2.0, "n_obs": n_obs},
    )

    # ---- Networks ----
    gcbf_net = GCBFNetwork(
        node_dim=vec_env.node_dim,    # 3
        edge_dim=vec_env.edge_dim,    # 8
        n_agents=num_agents,
    ).to(dev)
    policy_net = PolicyNetwork(
        node_dim=vec_env.node_dim,    # 3
        edge_dim=vec_env.edge_dim,    # 8
        action_dim=vec_env.action_dim,  # 3
        n_agents=num_agents,
    ).to(dev)

    optim_cbf = torch.optim.Adam(gcbf_net.parameters(), lr=lr_cbf)
    optim_actor = torch.optim.Adam(policy_net.parameters(), lr=lr_actor)

    # ---- Constants ----
    B_mat = torch.tensor(vec_env.g_x_matrix, dtype=torch.float32, device=dev)
    K_mat = vec_env._K_trans.to(dev)
    mass = vec_env.params["mass"]
    inertia = vec_env.params["inertia"]
    u_max = vec_env.params.get("u_max")
    alpha_max_val = vec_env.params.get("alpha_max")
    R_form = vec_env.params["R_form"]
    N_per = num_agents * 2 + n_obs  # nodes per sample
    T_loss = horizon - 1
    Kp_theta = vec_env._Kp_theta
    Kd_theta = vec_env._Kd_theta

    # ---- History ----
    history: Dict[str, list] = {
        "step": [], "loss/total": [], "loss/safe": [], "loss/unsafe": [],
        "loss/h_dot": [], "loss/action": [], "acc/safe": [], "acc/unsafe": [],
        "acc/h_dot": [],
    }

    print("=" * 60)
    print(f"  GCBF+ Swarm Training (VECTORIZED)"
          f"  |  swarms={num_agents}  batch={batch_size}  horizon={horizon}")
    print(f"  Data collection: {horizon} graph builds (was {batch_size}×{horizon}"
          f" = {batch_size*horizon} before)")
    print(f"  Training samples per step: {batch_size} × {T_loss} "
          f"= {batch_size * T_loss}")
    print(f"  State=6D  Action=3D  Edge=8D  Nodes/sample={N_per}")
    print("=" * 60)
    t_start = time.time()

    for step in range(1, num_steps + 1):

        # ============================================================
        # PHASE 1: VECTORIZED Data Collection (no_grad)
        # ============================================================
        # All B environments run in parallel — ONE graph build + ONE
        # GNN forward pass per timestep (instead of B each).
        # ============================================================
        vec_env.reset(dev)

        # Fixed across the rollout
        goal_states_fixed = vec_env._goal_states.clone()        # (B, n, 6)
        obs_states_fixed = vec_env._obstacle_states.clone()     # (B, n_obs, 6)

        all_agent_states = []   # list of (B, n, 6), length = horizon
        all_unsafe = []         # list of (B, n),    length = horizon

        with torch.no_grad():
            for t in range(horizon):
                all_agent_states.append(vec_env._agent_states.clone())
                all_unsafe.append(vec_env.unsafe_mask())

                # ONE mega-graph for all B envs
                mega = vec_env.build_batch_graph()
                # ONE GNN forward for all B envs
                pi_all = policy_net.gnn_layers[0](mega)  # (B*N_per, 3)
                pi_agents = extract_agent_outputs(
                    pi_all, num_agents, N_per, batch_size
                )  # (B*n, 3)

                u_ref = vec_env.nominal_controller()  # (B, n, 3)
                u = 2.0 * pi_agents.reshape(batch_size, num_agents, 3) + u_ref
                vec_env.step(u)

        # Compute safe labels from unsafe masks
        unsafe_stack = torch.stack(all_unsafe)  # (T, B, n)
        safe_list = []
        for t in range(horizon):
            window = unsafe_stack[:t+1]
            was_ever_unsafe = window.any(dim=0)  # (B, n)
            safe_t = ~was_ever_unsafe
            if t == 0:
                safe_t = torch.ones(batch_size, num_agents, dtype=torch.bool, device=dev)
            safe_list.append(safe_t)

        # ============================================================
        # PHASE 2: Reshape collected data for training
        # ============================================================
        # Use timesteps 0..T_loss-1 (= horizon-2)
        # Stack → (T_loss, B, n, 6) then reshape to (S, n, 6)
        S = T_loss * batch_size

        agent_states_4d = torch.stack(all_agent_states[:T_loss])  # (T_loss, B, n, 6)
        agent_states_3d = agent_states_4d.reshape(S, num_agents, 6)

        goal_rep = goal_states_fixed.unsqueeze(0).expand(T_loss, -1, -1, -1)
        goal_3d = goal_rep.reshape(S, num_agents, 6)

        obs_rep = obs_states_fixed.unsqueeze(0).expand(T_loss, -1, -1, -1)
        obs_3d = obs_rep.reshape(S, n_obs, 6)

        safe_4d = torch.stack(safe_list[:T_loss])   # (T_loss, B, n)
        unsafe_4d = torch.stack(all_unsafe[:T_loss]) # (T_loss, B, n)
        safe_flat = safe_4d.reshape(-1)               # (S*n,)
        unsafe_flat = unsafe_4d.reshape(-1)            # (S*n,)

        # ============================================================
        # PHASE 3: Build ONE mega-graph (gradient-tracked)
        # ============================================================
        agent_states_grad = agent_states_3d.detach().requires_grad_(True)

        mega_graph = build_vectorized_swarm_graph(
            agent_states=agent_states_grad,
            goal_states=goal_3d,
            obstacle_states=obs_3d,
            local_offsets=vec_env._local_offsets,
            comm_radius=vec_env.comm_radius,
            node_dim=vec_env.node_dim,
            edge_dim=vec_env.edge_dim,
        )

        # ============================================================
        # PHASE 4: ONE forward pass through GNN
        # ============================================================
        h_all = gcbf_net.gnn_layers[0](mega_graph)
        pi_all = policy_net.gnn_layers[0](mega_graph)

        h_agents = extract_agent_outputs(h_all, num_agents, N_per, S).squeeze(-1)
        pi_agents = extract_agent_outputs(pi_all, num_agents, N_per, S)

        # ============================================================
        # PHASE 5: Compute u_ref, action, Lie derivative
        # ============================================================
        states_flat = agent_states_grad.reshape(-1, 6)
        goals_flat = goal_3d.reshape(-1, 6)

        # Translational LQR
        err_trans = states_flat[:, :4].detach() - goals_flat[:, :4]
        u_trans = -torch.einsum("...j,ij->...i", err_trans, K_mat)
        if u_max is not None:
            u_trans = torch.clamp(u_trans, -u_max, u_max)

        # Angular PD
        theta_err = _wrap_angle(states_flat[:, 4].detach() - goals_flat[:, 4])
        omega = states_flat[:, 5].detach()
        u_alpha = -(Kp_theta * theta_err + Kd_theta * omega)
        if alpha_max_val is not None:
            u_alpha = torch.clamp(u_alpha, -alpha_max_val, alpha_max_val)

        u_ref_flat = torch.cat([u_trans, u_alpha.unsqueeze(-1)], dim=1)

        action_flat = 2.0 * pi_agents + u_ref_flat

        # State derivative ẋ = [v, a/m, ω, α/I]
        action_c = action_flat.clone()
        if u_max is not None:
            action_c = torch.cat([
                torch.clamp(action_c[:, :2], -u_max, u_max),
                action_c[:, 2:3],
            ], dim=1)
        if alpha_max_val is not None:
            action_c = torch.cat([
                action_c[:, :2],
                torch.clamp(action_c[:, 2:3], -alpha_max_val, alpha_max_val),
            ], dim=1)

        vel = states_flat[:, 2:4]
        accel = action_c[:, :2] / mass
        omega_s = states_flat[:, 5:6]
        ang_accel = action_c[:, 2:3] / inertia
        x_dot = torch.cat([vel, accel, omega_s, ang_accel], dim=1)

        # Lie derivative
        dh_dx_3d = torch.autograd.grad(
            h_agents.sum(), agent_states_grad,
            create_graph=True, retain_graph=True,
        )[0]
        dh_dx = dh_dx_3d.reshape(-1, 6)
        h_dot = (dh_dx * x_dot).sum(dim=-1)

        # ============================================================
        # PHASE 6: QP solve (batched, no_grad)
        # ============================================================
        with torch.no_grad():
            x_dot_f = torch.zeros_like(states_flat)
            x_dot_f[:, :2] = states_flat[:, 2:4].detach()
            x_dot_f[:, 4:5] = states_flat[:, 5:6].detach()

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
        # PHASE 7: Loss + backprop
        # ============================================================
        loss, info = compute_loss(
            h=h_agents,
            h_dot=h_dot,
            pi_action=action_flat,
            u_qp=u_qp,
            safe_mask=safe_flat,
            unsafe_mask=unsafe_flat,
            alpha=alpha,
            eps=eps,
            coef_safe=coef_safe,
            coef_unsafe=coef_unsafe,
            coef_h_dot=coef_h_dot,
            coef_action=coef_action,
        )

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

    # ---- Save ----
    ckpt = {
        "gcbf_net": gcbf_net.state_dict(),
        "policy_net": policy_net.state_dict(),
        "config": {
            "num_agents": num_agents,
            "area_size": area_size,
            "node_dim": vec_env.node_dim,
            "edge_dim": vec_env.edge_dim,
            "action_dim": vec_env.action_dim,
            "state_dim": vec_env.state_dim,
            "comm_radius": vec_env.comm_radius,
            "R_form": R_form,
            "n_obs": n_obs,
            "dt": vec_env.dt,
        },
        "history": history,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"  Checkpoint saved to: {checkpoint_path}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train GCBF+ Swarm (vectorized)")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--area_size", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr_cbf", type=float, default=1e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="gcbf_swarm_checkpoint.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    a = vars(args)
    a["checkpoint_path"] = a.pop("checkpoint")
    train(**a)


if __name__ == "__main__":
    main()
