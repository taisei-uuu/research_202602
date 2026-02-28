#!/usr/bin/env python3
"""
GCBF+ Training Loop — Swarm variant (4D Bounding Circle, VECTORIZED).

Simplified from 6D rigid body to 4D point mass with r_swarm = 0.4 bounding
circle.  Functionally identical to the baseline DoubleIntegrator training
but with a larger effective collision radius.

  State:  4D [px, py, vx, vy]
  Action: 2D [ax, ay]
  Edges:  4D [Δpx, Δpy, Δvx, Δvy]

Usage:
    python -m gcbf_plus.train_swarm --num_agents 3 --num_steps 10000
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
from gcbf_plus.utils.swarm_graph import build_vectorized_swarm_graph


def extract_agent_outputs(
    full_output: torch.Tensor,
    n_agents: int,
    n_nodes_per_sample: int,
    n_samples: int,
) -> torch.Tensor:
    """Extract agent-only rows from mega-graph GNN output."""
    offsets = torch.arange(n_samples, device=full_output.device) * n_nodes_per_sample
    agent_offsets = torch.arange(n_agents, device=full_output.device)
    idx = offsets.unsqueeze(1) + agent_offsets.unsqueeze(0)
    return full_output[idx.reshape(-1)]


def train(
    num_agents: int = 3,
    area_size: float = 4.0,
    num_steps: int = 10000,
    batch_size: int = 256,
    horizon: int = 32,
    lr_cbf: float = 1e-4,
    lr_actor: float = 1e-4,
    alpha: float = 1.0,
    eps: float = 0.02,
    coef_safe: float = 1.0,
    coef_unsafe: float = 2.0,
    coef_h_dot: float = 0.2,
    coef_action: float = 0.01,
    max_grad_norm: float = 2.0,
    log_interval: int = 100,
    seed: int = 0,
    checkpoint_path: str = "gcbf_swarm_checkpoint.pt",
    device: str = "auto",
) -> Dict[str, list]:
    """Train GCBF+ swarm networks (4D bounding circle, vectorized)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"  Device: {dev}")

    # ---- Env ----
    n_obs = 2
    vec_env = VectorizedSwarmEnv(
        num_agents=num_agents,
        batch_size=batch_size,
        area_size=area_size,
        params={"comm_radius": 2.0, "n_obs": n_obs},
    )

    # ---- Networks (4D state, 4D edges, 2D action) ----
    gcbf_net = GCBFNetwork(
        node_dim=vec_env.node_dim,    # 3
        edge_dim=vec_env.edge_dim,    # 4
        n_agents=num_agents,
    ).to(dev)
    policy_net = PolicyNetwork(
        node_dim=vec_env.node_dim,    # 3
        edge_dim=vec_env.edge_dim,    # 4
        action_dim=vec_env.action_dim,  # 2
        n_agents=num_agents,
    ).to(dev)

    optim_cbf = torch.optim.Adam(gcbf_net.parameters(), lr=lr_cbf)
    optim_actor = torch.optim.Adam(policy_net.parameters(), lr=lr_actor)

    # ---- Constants ----
    B_mat = torch.tensor(vec_env.g_x_matrix, dtype=torch.float32, device=dev)  # (4, 2)
    K_mat = vec_env._K.to(dev)  # (2, 4)
    mass = vec_env.params["mass"]
    u_max = vec_env.params.get("u_max")
    r_swarm = vec_env.params["r_swarm"]
    N_per = num_agents * 2 + n_obs
    T_loss = horizon - 1

    history: Dict[str, list] = {
        "step": [], "loss/total": [], "loss/safe": [], "loss/unsafe": [],
        "loss/h_dot": [], "loss/action": [], "acc/safe": [], "acc/unsafe": [],
        "acc/h_dot": [],
    }

    print("=" * 60)
    print(f"  GCBF+ Swarm Training (4D Bounding Circle, vectorized)")
    print(f"  swarms={num_agents}  batch={batch_size}  horizon={horizon}"
          f"  r_swarm={r_swarm}")
    print(f"  State=4D  Action=2D  Edge=4D  Nodes/sample={N_per}")
    print(f"  coef_action={coef_action}  coef_safe={coef_safe}"
          f"  coef_unsafe={coef_unsafe}  coef_h_dot={coef_h_dot}")
    print("=" * 60)
    t_start = time.time()

    for step in range(1, num_steps + 1):

        # ============================================================
        # PHASE 1: Vectorized data collection (no_grad)
        # ============================================================
        vec_env.reset(dev)
        goal_fixed = vec_env._goal_states.clone()
        obs_fixed = vec_env._obstacle_states.clone()

        all_agent_states = []
        all_unsafe = []

        with torch.no_grad():
            for t in range(horizon):
                all_agent_states.append(vec_env._agent_states.clone())
                all_unsafe.append(vec_env.unsafe_mask())

                mega = vec_env.build_batch_graph()
                pi_all = policy_net.gnn_layers[0](mega)
                pi_agents = extract_agent_outputs(pi_all, num_agents, N_per, batch_size)

                u_ref = vec_env.nominal_controller()
                u = 2.0 * pi_agents.reshape(batch_size, num_agents, 2) + u_ref
                vec_env.step(u)

        # Safe labels
        unsafe_stack = torch.stack(all_unsafe)  # (T, B, n)
        safe_list = []
        for t in range(horizon):
            was_ever = unsafe_stack[:t+1].any(dim=0)
            safe_t = ~was_ever
            if t == 0:
                safe_t = torch.ones(batch_size, num_agents, dtype=torch.bool, device=dev)
            safe_list.append(safe_t)

        # ============================================================
        # PHASE 2: Reshape for training
        # ============================================================
        S = T_loss * batch_size
        agent_4d = torch.stack(all_agent_states[:T_loss]).reshape(S, num_agents, 4)
        goal_rep = goal_fixed.unsqueeze(0).expand(T_loss, -1, -1, -1).reshape(S, num_agents, 4)
        obs_rep = obs_fixed.unsqueeze(0).expand(T_loss, -1, -1, -1).reshape(S, n_obs, 4)

        safe_flat = torch.stack(safe_list[:T_loss]).reshape(-1)
        unsafe_flat = torch.stack(all_unsafe[:T_loss]).reshape(-1)

        # ============================================================
        # PHASE 3: Gradient-tracked mega-graph
        # ============================================================
        agent_grad = agent_4d.detach().requires_grad_(True)

        mega = build_vectorized_swarm_graph(
            agent_states=agent_grad,
            goal_states=goal_rep,
            obstacle_states=obs_rep,
            comm_radius=vec_env.comm_radius,
            node_dim=vec_env.node_dim,
            edge_dim=vec_env.edge_dim,
        )

        # ============================================================
        # PHASE 4: GNN forward
        # ============================================================
        h_all = gcbf_net.gnn_layers[0](mega)
        pi_all = policy_net.gnn_layers[0](mega)

        h_agents = extract_agent_outputs(h_all, num_agents, N_per, S).squeeze(-1)
        pi_agents = extract_agent_outputs(pi_all, num_agents, N_per, S)

        # ============================================================
        # PHASE 5: u_ref, action, Lie derivative
        # ============================================================
        states_flat = agent_grad.reshape(-1, 4)     # (S*n, 4)
        goals_flat = goal_rep.reshape(-1, 4)

        err = states_flat[:, :4].detach() - goals_flat
        u_ref_flat = -torch.einsum("...j,ij->...i", err, K_mat)  # (S*n, 2)
        if u_max is not None:
            u_ref_flat = torch.clamp(u_ref_flat, -u_max, u_max)

        action_flat = 2.0 * pi_agents + u_ref_flat

        # Clamp for dynamics
        action_c = action_flat.clone()
        if u_max is not None:
            action_c = torch.clamp(action_c, -u_max, u_max)

        vel = states_flat[:, 2:4]
        accel = action_c / mass
        x_dot = torch.cat([vel, accel], dim=1)  # (S*n, 4)

        # Lie derivative
        dh_dx_3d = torch.autograd.grad(
            h_agents.sum(), agent_grad,
            create_graph=True, retain_graph=True,
        )[0]
        dh_dx = dh_dx_3d.reshape(-1, 4)
        h_dot = (dh_dx * x_dot).sum(dim=-1)

        # ============================================================
        # PHASE 6: QP solve
        # ============================================================
        with torch.no_grad():
            x_dot_f = torch.zeros_like(states_flat)
            x_dot_f[:, :2] = states_flat[:, 2:4].detach()

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
            h=h_agents, h_dot=h_dot,
            pi_action=action_flat, u_qp=u_qp,
            safe_mask=safe_flat, unsafe_mask=unsafe_flat,
            alpha=alpha, eps=eps,
            coef_safe=coef_safe, coef_unsafe=coef_unsafe,
            coef_h_dot=coef_h_dot, coef_action=coef_action,
        )

        optim_cbf.zero_grad()
        optim_actor.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gcbf_net.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
        optim_cbf.step()
        optim_actor.step()

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
            "r_swarm": r_swarm,
            "R_form": vec_env.params["R_form"],
            "n_obs": n_obs,
            "dt": vec_env.dt,
        },
        "history": history,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"  Checkpoint saved to: {checkpoint_path}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train GCBF+ Swarm (4D bounding circle)")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--area_size", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr_cbf", type=float, default=1e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--coef_action", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="gcbf_swarm_checkpoint.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    a = vars(args)
    a["checkpoint_path"] = a.pop("checkpoint")
    train(**a)


if __name__ == "__main__":
    main()
