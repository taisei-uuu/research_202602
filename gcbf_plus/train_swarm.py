#!/usr/bin/env python3
"""
Affine-Transform Swarm Training Loop (Vectorized + Shuffled Mini-Batch).

Architecture:
    GNN π(x) → affine offset [Δa_cx, Δa_cy, a_s]
    u_AT = u_nom + π(x)
    Analytical QP (Obs-CBF + Scale-CBF + HOCBF) → u_QP
    env.step(u_QP)

Data collection:
    Roll out `horizon` steps × `batch_size` envs → pool of N_total samples.
    Shuffle pool, draw `mini_batch_size` samples, train for `n_epochs` epochs.

Loss:
    L_goal:  Goal-reaching incentive
    L_qp:   QP-intervention penalty
    L_reg:  Action regularization

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
from gcbf_plus.nn import PolicyNetwork
from gcbf_plus.algo.loss import compute_affine_loss
from gcbf_plus.algo.affine_qp_solver import solve_affine_qp
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
    area_size: float = 15.0,
    n_obs: int = 6,
    num_steps: int = 10000,
    batch_size: int = 256,
    horizon: int = 32,
    mini_batch_size: int = 128,
    n_epochs: int = 4,
    lr_actor: float = 1e-4,
    coef_goal: float = 1.0,
    coef_qp: float = 2.0,
    coef_reg: float = 0.01,
    max_grad_norm: float = 2.0,
    log_interval: int = 100,
    seed: int = 0,
    checkpoint_path: str = "affine_swarm_checkpoint.pt",
    device: str = "auto",
) -> Dict[str, list]:
    """Train affine-transform swarm policy (vectorized + shuffled mini-batch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"  Device: {dev}")

    # ---- Env ----
    vec_env = VectorizedSwarmEnv(
        num_agents=num_agents,
        batch_size=batch_size,
        area_size=area_size,
        params={"n_obs": n_obs},
    )

    # ---- Network (GNN outputs 3D affine offset) ----
    policy_net = PolicyNetwork(
        node_dim=vec_env.node_dim,    # 3
        edge_dim=vec_env.edge_dim,    # 4
        action_dim=vec_env.action_dim,  # 3: [Δa_cx, Δa_cy, a_s]
        n_agents=num_agents,
    ).to(dev)

    optim = torch.optim.Adam(policy_net.parameters(), lr=lr_actor)

    # ---- Constants ----
    K_mat = vec_env._K.to(dev)  # (2, 4)
    mass = vec_env.params["mass"]
    u_max = vec_env.params.get("u_max")
    R_form = vec_env.params["R_form"]
    r_margin = vec_env.params["r_margin"]
    s_min = vec_env.params["s_min"]
    s_max = vec_env.params["s_max"]
    N_per = num_agents * 2 + n_obs

    # Payload / HOCBF constants (dynamic gamma)
    cable_length = vec_env.params["cable_length"]
    gravity = vec_env.params["gravity"]
    gamma_min = vec_env.params["gamma_min"]
    gamma_max_full = vec_env.params["gamma_max_full"]
    payload_damping = vec_env.params["payload_damping"]
    # Spring force constants
    k_spring = vec_env.params.get("k_spring", 0.5)
    c_spring_damp = vec_env.params.get("c_spring_damp", 0.3)

    # Pool size: horizon * batch_size (number of environment snapshots)
    N_pool = horizon * batch_size

    history: Dict[str, list] = {
        "step": [], "loss/total": [], "loss/goal": [],
        "loss/qp": [], "loss/reg": [],
    }

    print("=" * 60)
    print(f"  Affine-Transform Swarm Training (Shuffled Mini-Batch)")
    print(f"  swarms={num_agents}  batch={batch_size}  horizon={horizon}"
          f"  area={area_size}")
    print(f"  pool_size={N_pool}  mini_batch={mini_batch_size}  epochs={n_epochs}")
    print(f"  State=4D  Action=3D(affine)  Edge=4D  Nodes/sample={N_per}")
    print(f"  R_form={R_form}  s_min={s_min}  s_max={s_max}")
    print(f"  coef_goal={coef_goal}  coef_qp={coef_qp}  coef_reg={coef_reg}")
    print("=" * 60)
    t_start = time.time()

    for step in range(1, num_steps + 1):

        # ============================================================
        # PHASE 1: Vectorized data collection (no_grad)
        # ============================================================
        vec_env.reset(dev)
        goal_fixed = vec_env._goal_states.clone()       # (B, n, 4)
        obs_fixed = vec_env._obstacle_states.clone()     # (B, n_obs, 4)
        obs_centers = vec_env._obstacle_centers.clone() if vec_env._obstacle_centers is not None else None
        obs_half_sizes = vec_env._obstacle_half_sizes.clone() if vec_env._obstacle_half_sizes is not None else None

        # Collect state snapshots
        pool_agent = []     # each: (B, n, 4)
        pool_scale = []     # each: (B, n, 2)
        pool_payload = []   # each: (B, n, 4)
        pool_goal = []      # each: (B, n, 4)
        pool_obs = []       # each: (B, n_obs, 4)

        with torch.no_grad():
            for t in range(horizon):
                pool_agent.append(vec_env._agent_states.clone())
                pool_scale.append(vec_env._scale_states.clone())
                pool_payload.append(vec_env._payload_states.clone())
                pool_goal.append(goal_fixed.clone())
                pool_obs.append(obs_fixed.clone())

                # Policy forward + QP for rollout
                mega = vec_env.build_batch_graph()
                pi_all = policy_net.gnn_layers[0](mega)
                pi_agents = extract_agent_outputs(pi_all, num_agents, N_per, batch_size)
                pi_agents = pi_agents.reshape(batch_size, num_agents, 3)

                u_ref = vec_env.nominal_controller()  # (B, n, 3)
                u_at = u_ref + pi_agents

                # QP solve
                BN = batch_size * num_agents
                u_at_flat = u_at.reshape(BN, 3)
                pos_flat = vec_env._agent_states[:, :, :2].reshape(BN, 2)
                vel_flat = vec_env._agent_states[:, :, 2:4].reshape(BN, 2)
                s_flat = vec_env._scale_states[:, :, 0].reshape(BN)
                s_dot_flat = vec_env._scale_states[:, :, 1].reshape(BN)
                ps_flat = vec_env._payload_states.reshape(BN, 4)

                if obs_centers is not None:
                    obs_c_exp = obs_centers.unsqueeze(1).expand(-1, num_agents, -1, -1).reshape(BN, n_obs, 2)
                    obs_hs_exp = obs_half_sizes.unsqueeze(1).expand(-1, num_agents, -1, -1).reshape(BN, n_obs, 2)
                else:
                    obs_c_exp = None
                    obs_hs_exp = None

                u_qp_flat = solve_affine_qp(
                    u_at=u_at_flat,
                    obs_centers=obs_c_exp, obs_half_sizes=obs_hs_exp,
                    agent_pos=pos_flat, agent_vel=vel_flat,
                    s=s_flat, s_dot=s_dot_flat,
                    R_form=R_form, r_margin=r_margin, mass=mass,
                    s_min=s_min, s_max=s_max,
                    payload_states=ps_flat,
                    cable_length=cable_length, gravity=gravity,
                    gamma_min=gamma_min, gamma_max_full=gamma_max_full,
                    payload_damping=payload_damping,
                    u_max=u_max,
                )
                u_qp = u_qp_flat.reshape(batch_size, num_agents, 3)
                vec_env.step(u_qp)

        # ============================================================
        # PHASE 2: Flatten pool → (N_pool, n, ...) and shuffle
        # ============================================================
        # Stack: (horizon, B, n, dim) → (horizon*B, n, dim)
        all_agent = torch.stack(pool_agent).reshape(N_pool, num_agents, 4)
        all_scale = torch.stack(pool_scale).reshape(N_pool, num_agents, 2)
        all_payload = torch.stack(pool_payload).reshape(N_pool, num_agents, 4)
        all_goal = torch.stack(pool_goal).reshape(N_pool, num_agents, 4)
        all_obs_st = torch.stack(pool_obs).reshape(N_pool, n_obs, 4)

        # Also prepare obstacle geometry for QP (stays same within one rollout)
        if obs_centers is not None:
            # (B, n_obs, 2) → expand to (horizon, B, n_obs, 2) → (N_pool, n_obs, 2)
            all_obs_c = obs_centers.unsqueeze(0).expand(horizon, -1, -1, -1).reshape(N_pool, n_obs, 2)
            all_obs_hs = obs_half_sizes.unsqueeze(0).expand(horizon, -1, -1, -1).reshape(N_pool, n_obs, 2)
        else:
            all_obs_c = None
            all_obs_hs = None

        # Shuffle indices
        perm = torch.randperm(N_pool, device=dev)

        # ============================================================
        # PHASE 3: Mini-batch training (n_epochs passes)
        # ============================================================
        epoch_losses = []

        for epoch in range(n_epochs):
            # Re-shuffle each epoch
            if epoch > 0:
                perm = torch.randperm(N_pool, device=dev)

            n_batches = max(1, N_pool // mini_batch_size)

            for bi in range(n_batches):
                idx = perm[bi * mini_batch_size : (bi + 1) * mini_batch_size]
                mb_size = idx.shape[0]

                mb_agent = all_agent[idx]      # (mb, n, 4)
                mb_scale = all_scale[idx]      # (mb, n, 2)
                mb_payload = all_payload[idx]  # (mb, n, 4)
                mb_goal = all_goal[idx]        # (mb, n, 4)
                mb_obs_st = all_obs_st[idx]    # (mb, n_obs, 4)

                # ── Build graph for this mini-batch ──
                mega = build_vectorized_swarm_graph(
                    agent_states=mb_agent,
                    goal_states=mb_goal,
                    obstacle_states=mb_obs_st,
                    comm_radius=vec_env.comm_radius,
                    node_dim=vec_env.node_dim,
                    edge_dim=vec_env.edge_dim,
                )

                # ── GNN forward ──
                pi_all = policy_net.gnn_layers[0](mega)
                pi_agents = extract_agent_outputs(pi_all, num_agents, N_per, mb_size)
                pi_agents = pi_agents.reshape(mb_size * num_agents, 3)

                # ── Nominal control (LQR + spring force) ──
                states_flat = mb_agent.reshape(-1, 4)   # (mb*n, 4)
                goals_flat = mb_goal.reshape(-1, 4)
                err = states_flat - goals_flat
                u_ref_flat = -torch.einsum("...j,ij->...i", err, K_mat)  # (mb*n, 2)
                if u_max is not None:
                    u_ref_flat = torch.clamp(u_ref_flat, -u_max, u_max)
                # Spring-force scale component: a_s_nom = k*(s_max-s) - c*s_dot
                mb_s = mb_scale.reshape(-1, 2)[:, 0]       # (mb*n,)
                mb_sd = mb_scale.reshape(-1, 2)[:, 1]      # (mb*n,)
                a_s_nom = k_spring * (s_max - mb_s) - c_spring_damp * mb_sd
                u_ref_3d = torch.cat([u_ref_flat, a_s_nom.unsqueeze(-1)], dim=-1)

                u_at_flat = u_ref_3d + pi_agents  # (mb*n, 3)

                # ── QP solve (no grad) ──
                with torch.no_grad():
                    pos_f = states_flat[:, :2]
                    vel_f = states_flat[:, 2:4]
                    s_f = mb_scale.reshape(-1, 2)[:, 0]
                    sd_f = mb_scale.reshape(-1, 2)[:, 1]
                    ps_f = mb_payload.reshape(-1, 4)

                    if all_obs_c is not None:
                        mb_obs_c = all_obs_c[idx]    # (mb, n_obs, 2)
                        mb_obs_hs = all_obs_hs[idx]  # (mb, n_obs, 2)
                        obs_c_exp = mb_obs_c.unsqueeze(1).expand(-1, num_agents, -1, -1) \
                                           .reshape(mb_size * num_agents, n_obs, 2)
                        obs_hs_exp = mb_obs_hs.unsqueeze(1).expand(-1, num_agents, -1, -1) \
                                            .reshape(mb_size * num_agents, n_obs, 2)
                    else:
                        obs_c_exp = None
                        obs_hs_exp = None

                    u_qp_flat = solve_affine_qp(
                        u_at=u_at_flat.detach(),
                        obs_centers=obs_c_exp, obs_half_sizes=obs_hs_exp,
                        agent_pos=pos_f, agent_vel=vel_f,
                        s=s_f, s_dot=sd_f,
                        R_form=R_form, r_margin=r_margin, mass=mass,
                        s_min=s_min, s_max=s_max,
                        payload_states=ps_f,
                        cable_length=cable_length, gravity=gravity,
                        gamma_min=gamma_min, gamma_max_full=gamma_max_full,
                        payload_damping=payload_damping,
                        u_max=u_max,
                    )

                # ── Loss ──
                goal_dist = torch.norm(
                    states_flat[:, :2] - goals_flat[:, :2], dim=-1
                )

                loss, info = compute_affine_loss(
                    pi_action=pi_agents,
                    u_at=u_at_flat,
                    u_qp=u_qp_flat,
                    goal_dist=goal_dist,
                    coef_goal=coef_goal,
                    coef_qp=coef_qp,
                    coef_reg=coef_reg,
                )

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                optim.step()

                epoch_losses.append(info)

        # ============================================================
        # Logging
        # ============================================================
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t_start

            # Average info over all mini-batches
            avg_info = {}
            for k in epoch_losses[0]:
                avg_info[k] = np.mean([d[k] for d in epoch_losses])

            # Scale statistics (from collected pool)
            with torch.no_grad():
                all_s_vals = all_scale[:, :, 0]  # (N_pool, n)
                mean_s = all_s_vals.mean().item()
                min_s = all_s_vals.min().item()
                max_s = all_s_vals.max().item()
                # Payload swing — dynamic violation check
                gamma_abs = torch.sqrt(
                    all_payload[:, :, 0]**2 + all_payload[:, :, 1]**2
                )
                max_gamma = gamma_abs.max().item()
                mean_gamma = gamma_abs.mean().item()
                p95_gamma = torch.quantile(gamma_abs.float(), 0.95).item()
                # Dynamic γ_max(s) per sample
                t_sc = ((all_s_vals - s_min) / (s_max - s_min + 1e-8)).clamp(0.0, 1.0)
                gamma_dyn = gamma_min + (gamma_max_full - gamma_min) * t_sc  # (N_pool, n)
                viol_rate = (gamma_abs > gamma_dyn).float().mean().item()
                mean_gamma_limit = gamma_dyn.mean().item()

            n_updates = len(epoch_losses)
            print(
                f"  step {step:5d}/{num_steps}"
                f"  |  loss {avg_info.get('loss/total', 0):.4f}"
                f"  goal {avg_info.get('loss/goal', 0):.4f}"
                f"  qp {avg_info.get('loss/qp', 0):.4f}"
                f"  reg {avg_info.get('loss/reg', 0):.4f}"
                f"  |  updates={n_updates}"
                f"  |  s: mean={mean_s:.3f} [{min_s:.2f},{max_s:.2f}]"
                f"  |  γ: mean={mean_gamma:.3f} p95={p95_gamma:.3f}"
                f"  max={max_gamma:.3f} lim={mean_gamma_limit:.3f} viol={viol_rate:.1%}"
                f"  |  {elapsed:.1f}s"
            )
            history["step"].append(step)
            for k in history:
                if k != "step" and k in avg_info:
                    history[k].append(avg_info[k])

    print("=" * 60)
    print(f"  Training complete in {time.time() - t_start:.1f}s")
    print("=" * 60)

    ckpt = {
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
            "r_margin": r_margin,
            "s_min": s_min,
            "s_max": s_max,
            "n_obs": n_obs,
            "dt": vec_env.dt,
            "cable_length": cable_length,
            "gamma_min": gamma_min,
            "gamma_max_full": gamma_max_full,
            "architecture": "affine_transform",
        },
        "history": history,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"  Checkpoint saved to: {checkpoint_path}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Affine-Transform Swarm Policy")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--area_size", type=float, default=15.0)
    parser.add_argument("--n_obs", type=int, default=6)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--coef_goal", type=float, default=1.0)
    parser.add_argument("--coef_qp", type=float, default=2.0)
    parser.add_argument("--coef_reg", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="affine_swarm_checkpoint.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    a = vars(args)
    a["checkpoint_path"] = a.pop("checkpoint")
    train(**a)


if __name__ == "__main__":
    main()
