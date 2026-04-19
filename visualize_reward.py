#!/usr/bin/env python3
"""
1エピソードの報酬成分をステップごとに可視化するスクリプト。

Usage:
    python visualize_reward.py                          # ランダムポリシー、seed=42
    python visualize_reward.py --checkpoint path.pt     # 学習済みチェックポイント
    python visualize_reward.py --seed 123               # シード指定
    python visualize_reward.py --max_steps 200          # エピソード長
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gcbf_plus.env.vectorized_swarm import VectorizedSwarmEnv
from gcbf_plus.nn import PolicyNetwork
from gcbf_plus.algo.reward import compute_reward
from gcbf_plus.algo.affine_qp_solver import solve_affine_qp
from gcbf_plus.algo.nominal_controller import NominalController
from gcbf_plus.utils.swarm_graph import build_vectorized_swarm_graph


def extract_agent_outputs(full_output, n_agents, n_nodes_per_sample, n_samples):
    offsets = torch.arange(n_samples, device=full_output.device) * n_nodes_per_sample
    agent_offsets = torch.arange(n_agents, device=full_output.device)
    idx = offsets.unsqueeze(1) + agent_offsets.unsqueeze(0)
    return full_output[idx.reshape(-1)]


def run_episode(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # ---- 環境セットアップ (batch_size=1 で1エピソード) ----
    B = 1
    num_agents = args.num_agents
    n_obs = args.n_obs
    area_size = args.area_size
    max_steps = args.max_steps

    env_params = {"n_obs": n_obs, "use_payload": False}
    env = VectorizedSwarmEnv(
        num_agents=num_agents,
        batch_size=B,
        area_size=area_size,
        max_steps=max_steps,
        params=env_params,
    )
    env.reset(dev, seed=args.seed)

    # ---- ポリシーネットワーク ----
    N_per = num_agents * 2 + n_obs
    policy_net = PolicyNetwork(
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        action_dim=3,
        n_agents=num_agents,
    ).to(dev)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=dev)
        state = ckpt.get("policy_net", ckpt.get("model_state_dict", ckpt))
        policy_net.load_state_dict(state)
        print(f"Checkpoint loaded: {args.checkpoint}")
    else:
        print("No checkpoint — using random policy.")

    policy_net.eval()

    # ---- コントローラ定数 ----
    mass = env.params["mass"]
    u_max = env.params.get("u_max")
    v_max = env.params.get("v_max", 1.0)
    R_form = env.params["R_form"]
    r_margin = env.params["r_margin"]
    s_min = env.params["s_min"]
    s_max = env.params["s_max"]
    cable_length = env.params["cable_length"]
    gravity = env.params["gravity"]
    gamma_min = env.params["gamma_min"]
    gamma_max_full = env.params["gamma_max_full"]
    payload_damping = env.params["payload_damping"]

    nominal_ctrl = NominalController(
        comm_radius=env.params["comm_radius"],
        u_max=u_max,
        u_max_scale=u_max * 0.3,
        K_s_pos=env.params.get("K_s_pos", 1.0),
        K_s=env.params.get("K_s", 2.0),
    )

    a_max_gnn = 2.0
    a_max_gnn_s = 0.5

    # ---- ロールアウト ----
    history = {
        "total": [], "progress": [], "arrival": [],
        "qp": [], "avoid": [], "dist_to_goal": [],
    }

    print(f"\nRunning episode (max_steps={max_steps}) ...")

    with torch.no_grad():
        for step in range(max_steps):
            # GNN forward
            mega = env.build_batch_graph()
            pi_raw = policy_net.gnn_layers[0](mega)
            pi_tanh = torch.tanh(pi_raw)
            pi_agents = extract_agent_outputs(pi_tanh, num_agents, N_per, B)
            pi_agents = pi_agents.reshape(B, num_agents, 3)

            pi_scaled = pi_agents.clone()
            pi_scaled[:, :, :2] *= a_max_gnn
            pi_scaled[:, :, 2] *= a_max_gnn_s

            u_nom = nominal_ctrl(
                env._agent_states, env._goal_states, env._scale_states,
            ) + pi_scaled

            # QP solve
            BN = B * num_agents
            u_nom_flat = u_nom.reshape(BN, 3)
            pos_flat = env._agent_states[:, :, :2].reshape(BN, 2)
            vel_flat = env._agent_states[:, :, 2:4].reshape(BN, 2)
            s_flat = env._scale_states[:, :, 0].reshape(BN)
            s_dot_flat = env._scale_states[:, :, 1].reshape(BN)
            ps_flat = env._payload_states.reshape(BN, 4)

            n_obs_qp = env._obstacle_states.shape[1]
            obs_hits_flat = (
                env._obstacle_states[:, :, :2]
                .unsqueeze(1).expand(B, num_agents, n_obs_qp, 2)
                .reshape(BN, n_obs_qp, 2)
            ) if n_obs_qp > 0 else None
            obs_radii_qp = (
                env._obstacle_states[:, :, 2]
                .unsqueeze(1).expand(B, num_agents, n_obs_qp)
                .reshape(BN, n_obs_qp)
            ) if n_obs_qp > 0 else None

            if num_agents > 1:
                agent_idx = torch.arange(num_agents, device=dev)
                mask = agent_idx.unsqueeze(0) != agent_idx.unsqueeze(1)
                mask_b = mask.unsqueeze(0).expand(B, num_agents, num_agents)
                pos_other = env._agent_states[:, :, :2].unsqueeze(1).expand(B, num_agents, num_agents, 2)
                other_pos_flat = pos_other[mask_b].view(BN, num_agents - 1, 2)
                vel_other = env._agent_states[:, :, 2:4].unsqueeze(1).expand(B, num_agents, num_agents, 2)
                other_vel_flat = vel_other[mask_b].view(BN, num_agents - 1, 2)
                s_other = env._scale_states[:, :, 0].unsqueeze(1).expand(B, num_agents, num_agents)
                other_s_flat = s_other[mask_b].view(BN, num_agents - 1)
                sd_other = env._scale_states[:, :, 1].unsqueeze(1).expand(B, num_agents, num_agents)
                other_sd_flat = sd_other[mask_b].view(BN, num_agents - 1)
            else:
                other_pos_flat = other_vel_flat = other_s_flat = other_sd_flat = None

            u_qp_flat = solve_affine_qp(
                u_nom=u_nom_flat,
                obs_hits=obs_hits_flat,
                obs_radii=obs_radii_qp,
                agent_pos=pos_flat, agent_vel=vel_flat,
                s=s_flat, s_dot=s_dot_flat,
                other_agent_pos=other_pos_flat,
                other_agent_vel=other_vel_flat,
                other_agent_s=other_s_flat,
                other_agent_s_dot=other_sd_flat,
                R_form=R_form, r_margin=r_margin, mass=mass,
                s_min=s_min, s_max=s_max,
                payload_states=None,
                cable_length=cable_length, gravity=gravity,
                gamma_min=gamma_min, gamma_max_full=gamma_max_full,
                payload_damping=payload_damping,
                u_max=u_max,
            )
            u_qp = u_qp_flat.reshape(B, num_agents, 3)

            # 報酬計算 (step前の位置で距離を計算)
            pos_before = env._agent_states[:, :, :2].clone()
            goal_pos = env._goal_states[:, :, :2]
            dist_before = (pos_before - goal_pos).norm(dim=-1)  # (B, n)

            env.step(u_qp)

            pos_after = env._agent_states[:, :, :2]
            dist_after = (pos_after - goal_pos).norm(dim=-1)  # (B, n)
            dist_reduction = (dist_before - dist_after).reshape(BN)
            dist_to_goal_flat = dist_after.reshape(BN)

            _, reward_info = compute_reward(
                pi_action=pi_agents.reshape(BN, 3),
                u_nom=u_nom_flat,
                u_qp=u_qp_flat,
                dist_reduction=dist_reduction,
                dist_to_goal=dist_to_goal_flat,
                agent_pos=pos_flat,
                agent_vel=vel_flat,
                obs_centers=obs_hits_flat,
                obs_radii=obs_radii_qp,
            )

            history["total"].append(reward_info["reward/total"])
            history["progress"].append(reward_info["reward/progress"])
            history["arrival"].append(reward_info["reward/arrival"])
            history["qp"].append(reward_info["reward/qp"])
            history["avoid"].append(reward_info["reward/avoid"])
            history["dist_to_goal"].append(dist_after.mean().item())

            # エピソード終了チェック
            done = env.get_done_masks()
            if done.any():
                print(f"  Episode ended at step {step+1}")
                break

    steps = list(range(1, len(history["total"]) + 1))
    print(f"  Steps recorded: {len(steps)}")
    print(f"  Final dist_to_goal: {history['dist_to_goal'][-1]:.3f} m")
    print(f"  Cumulative total reward: {sum(history['total']):.3f}")

    # ---- プロット ----
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        f"Episode Reward Breakdown  (seed={args.seed}, agents={num_agents}, n_obs={n_obs})\n"
        f"checkpoint={os.path.basename(args.checkpoint) if args.checkpoint else 'random policy'}",
        fontsize=12,
    )

    def plot(ax, data, label, color, ylabel=None):
        ax.plot(steps, data, color=color, linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel or label)
        ax.grid(True, alpha=0.3)
        # 累積値をテキスト表示
        ax.text(0.98, 0.02, f"sum={sum(data):.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    plot(axes[0, 0], history["total"],    "Total Reward",    "black")
    plot(axes[0, 1], history["dist_to_goal"], "Dist to Goal (m)", "purple", "m")
    plot(axes[1, 0], history["progress"], "R_progress",      "steelblue")
    plot(axes[1, 1], history["arrival"],  "R_arrival",       "green")
    plot(axes[2, 0], history["qp"],       "R_qp",            "red")
    plot(axes[2, 1], history["avoid"],    "R_avoid",         "orange")

    plt.tight_layout()
    out_path = args.output
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--n_obs", type=int, default=6)
    parser.add_argument("--area_size", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--output", type=str, default="reward_episode.png")
    args = parser.parse_args()
    run_episode(args)
