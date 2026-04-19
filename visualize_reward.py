#!/usr/bin/env python3
"""
1エピソードの報酬成分をステップごとに可視化するスクリプト。

visualize.py と同じ SwarmIntegrator を使うため、
同じ --seed を渡せば同じ環境が生成される。

Usage:
    python visualize_reward.py                          # ランダムポリシー、seed=42
    python visualize_reward.py --checkpoint path.pt     # 学習済みチェックポイント
    python visualize_reward.py --seed 0                 # visualize.py --seed 0 と同じ環境
    python visualize_reward.py --max_steps 200          # エピソード長
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gcbf_plus.env import SwarmIntegrator
from gcbf_plus.nn import PolicyNetwork
from gcbf_plus.algo.reward import compute_reward
from gcbf_plus.algo.affine_qp_solver import solve_affine_qp
from gcbf_plus.algo.nominal_controller import NominalController


def run_episode(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- ポリシーネットワークとコンフィグ ----
    cfg = None
    policy_net = None

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        policy_net = PolicyNetwork(
            node_dim=cfg["node_dim"],
            edge_dim=cfg["edge_dim"],
            action_dim=cfg["action_dim"],
            n_agents=cfg["num_agents"],
        )
        policy_net.load_state_dict(ckpt["policy_net"])
        policy_net.eval()
        print(f"Checkpoint loaded: {args.checkpoint}")
        print(f"  agents={cfg['num_agents']}  area={cfg['area_size']}  n_obs={cfg['n_obs']}")
    else:
        print("No checkpoint — using random policy.")

    # ---- 環境パラメータ (チェックポイント優先、CLIで上書き可) ----
    num_agents = cfg["num_agents"] if cfg else args.num_agents
    area_size  = cfg["area_size"]  if cfg else args.area_size
    n_obs      = cfg["n_obs"]      if cfg else args.n_obs
    dt         = cfg["dt"]         if cfg else 0.03
    max_steps  = args.max_steps

    env_params = cfg.copy() if cfg else {}
    env_params["n_obs"] = n_obs

    env = SwarmIntegrator(
        num_agents=num_agents,
        area_size=area_size,
        dt=dt,
        max_steps=max_steps,
        params=env_params,
    )
    env.reset(seed=args.seed)   # ← visualize.py と同じ呼び方

    if policy_net is None:
        # ランダムポリシー用のダミーネット
        policy_net = PolicyNetwork(
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            action_dim=3,
            n_agents=num_agents,
        )
        policy_net.eval()

    # ---- コントローラ定数 ----
    mass          = env.params["mass"]
    u_max         = env.params.get("u_max")
    R_form        = env.params["R_form"]
    r_margin      = env.params["r_margin"]
    s_min         = env.params["s_min"]
    s_max         = env.params["s_max"]
    cable_length  = env.params["cable_length"]
    gravity       = env.params["gravity"]
    gamma_min     = env.params["gamma_min"]
    gamma_max_full= env.params["gamma_max_full"]
    payload_damping = env.params["payload_damping"]

    nominal_ctrl = NominalController(
        comm_radius=env.params["comm_radius"],
        u_max=u_max,
        u_max_scale=u_max * 0.3,
        K_s_pos=env.params.get("K_s_pos", 1.0),
        K_s=env.params.get("K_s", 2.0),
    )

    a_max_gnn   = 2.0   # visualize_reward 独自スケール (train_swarm.py デフォルト)
    a_max_gnn_s = 0.5

    # ---- ロールアウト ----
    history = {
        "total": [], "progress": [], "arrival": [],
        "qp": [], "avoid": [], "dist_to_goal": [],
    }

    print(f"\nRunning episode (max_steps={max_steps}, seed={args.seed}) ...")

    with torch.no_grad():
        for step in range(max_steps):
            # GNN forward (visualize.py と同じ: policy_net(graph) → (n, 3) tanh済み)
            graph = env._get_graph()
            pi_tanh = policy_net(graph)          # (n, 3) in [-1, 1]
            pi_scaled = pi_tanh.clone()
            pi_scaled[:, :2] *= a_max_gnn
            pi_scaled[:, 2]  *= a_max_gnn_s

            # Nominal + GNN offset → u_nom (n, 3)
            u_nom = nominal_ctrl(
                env.agent_states, env.goal_states, env.scale_states,
            ) + pi_scaled

            # QP solve: SwarmIntegrator は (n, 3) / (n, 2) / (n,) の形
            n = num_agents
            pos_flat = env.agent_states[:, :2]   # (n, 2)
            vel_flat = env.agent_states[:, 2:4]  # (n, 2)
            s_flat   = env.scale_states[:, 0]    # (n,)
            s_dot_flat = env.scale_states[:, 1]  # (n,)

            n_obs_env = env._obstacle_states.shape[0] if env._obstacle_states is not None else 0
            obs_hits_flat = (
                env._obstacle_states[:, :2]
                .unsqueeze(0).expand(n, n_obs_env, 2)
            ) if n_obs_env > 0 else None
            obs_radii_flat = (
                env._obstacle_states[:, 2]
                .unsqueeze(0).expand(n, n_obs_env)
            ) if n_obs_env > 0 else None

            if n > 1:
                agent_idx = torch.arange(n)
                mask = agent_idx.unsqueeze(0) != agent_idx.unsqueeze(1)
                pos_exp = pos_flat.unsqueeze(0).expand(n, n, 2)
                other_pos = pos_exp[mask].reshape(n, n - 1, 2)
                vel_exp = vel_flat.unsqueeze(0).expand(n, n, 2)
                other_vel = vel_exp[mask].reshape(n, n - 1, 2)
                s_exp = s_flat.unsqueeze(0).expand(n, n)
                other_s = s_exp[mask].reshape(n, n - 1)
                sd_exp = s_dot_flat.unsqueeze(0).expand(n, n)
                other_sd = sd_exp[mask].reshape(n, n - 1)
            else:
                other_pos = other_vel = other_s = other_sd = None

            u_qp = solve_affine_qp(
                u_nom=u_nom,
                obs_hits=obs_hits_flat,
                obs_radii=obs_radii_flat,
                agent_pos=pos_flat, agent_vel=vel_flat,
                s=s_flat, s_dot=s_dot_flat,
                other_agent_pos=other_pos,
                other_agent_vel=other_vel,
                other_agent_s=other_s,
                other_agent_s_dot=other_sd,
                R_form=R_form, r_margin=r_margin, mass=mass,
                s_min=s_min, s_max=s_max,
                payload_states=None,
                cable_length=cable_length, gravity=gravity,
                gamma_min=gamma_min, gamma_max_full=gamma_max_full,
                payload_damping=payload_damping,
                u_max=u_max,
            )

            # 報酬計算 (step 前の距離を使う)
            pos_before = env.agent_states[:, :2].clone()
            goal_pos   = env.goal_states[:, :2]
            dist_before = (pos_before - goal_pos).norm(dim=-1)  # (n,)

            _, info = env.step(u_qp)

            dist_after = (env.agent_states[:, :2] - goal_pos).norm(dim=-1)  # (n,)
            dist_reduction = dist_before - dist_after  # (n,)

            _, reward_info = compute_reward(
                pi_action=pi_tanh,
                u_nom=u_nom,
                u_qp=u_qp,
                dist_reduction=dist_reduction,
                dist_to_goal=dist_after,
                agent_pos=pos_flat,
                agent_vel=vel_flat,
                obs_centers=obs_hits_flat,
                obs_radii=obs_radii_flat,
            )

            history["total"].append(reward_info["reward/total"])
            history["progress"].append(reward_info["reward/progress"])
            history["arrival"].append(reward_info["reward/arrival"])
            history["qp"].append(reward_info["reward/qp"])
            history["avoid"].append(reward_info["reward/avoid"])
            history["dist_to_goal"].append(dist_after.mean().item())

            if info["done"]:
                reason = "collision" if info["collision"] else "goal/timeout"
                print(f"  Episode ended at step {step+1} ({reason})")
                break

    steps = list(range(1, len(history["total"]) + 1))
    print(f"  Steps recorded   : {len(steps)}")
    print(f"  Final dist_to_goal: {history['dist_to_goal'][-1]:.3f} m")
    print(f"  Cumulative reward : {sum(history['total']):.3f}")

    # ---- プロット ----
    ckpt_label = os.path.basename(args.checkpoint) if args.checkpoint else "random policy"
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        f"Episode Reward Breakdown  (seed={args.seed}, agents={num_agents}, n_obs={n_obs})\n"
        f"checkpoint={ckpt_label}",
        fontsize=12,
    )

    def plot(ax, data, label, color, ylabel=None):
        ax.plot(steps, data, color=color, linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel or label)
        ax.grid(True, alpha=0.3)
        ax.text(0.98, 0.02, f"sum={sum(data):.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    plot(axes[0, 0], history["total"],        "Total Reward",      "black")
    plot(axes[0, 1], history["dist_to_goal"], "Dist to Goal (m)",  "purple", "m")
    plot(axes[1, 0], history["progress"],     "R_progress",        "steelblue")
    plot(axes[1, 1], history["arrival"],      "R_arrival",         "green")
    plot(axes[2, 0], history["qp"],           "R_qp",              "red")
    plot(axes[2, 1], history["avoid"],        "R_avoid",           "orange")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--n_obs",      type=int, default=6)
    parser.add_argument("--area_size",  type=float, default=10.0)
    parser.add_argument("--max_steps",  type=int, default=512)
    parser.add_argument("--output",     type=str, default="reward_episode.png")
    args = parser.parse_args()
    run_episode(args)
