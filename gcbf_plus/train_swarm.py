#!/usr/bin/env python3
"""
Hierarchical Velocity-Command Swarm Training Loop.

Architecture (4 Levels):
    Level 1: GNN π(x) → tanh → velocity commands (Δv_x, Δv_y, ṡ_target)
             v_target = K_pos*(goal-pos) + v_GNN_offset
    Level 2: PD Controller → a_nom = K_v*(v_target - v_current)
    Level 3: Analytical QP (HOCBF→Obs-CBF→Scale-CBF, Dykstra iteration)
    Level 4: env.step(X*) → distribute to individual drones

Data collection:
    Roll out `horizon` steps × `batch_size` envs → pool of N_total samples.
    Shuffle pool, draw `mini_batch_size` samples, train for `n_epochs` epochs.

Loss:
    L_goal:  Goal‐reaching incentive
    L_qp:   QP‐intervention penalty (||u_nom - X*||²)
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


def _velocity_to_accel(
    pi_scaled: torch.Tensor,
    agent_states: torch.Tensor,
    goal_states: torch.Tensor,
    scale_states: torch.Tensor,
    K_pos: float,
    K_v: float,
    K_s: float,
    v_max: float,
    u_max: float = None,
    mass: float = 0.1,
    n_drones: int = 3,
) -> torch.Tensor:
    """Level 1+2: GNN velocity commands → PD → nominal acceleration (pre-clamped).

    Parameters
    ----------
    pi_scaled : (..., 3) — (Δv_x, Δv_y, ṡ_target) already in physical units.
    agent_states : (..., 4) — [px, py, vx, vy]
    goal_states : (..., 4) — [gx, gy, gvx, gvy]
    scale_states : (..., 2) — [s, s_dot]
    u_max : float or None — per-motor thrust limit. If given, pre-clamp output.
    mass  : float — per-drone mass (for computing feasible acceleration).
    n_drones : int — drones per swarm (default 3).

    Returns
    -------
    u_nom : (..., 3) — [a_cx_nom, a_cy_nom, a_s_nom]
    """
    # Level 1: target velocity
    pos = agent_states[..., :2]
    goal_pos = goal_states[..., :2]
    v_current = agent_states[..., 2:4]

    v_ref = K_pos * (goal_pos - pos)
    v_ref = torch.clamp(v_ref, -v_max, v_max)
    v_target = v_ref + pi_scaled[..., :2]

    s_dot_target = pi_scaled[..., 2]
    s_dot_current = scale_states[..., 1]

    # Level 2: PD controller
    a_trans = K_v * (v_target - v_current)
    a_s = K_s * (s_dot_target - s_dot_current)

    u_nom = torch.cat([a_trans, a_s.unsqueeze(-1)], dim=-1)

    # Pre-clamp: keep u_nom within physically feasible range (out-of-place for autograd)
    if u_max is not None:
        a_max_trans = n_drones * u_max / mass * 0.7   # 70% margin for translation
        a_max_scale = n_drones * u_max / mass * 0.3   # 30% margin for scale
        
        # NEW: Payload-aware clamping
        # The payload HOCBF strictly limits acceleration to ~0.4 m/s^2.
        # If LQR requests more, QP intervenes massively (L_qp explodes),
        # causing the GNN to learn an arbitrary bias to negate the LQR.
        a_max_payload = 0.5
        actual_a_max_t = min(a_max_trans, a_max_payload)
        
        clamped_trans = u_nom[..., :2].clamp(-actual_a_max_t, actual_a_max_t)
        clamped_scale = u_nom[..., 2:].clamp(-a_max_scale, a_max_scale)
        u_nom = torch.cat([clamped_trans, clamped_scale], dim=-1)

    return u_nom


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
    coef_scale: float = 0.5,
    max_grad_norm: float = 2.0,
    log_interval: int = 100,
    seed: int = 0,
    checkpoint_path: str = "affine_swarm_checkpoint.pt",
    device: str = "auto",
) -> Dict[str, list]:
    """Train hierarchical velocity-command swarm policy."""
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

    # ---- Network (GNN outputs 3D velocity commands) ----
    policy_net = PolicyNetwork(
        node_dim=vec_env.node_dim,    # 3
        edge_dim=vec_env.edge_dim,    # 4
        action_dim=3,                 # (Δv_x, Δv_y, ṡ_target)
        n_agents=num_agents,
    ).to(dev)

    optim = torch.optim.Adam(policy_net.parameters(), lr=lr_actor)

    # ---- Constants ----
    mass = vec_env.params["mass"]
    u_max = vec_env.params.get("u_max")
    v_max = vec_env.params.get("v_max", 1.0)
    s_dot_max = vec_env.params.get("s_dot_max", 1.0)
    R_form = vec_env.params["R_form"]
    r_margin = vec_env.params["r_margin"]
    s_min = vec_env.params["s_min"]
    s_max = vec_env.params["s_max"]
    N_per = num_agents * 2 + n_obs

    # Hierarchical velocity-command gains
    K_pos = vec_env.params.get("K_pos", 0.5)
    K_v = vec_env.params.get("K_v", 2.0)
    K_s = vec_env.params.get("K_s", 2.0)

    # Payload / HOCBF constants
    cable_length = vec_env.params["cable_length"]
    gravity = vec_env.params["gravity"]
    gamma_min = vec_env.params["gamma_min"]
    gamma_max_full = vec_env.params["gamma_max_full"]
    payload_damping = vec_env.params["payload_damping"]

    # Pool size
    N_pool = horizon * batch_size

    history: Dict[str, list] = {
        "step": [], "loss/total": [], "loss/goal": [],
        "loss/qp": [], "loss/scale": [],
    }

    print("=" * 60)
    print(f"  Hierarchical Velocity-Command Swarm Training")
    print(f"  swarms={num_agents}  batch={batch_size}  horizon={horizon}"
          f"  area={area_size}")
    print(f"  pool_size={N_pool}  mini_batch={mini_batch_size}  epochs={n_epochs}")
    print(f"  State=4D  Action=3D(vel_cmd)  Edge=4D  Nodes/sample={N_per}")
    print(f"  R_form={R_form}  s_min={s_min}  s_max={s_max}")
    print(f"  K_pos={K_pos}  K_v={K_v}  K_s={K_s}")
    print(f"  coef_goal={coef_goal}  coef_qp={coef_qp}  coef_scale={coef_scale}")
    print("=" * 60)
    t_start = time.time()

    for step in range(1, num_steps + 1):

        # ============================================================
        # PHASE 1: Vectorized data collection (no_grad)
        # ============================================================
        vec_env.reset(dev)
        goal_fixed = vec_env._goal_states.clone()
        obs_fixed = vec_env._obstacle_states.clone()
        obs_centers = vec_env._obstacle_centers.clone() if vec_env._obstacle_centers is not None else None
        obs_half_sizes = vec_env._obstacle_half_sizes.clone() if vec_env._obstacle_half_sizes is not None else None

        pool_agent = []
        pool_scale = []
        pool_payload = []
        pool_goal = []
        pool_obs = []

        with torch.no_grad():
            for t in range(horizon):
                pool_agent.append(vec_env._agent_states.clone())
                pool_scale.append(vec_env._scale_states.clone())
                pool_payload.append(vec_env._payload_states.clone())
                pool_goal.append(goal_fixed.clone())
                pool_obs.append(obs_fixed.clone())

                # Level 1: GNN → tanh → scale to physical velocity
                mega = vec_env.build_batch_graph()
                pi_raw = policy_net.gnn_layers[0](mega)  # no tanh yet (raw)
                pi_tanh = torch.tanh(pi_raw)
                pi_agents = extract_agent_outputs(pi_tanh, num_agents, N_per, batch_size)
                pi_agents = pi_agents.reshape(batch_size, num_agents, 3)

                # Scale to physical units: (Δv_x, Δv_y) * v_max, ṡ * s_dot_max
                pi_scaled = pi_agents.clone()
                pi_scaled[:, :, :2] *= v_max
                pi_scaled[:, :, 2] *= s_dot_max

                # Level 1+2: velocity → PD → acceleration (pre-clamped)
                u_nom = _velocity_to_accel(
                    pi_scaled, vec_env._agent_states, goal_fixed,
                    vec_env._scale_states, K_pos, K_v, K_s, v_max,
                    u_max=u_max, mass=mass,
                )

                # Level 3: QP solve
                BN = batch_size * num_agents
                u_nom_flat = u_nom.reshape(BN, 3)
                pos_flat = vec_env._agent_states[:, :, :2].reshape(BN, 2)
                vel_flat = vec_env._agent_states[:, :, 2:4].reshape(BN, 2)
                s_flat = vec_env._scale_states[:, :, 0].reshape(BN)
                s_dot_flat = vec_env._scale_states[:, :, 1].reshape(BN)
                ps_flat = vec_env._payload_states.reshape(BN, 4)

                # Use LiDAR hits for QP instead of centers
                lidar_hits = vec_env.get_lidar_hits(num_beams=16) # (B, n, nb, 4)
                obs_hits_flat = lidar_hits[..., :2].reshape(BN, 16, 2)
                
                # Agent-Agent info
                if num_agents > 1:
                    dev = vec_env._agent_states.device
                    agent_idx = torch.arange(num_agents, device=dev)
                    mask = agent_idx.unsqueeze(0) != agent_idx.unsqueeze(1) # (n, n)
                    mask_b = mask.unsqueeze(0).expand(batch_size, num_agents, num_agents)
                    
                    pos_other = vec_env._agent_states[:, :, :2].unsqueeze(1).expand(batch_size, num_agents, num_agents, 2)
                    other_pos_flat = pos_other[mask_b].view(BN, num_agents - 1, 2)
                    
                    vel_other = vec_env._agent_states[:, :, 2:4].unsqueeze(1).expand(batch_size, num_agents, num_agents, 2)
                    other_vel_flat = vel_other[mask_b].view(BN, num_agents - 1, 2)
                    
                    s_other = vec_env._scale_states[:, :, 0].unsqueeze(1).expand(batch_size, num_agents, num_agents)
                    other_s_flat = s_other[mask_b].view(BN, num_agents - 1)
                    
                    sd_other = vec_env._scale_states[:, :, 1].unsqueeze(1).expand(batch_size, num_agents, num_agents)
                    other_sd_flat = sd_other[mask_b].view(BN, num_agents - 1)
                else:
                    other_pos_flat = None
                    other_vel_flat = None
                    other_s_flat = None
                    other_sd_flat = None

                u_qp_flat = solve_affine_qp(
                    u_nom=u_nom_flat,
                    obs_hits=obs_hits_flat,
                    agent_pos=pos_flat, agent_vel=vel_flat,
                    s=s_flat, s_dot=s_dot_flat,
                    other_agent_pos=other_pos_flat,
                    other_agent_vel=other_vel_flat,
                    other_agent_s=other_s_flat,
                    other_agent_s_dot=other_sd_flat,
                    R_form=R_form, r_margin=r_margin, mass=mass,
                    s_min=s_min, s_max=s_max,
                    payload_states=ps_flat,
                    cable_length=cable_length, gravity=gravity,
                    gamma_min=gamma_min, gamma_max_full=gamma_max_full,
                    payload_damping=payload_damping,
                    u_max=u_max,
                )
                u_qp = u_qp_flat.reshape(batch_size, num_agents, 3)

                # Level 4: env.step distributes to drones
                vec_env.step(u_qp)

        # ============================================================
        # PHASE 2: Flatten pool → (N_pool, n, ...) and shuffle
        # ============================================================
        all_agent = torch.stack(pool_agent).reshape(N_pool, num_agents, 4)
        all_scale = torch.stack(pool_scale).reshape(N_pool, num_agents, 2)
        all_payload = torch.stack(pool_payload).reshape(N_pool, num_agents, 4)
        all_goal = torch.stack(pool_goal).reshape(N_pool, num_agents, 4)
        all_obs_st = torch.stack(pool_obs).reshape(N_pool, n_obs, 4)

        if obs_centers is not None:
            all_obs_c = obs_centers.unsqueeze(0).expand(horizon, -1, -1, -1).reshape(N_pool, n_obs, 2)
            all_obs_hs = obs_half_sizes.unsqueeze(0).expand(horizon, -1, -1, -1).reshape(N_pool, n_obs, 2)
        else:
            all_obs_c = None
            all_obs_hs = None

        perm = torch.randperm(N_pool, device=dev)

        # ============================================================
        # PHASE 3: Mini-batch training (n_epochs passes)
        # ============================================================
        epoch_losses = []

        for epoch in range(n_epochs):
            if epoch > 0:
                perm = torch.randperm(N_pool, device=dev)

            n_batches = max(1, N_pool // mini_batch_size)

            for bi in range(n_batches):
                idx = perm[bi * mini_batch_size : (bi + 1) * mini_batch_size]
                mb_size = idx.shape[0]

                mb_agent = all_agent[idx]
                mb_scale = all_scale[idx]
                mb_payload = all_payload[idx]
                mb_goal = all_goal[idx]
                mb_obs_st = all_obs_st[idx]

                # ── Build graph (dynamic comm_radius) ──
                dyn_cr = vec_env.comm_radius * mb_scale[:, :, 0]  # (mb, n)
                mega = build_vectorized_swarm_graph(
                    agent_states=mb_agent,
                    goal_states=mb_goal,
                    obstacle_states=mb_obs_st,
                    comm_radius=dyn_cr,
                    node_dim=vec_env.node_dim,
                    edge_dim=vec_env.edge_dim,
                )

                # ── Level 1: GNN → tanh → scale ──
                pi_raw = policy_net.gnn_layers[0](mega)
                pi_tanh = torch.tanh(pi_raw)
                pi_agents = extract_agent_outputs(pi_tanh, num_agents, N_per, mb_size)
                pi_agents = pi_agents.reshape(mb_size, num_agents, 3)

                pi_scaled = pi_agents.clone()
                pi_scaled[:, :, :2] = pi_scaled[:, :, :2] * v_max
                pi_scaled[:, :, 2] = pi_scaled[:, :, 2] * s_dot_max

                # ── Level 1+2: velocity → PD → acceleration (pre-clamped) ──
                u_nom = _velocity_to_accel(
                    pi_scaled, mb_agent, mb_goal,
                    mb_scale, K_pos, K_v, K_s, v_max,
                    u_max=u_max, mass=mass,
                )
                u_nom_flat = u_nom.reshape(mb_size * num_agents, 3)

                # ── Level 3: QP solve (no grad) ──
                with torch.no_grad():
                    states_flat = mb_agent.reshape(-1, 4)
                    goals_flat = mb_goal.reshape(-1, 4)
                    s_f = mb_scale.reshape(-1, 2)[:, 0]
                    sd_f = mb_scale.reshape(-1, 2)[:, 1]
                    ps_f = mb_payload.reshape(-1, 4)

                    if all_obs_c is not None:
                        mb_obs_c = all_obs_c[idx]
                        mb_obs_hs = all_obs_hs[idx]
                        obs_c_exp = mb_obs_c.unsqueeze(1).expand(-1, num_agents, -1, -1) \
                                           .reshape(mb_size * num_agents, n_obs, 2)
                        obs_hs_exp = mb_obs_hs.unsqueeze(1).expand(-1, num_agents, -1, -1) \
                                            .reshape(mb_size * num_agents, n_obs, 2)
                    else:
                        obs_c_exp = None
                        obs_hs_exp = None

                    # Agent-Agent info
                    if num_agents > 1:
                        dev = mb_agent.device
                        agent_idx = torch.arange(num_agents, device=dev)
                        mask = agent_idx.unsqueeze(0) != agent_idx.unsqueeze(1)
                        mask_b = mask.unsqueeze(0).expand(mb_size, num_agents, num_agents)
                        
                        pos_other = mb_agent[:, :, :2].unsqueeze(1).expand(mb_size, num_agents, num_agents, 2)
                        mb_other_pos_flat = pos_other[mask_b].reshape(mb_size * num_agents, num_agents - 1, 2)
                        
                        vel_other = mb_agent[:, :, 2:4].unsqueeze(1).expand(mb_size, num_agents, num_agents, 2)
                        mb_other_vel_flat = vel_other[mask_b].reshape(mb_size * num_agents, num_agents - 1, 2)
                        
                        s_other = mb_scale[:, :, 0].unsqueeze(1).expand(mb_size, num_agents, num_agents)
                        mb_other_s_flat = s_other[mask_b].reshape(mb_size * num_agents, num_agents - 1)
                        
                        sd_other = mb_scale[:, :, 1].unsqueeze(1).expand(mb_size, num_agents, num_agents)
                        mb_other_sd_flat = sd_other[mask_b].reshape(mb_size * num_agents, num_agents - 1)
                    else:
                        mb_other_pos_flat = None
                        mb_other_vel_flat = None
                        mb_other_s_flat = None
                        mb_other_sd_flat = None

                    u_qp_flat = solve_affine_qp(
                        u_nom=u_nom_flat.detach(),
                        obs_centers=obs_c_exp, obs_half_sizes=obs_hs_exp,
                        agent_pos=states_flat[:, :2], agent_vel=states_flat[:, 2:4],
                        s=s_f, s_dot=sd_f,
                        other_agent_pos=mb_other_pos_flat,
                        other_agent_vel=mb_other_vel_flat,
                        other_agent_s=mb_other_s_flat,
                        other_agent_s_dot=mb_other_sd_flat,
                        R_form=R_form, r_margin=r_margin, mass=mass,
                        s_min=s_min, s_max=s_max,
                        payload_states=ps_f,
                        cable_length=cable_length, gravity=gravity,
                        gamma_min=gamma_min, gamma_max_full=gamma_max_full,
                        payload_damping=payload_damping,
                        u_max=u_max,
                    )

                # ── Loss ──
                # Extract per-agent scale and GNN scale output for L_scale
                s_flat = mb_scale[:, :, 0].reshape(-1)          # (mb*n,)
                s_dot_target_flat = pi_scaled[:, :, 2].reshape(-1)  # (mb*n,) — has grad

                loss, info = compute_affine_loss(
                    pi_action=pi_agents.reshape(-1, 3),
                    u_nom=u_nom_flat,
                    u_qp=u_qp_flat,
                    s_current=s_flat,
                    s_dot_target=s_dot_target_flat,
                    s_max=s_max,
                    coef_goal=coef_goal,
                    coef_qp=coef_qp,
                    coef_scale=coef_scale,
                )
                # QP intervention tracking
                qp_intervention = (u_nom_flat - u_qp_flat).pow(2).sum(dim=-1)
                info["qp_cut/mean"] = qp_intervention.mean().item()
                info["qp_cut/max"] = qp_intervention.max().item()

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

            avg_info = {}
            for k in epoch_losses[0]:
                avg_info[k] = np.mean([d[k] for d in epoch_losses])

            with torch.no_grad():
                all_s_vals = all_scale[:, :, 0]
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
                t_sc = ((all_s_vals - s_min) / (s_max - s_min + 1e-8)).clamp(0.0, 1.0)
                gamma_dyn = gamma_min + (gamma_max_full - gamma_min) * t_sc
                viol_rate = (gamma_abs > gamma_dyn).float().mean().item()
                mean_gamma_limit = gamma_dyn.mean().item()

            n_updates = len(epoch_losses)
            qp_cut_mean = avg_info.get("qp_cut/mean", 0)
            qp_cut_max = avg_info.get("qp_cut/max", 0)
            print(
                f"  step {step:5d}/{num_steps}"
                f"  |  loss {avg_info.get('loss/total', 0):.4f}"
                f"  goal {avg_info.get('loss/goal', 0):.4f}"
                f"  qp {avg_info.get('loss/qp', 0):.4f}"
                f"  scale {avg_info.get('loss/scale', 0):.4f}"
                f"  |  upd={n_updates}"
                f"  |  s: {mean_s:.2f} [{min_s:.2f},{max_s:.2f}]"
                f"  |  γ: {mean_gamma:.3f} p95={p95_gamma:.3f}"
                f"  max={max_gamma:.3f} lim={mean_gamma_limit:.3f} viol={viol_rate:.1%}"
                f"  |  cut: {qp_cut_mean:.3f}/{qp_cut_max:.3f}"
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
            "action_dim": 3,
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
            "K_pos": K_pos,
            "K_v": K_v,
            "K_s": K_s,
            "v_max": v_max,
            "s_dot_max": s_dot_max,
            "architecture": "hierarchical_velocity_command",
        },
        "history": history,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"  Checkpoint saved to: {checkpoint_path}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical Velocity-Command Swarm Policy")
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
    parser.add_argument("--coef_scale", type=float, default=0.5)
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
