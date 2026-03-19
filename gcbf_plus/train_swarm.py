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
    s_max: float = 1.5,
    K_s_pos: float = 1.0,
    u_max: float = None,
    mass: float = 0.1,
    n_drones: int = 3,
    use_payload: bool = True,
) -> torch.Tensor:
    """Level 1+2: GNN velocity commands → PD → nominal acceleration (pre-clamped).

    Parameters
    ----------
    pi_scaled : (..., 3) — (Δv_x, Δv_y, Δṡ) already in physical units.
    agent_states : (..., 4) — [px, py, vx, vy]
    goal_states : (..., 4) — [gx, gy, gvx, gvy]
    scale_states : (..., 2) — [s, s_dot]
    s_max : float — maximum scale value (expansion target).
    K_s_pos : float — proportional gain for scale expansion PD.
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

    dist = torch.norm(goal_pos - pos, dim=-1, keepdim=True)
    unit_vec = (goal_pos - pos) / (dist + 1e-6)
    
    # v_ref = unit_vec * min(v_max, K_pos * dist)
    v_ref = unit_vec * torch.clamp(K_pos * dist, max=v_max)
    v_ref = torch.clamp(v_ref, -v_max, v_max)  # Safety secondary clamp

    v_target = v_ref + pi_scaled[..., :2]

    # Scale: PD toward s_max (expansion potential) + GNN offset
    s_current = scale_states[..., 0]
    s_dot_current = scale_states[..., 1]
    s_dot_ref = K_s_pos * (s_max - s_current)  # "want to expand toward s_max"
    s_dot_target = s_dot_ref + pi_scaled[..., 2]  # GNN can correct (e.g. contract)

    # Level 2: PD controller
    a_trans = K_v * (v_target - v_current)
    a_s = K_s * (s_dot_target - s_dot_current)

    u_nom = torch.cat([a_trans, a_s.unsqueeze(-1)], dim=-1)

    # Pre-clamp: keep u_nom within physically feasible range (out-of-place for autograd)
    if u_max is not None:
        a_max_trans = n_drones * u_max / mass * 0.7   # 70% margin for translation
        a_max_scale = n_drones * u_max / mass * 0.3   # 30% margin for scale
        
        # Payload-aware clamping: when payload is active the HOCBF limits
        # acceleration to ~0.5 m/s^2, so pre-clamp u_nom to avoid massive
        # QP intervention that teaches the GNN a meaningless bias.
        if use_payload:
            a_max_payload = 0.5
            actual_a_max_t = min(a_max_trans, a_max_payload)
        else:
            actual_a_max_t = a_max_trans
        
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
    coef_effort: float = 0.3,
    w_scale: float = 2.0,
    coef_arrival: float = 5.0,
    arrival_radius: float = 0.3,
    max_grad_norm: float = 2.0,
    log_interval: int = 100,
    seed: int = 0,
    checkpoint_path: str = "affine_swarm_checkpoint.pt",
    device: str = "auto",
    use_payload: bool = True,
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
        max_steps=512,  # 動画の長さに合わせる
    )
    vec_env.reset(dev)

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
        "step": [], "loss/total": [], "loss/progress": [],
        "loss/qp": [], "loss/effort": [],
    }

    print("=" * 60)
    print(f"  Hierarchical Velocity-Command Swarm Training")
    print(f"  swarms={num_agents}  batch={batch_size}  horizon={horizon}"
          f"  area={area_size}")
    print(f"  pool_size={N_pool}  mini_batch={mini_batch_size}  epochs={n_epochs}")
    print(f"  State=4D  Action=3D(vel_cmd)  Edge=4D  Nodes/sample={N_per}")
    print(f"  R_form={R_form}  s_min={s_min}  s_max={s_max}")
    print(f"  K_pos={K_pos}  K_v={K_v}  K_s={K_s}")
    print(f"  coef_progress={coef_goal}  coef_qp={coef_qp}  coef_effort={coef_effort}  w_scale={w_scale}")
    print(f"  coef_arrival={coef_arrival}  arrival_radius={arrival_radius}m")
    print(f"  use_payload={use_payload}")
    print("=" * 60)
    t_start = time.time()

    # Initialize environment once at the beginning
    info = {
        "life/avg": 0.0, "life/min": 0.0, "life/max": 0.0, "life/reset_rate": 0.0
    }

    for step in range(1, num_steps + 1):

        # ============================================================
        # PHASE 1: Vectorized data collection (no_grad)
        # ============================================================
        # (Reset is now handled batch-wise below)

        pool_agent = []
        pool_scale = []
        pool_payload = []
        pool_goal = []
        pool_obs = []
        pool_obs_hits = []  # LiDAR hit points pool
        reset_count = 0

        with torch.no_grad():
            for t in range(horizon):
                pool_agent.append(vec_env._agent_states.clone())
                pool_scale.append(vec_env._scale_states.clone())
                pool_payload.append(vec_env._payload_states.clone())
                pool_goal.append(vec_env._goal_states.clone())
                pool_obs.append(vec_env._obstacle_states.clone())
                # Collect LiDAR hits (B, n, nb, 2) - position only
                nb = 32
                lidar_hits_t = vec_env.get_lidar_hits(num_beams=nb)[..., :2]  # (B, n, nb, 2)
                pool_obs_hits.append(lidar_hits_t.clone())

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
                    pi_scaled, vec_env._agent_states, vec_env._goal_states,
                    vec_env._scale_states, K_pos, K_v, K_s, v_max,
                    s_max=s_max, u_max=u_max, mass=mass, use_payload=use_payload,
                )

                # Level 3: QP solve
                BN = batch_size * num_agents
                u_nom_flat = u_nom.reshape(BN, 3)
                pos_flat = vec_env._agent_states[:, :, :2].reshape(BN, 2)
                vel_flat = vec_env._agent_states[:, :, 2:4].reshape(BN, 2)
                s_flat = vec_env._scale_states[:, :, 0].reshape(BN)
                s_dot_flat = vec_env._scale_states[:, :, 1].reshape(BN)
                ps_flat = vec_env._payload_states.reshape(BN, 4)

                # Use LiDAR hits for QP instead of centers (already collected)
                obs_hits_flat = lidar_hits_t.reshape(BN, nb, 2)
                
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
                    payload_states=ps_flat if use_payload else None,
                    cable_length=cable_length, gravity=gravity,
                    gamma_min=gamma_min, gamma_max_full=gamma_max_full,
                    payload_damping=payload_damping,
                    u_max=u_max,
                )
                u_qp = u_qp_flat.reshape(batch_size, num_agents, 3)

                # Level 4: env.step distributes to drones
                vec_env.step(u_qp)

                # ── Individual Auto-Reset ──
                # Check for goal, collision, or timeout per batch
                done_masks = vec_env.get_done_masks()  # (B,) boolean
                reset_indices = torch.where(done_masks)[0]
                if len(reset_indices) > 0:
                    vec_env.reset_at_indices(reset_indices)
                    reset_count += len(reset_indices)
        
        # Tracking lifecycle for logging (moved outside 't' loop for reliability)
        with torch.no_grad():
            cur_steps = vec_env._step_counts.float()
            info["life/avg"] = cur_steps.mean().item()
            info["life/max"] = cur_steps.max().item()
            info["life/min"] = cur_steps.min().item()
            info["life/reset_rate"] = (reset_count / (batch_size * horizon))

        # ============================================================
        # PHASE 2: Flatten pool → (N_pool, n, ...) and shuffle
        # ============================================================
        all_agent = torch.stack(pool_agent).reshape(N_pool, num_agents, 4)
        all_scale = torch.stack(pool_scale).reshape(N_pool, num_agents, 2)
        all_payload = torch.stack(pool_payload).reshape(N_pool, num_agents, 4)
        all_goal = torch.stack(pool_goal).reshape(N_pool, num_agents, 4)
        all_obs_st = torch.stack(pool_obs).reshape(N_pool, -1, 4)

        all_obs_hits = torch.stack(pool_obs_hits).reshape(N_pool, num_agents, -1, 2)  # (N_pool, n, nb, 2)

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
                mb_obs_hits = all_obs_hits[idx]  # (mb, n, nb, 2)

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
                    s_max=s_max, u_max=u_max, mass=mass, use_payload=use_payload,
                )
                u_nom_flat = u_nom.reshape(mb_size * num_agents, 3)

                # ── Level 3: QP solve (no grad) ──
                with torch.no_grad():
                    states_flat = mb_agent.reshape(-1, 4)
                    goals_flat = mb_goal.reshape(-1, 4)
                    s_f = mb_scale.reshape(-1, 2)[:, 0]
                    sd_f = mb_scale.reshape(-1, 2)[:, 1]
                    ps_f = mb_payload.reshape(-1, 4)

                    # LiDAR hits for this mini-batch
                    # mb_obs_hits shape: (mb_size, n, nb, 2) — already per-agent
                    if mb_obs_hits is not None:
                        obs_hits_mb = mb_obs_hits.reshape(mb_size * num_agents, -1, 2)
                    else:
                        obs_hits_mb = None

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
                        obs_hits=obs_hits_mb,
                        agent_pos=states_flat[:, :2], agent_vel=states_flat[:, 2:4],
                        s=s_f, s_dot=sd_f,
                        other_agent_pos=mb_other_pos_flat,
                        other_agent_vel=mb_other_vel_flat,
                        other_agent_s=mb_other_s_flat,
                        other_agent_s_dot=mb_other_sd_flat,
                        R_form=R_form, r_margin=r_margin, mass=mass,
                        s_min=s_min, s_max=s_max,
                        payload_states=ps_f if use_payload else None,
                        cable_length=cable_length, gravity=gravity,
                        gamma_min=gamma_min, gamma_max_full=gamma_max_full,
                        payload_damping=payload_damping,
                        u_max=u_max,
                    )

                # ── Loss ──
                agent_pos_mb = mb_agent[:, :, :2].detach()   # (mb, n, 2)
                goal_pos_mb = mb_goal[:, :, :2].detach()     # (mb, n, 2)

                # v_target: has gradient through pi_scaled → GNN
                dist_mb = torch.norm(goal_pos_mb - agent_pos_mb, dim=-1, keepdim=True)
                unit_vec_mb = (goal_pos_mb - agent_pos_mb) / (dist_mb + 1e-6)
                v_ref_mb = unit_vec_mb * torch.clamp(K_pos * dist_mb, max=v_max)
                v_ref_mb = torch.clamp(v_ref_mb, -v_max, v_max)
                v_target_mb = v_ref_mb + pi_scaled[:, :, :2]  # (mb, n, 2) — has grad

                # One-step lookahead position (gradient flows through v_target → GNN)
                dt = vec_env.dt
                pos_next_mb = agent_pos_mb + v_target_mb * dt  # (mb, n, 2)

                # Distance reduction: positive when approaching goal
                dist_now = dist_mb.squeeze(-1)                                    # (mb, n) detached
                dist_next = torch.norm(goal_pos_mb - pos_next_mb, dim=-1)        # (mb, n) has grad
                dist_reduction_flat = (dist_now - dist_next).reshape(-1)         # (mb*n,)
                dist_to_goal_flat = dist_now.reshape(-1).detach()                # (mb*n,) detached

                loss, batch_info = compute_affine_loss(
                    pi_action=pi_agents.reshape(-1, 3),
                    u_nom=u_nom_flat,
                    u_qp=u_qp_flat,
                    dist_reduction=dist_reduction_flat,
                    dist_to_goal=dist_to_goal_flat,
                    coef_progress=coef_goal,
                    coef_qp=coef_qp,
                    coef_effort=coef_effort,
                    w_scale=w_scale,
                    coef_arrival=coef_arrival,
                    arrival_radius=arrival_radius,
                )
                # QP intervention tracking
                qp_intervention = (u_nom_flat - u_qp_flat).pow(2).sum(dim=-1)
                batch_info["qp_cut/mean"] = qp_intervention.mean().item()
                batch_info["qp_cut/max"] = qp_intervention.max().item()

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                optim.step()

                epoch_losses.append(batch_info)

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
                if use_payload:
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
            if use_payload:
                payload_str = f" | G: {mean_gamma:.2f}/{mean_gamma_limit:.2f} (max:{max_gamma:.2f}, p95:{p95_gamma:.2f}, v:{viol_rate:.2%})"
            else:
                payload_str = " | G: (no payload)"
            arrival_str = f"{avg_info.get('loss/arrival', 0.0):.4f}"
            print(f"Step {step:5d} | "
                  f"L: {avg_info['loss/total']:.4f} (qp:{avg_info['loss/qp']:.4f}, pr:{avg_info['loss/progress']:.4f}, ar:{arrival_str}, ef:{avg_info.get('loss/effort',0):.4f}) | "
                  f"S: {mean_s:.2f} ({min_s:.2f}-{max_s:.2f})"
                  f"{payload_str}")
            print(f"      Life: {info.get('life/avg', 0.0):.1f} ({info.get('life/min', 0.0):.0f}-{info.get('life/max', 0.0):.0f}) | "
                  f"Reset: {info.get('life/reset_rate', 0.0):.1%} | {elapsed:.0f}s")
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
            "use_payload": use_payload,
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
    parser.add_argument("--coef_effort", type=float, default=0.3)
    parser.add_argument("--w_scale", type=float, default=2.0)
    parser.add_argument("--coef_arrival", type=float, default=5.0,
                        help="Arrival bonus coefficient (default 5.0)")
    parser.add_argument("--arrival_radius", type=float, default=0.3,
                        help="Goal-reached threshold in metres (default 0.3)")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="affine_swarm_checkpoint.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_payload", action="store_true", default=False,
                        help="Disable payload dynamics and HOCBF constraint")
    args = parser.parse_args()
    a = vars(args)
    a["checkpoint_path"] = a.pop("checkpoint")
    a["use_payload"] = not a.pop("no_payload")
    train(**a)


if __name__ == "__main__":
    main()
