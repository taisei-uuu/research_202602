#!/usr/bin/env python3
"""
Visualization: Animate multi-agent trajectories as an MP4 video.

Supports:
  1. DoubleIntegrator (single agent dots)
  2. SwarmIntegrator  (3-drone triangle formations + bounding circles)

Usage (Colab):
    # LQR mode (single agent)
    !python visualize.py --num_agents 4 --seed 0

    # Swarm LQR (no training)
    !python visualize.py --swarm_lqr --num_agents 3 --seed 0

    # Trained swarm policy
    !python visualize.py --checkpoint gcbf_swarm_checkpoint.pt --seed 0
    
    # Trained swarm policy with heatmap
    !python visualize.py --checkpoint gcbf_swarm_checkpoint.pt --seed 0 --heatmap
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import numpy as np
import torch

from gcbf_plus.env import DoubleIntegrator, SwarmIntegrator
from gcbf_plus.nn import GCBFNetwork, PolicyNetwork
from gcbf_plus.utils.swarm_graph import build_vectorized_swarm_graph
from gcbf_plus.train_swarm import extract_agent_outputs


AGENT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000",
]


def compute_triangle_vertices(com_x, com_y, theta, R_form):
    """Compute 3 vertices of equilateral triangle at (com_x, com_y) with orientation theta."""
    s32 = math.sqrt(3.0) / 2.0
    local = np.array([
        [R_form, 0.0],
        [-R_form / 2.0,  R_form * s32],
        [-R_form / 2.0, -R_form * s32],
    ])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = local @ R.T + np.array([com_x, com_y])
    return verts


def load_trained_policy(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    policy_net = PolicyNetwork(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        action_dim=cfg["action_dim"],
        n_agents=cfg["num_agents"],
    )
    policy_net.load_state_dict(ckpt["policy_net"])
    policy_net.eval()
    
    gcbf_net = GCBFNetwork(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        n_agents=cfg["num_agents"],
    )
    gcbf_net.load_state_dict(ckpt["gcbf_net"])
    gcbf_net.eval()
    
    print(f"  Loaded trained policy and GCBF from: {checkpoint_path}")
    print(f"    agents={cfg['num_agents']}  area={cfg['area_size']}  "
          f"n_obs={cfg['n_obs']}  dt={cfg['dt']}  comm_radius={cfg['comm_radius']}")
    if "r_swarm" in cfg:
        print(f"    r_swarm={cfg['r_swarm']}  R_form={cfg.get('R_form', 'N/A')}")
    return policy_net, gcbf_net, cfg


def run_simulation(
    num_agents: int = 4,
    area_size: float = 2.0,
    max_steps: int = 512,
    dt: float = 0.03,
    n_obs: int = 2,
    seed: int = 0,
    checkpoint_path: Optional[str] = None,
    force_lqr: bool = False,
    swarm_lqr: bool = False,
):
    policy_net = None
    gcbf_net = None
    mode = "lqr"
    is_swarm = swarm_lqr
    R_form = 0.3
    r_swarm = 0.4
    comm_radius = 2.0 if swarm_lqr else 1.5
    if swarm_lqr:
        area_size = max(area_size, 4.0)  # swarm needs larger area for spacing

    if checkpoint_path is not None:
        policy_net, gcbf_net, cfg = load_trained_policy(checkpoint_path)
        num_agents = cfg["num_agents"]
        area_size = cfg["area_size"]
        n_obs = cfg["n_obs"]
        dt = cfg["dt"]
        comm_radius = cfg["comm_radius"]
        mode = "trained_policy"
        is_swarm = "r_swarm" in cfg
        R_form = cfg.get("R_form", 0.3)
        r_swarm = cfg.get("r_swarm", 0.4)
        if force_lqr:
            print("  [force_lqr] Ignoring trained policy — using LQR only")
            policy_net = None
            gcbf_net = None
            mode = "lqr"

    if is_swarm:
        env = SwarmIntegrator(
            num_agents=num_agents, area_size=area_size, dt=dt, max_steps=max_steps,
            params={"n_obs": n_obs, "comm_radius": comm_radius, "R_form": R_form},
        )
    else:
        env = DoubleIntegrator(
            num_agents=num_agents, area_size=area_size, dt=dt, max_steps=max_steps,
            params={"n_obs": n_obs, "comm_radius": comm_radius},
        )

    env.reset(seed=seed)

    trajectories: List[np.ndarray] = []
    # Save FULL 4D state into trajectories for heatmap evaluation
    trajectories.append(env.agent_states.detach().numpy().copy())

    goals = env.goal_states.detach().numpy().copy()  # Full 4D
    obstacle_info = [(obs.center.numpy().copy(), obs.half_size.numpy().copy())
                     for obs in env._obstacles]

    for _ in range(max_steps):
        if policy_net is not None:
            with torch.no_grad():
                graph = env._get_graph()
                u_ref = env.nominal_controller()
                pi_raw = policy_net(graph)
                u = 2.0 * pi_raw + u_ref
        else:
            u = env.nominal_controller()

        _, info = env.step(u)
        trajectories.append(env.agent_states.detach().numpy().copy())
        if info["done"]:
            break

    trajectories = np.array(trajectories)
    displacement = np.linalg.norm(trajectories[-1, :, :2] - trajectories[0, :, :2], axis=1)
    print(f"  Agent displacements: {displacement}")

    return trajectories, goals, obstacle_info, area_size, comm_radius, mode, is_swarm, R_form, r_swarm, gcbf_net


# ═══════════════════════════════════════════════════════════════════════
# Video creation
# ═══════════════════════════════════════════════════════════════════════

def create_video(
    trajectories, goals, obstacle_info, area_size,
    save_path="trajectories.mp4", fps=30, skip=1,
    mode="lqr", comm_radius=1.5,
    is_swarm=False, R_form=0.3, r_swarm=0.4,
    gcbf_net=None, heatmap=False, grid_res=40,
):
    n_agents = trajectories.shape[1]
    total_frames = trajectories.shape[0]
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)]

    frame_indices = list(range(0, total_frames, skip))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)
    n_frames = len(frame_indices)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(-0.3, area_size + 0.3)
    ax.set_ylim(-0.3, area_size + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#adb5bd")

    # Optional Heatmap setup
    X_grid, Y_grid = None, None
    grid_pts = None
    N_grid = 0
    contour_col = [None, None]  # [contourf, contourlines]
    dev = next(gcbf_net.parameters()).device if gcbf_net is not None else torch.device("cpu")
    
    if heatmap and gcbf_net is not None and is_swarm:
        grid_x = np.linspace(-0.3, area_size + 0.3, grid_res)
        grid_y = np.linspace(-0.3, area_size + 0.3, grid_res)
        X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
        grid_pts = np.c_[X_grid.ravel(), Y_grid.ravel()]
        N_grid = len(grid_pts)

    # Obstacles
    for center, half_size in obstacle_info:
        rect = mpatches.FancyBboxPatch(
            (center[0] - half_size[0], center[1] - half_size[1]),
            2 * half_size[0], 2 * half_size[1],
            boxstyle="round,pad=0.01",
            facecolor="#adb5bd", edgecolor="#6c757d",
            linewidth=1.2, alpha=0.65, zorder=5,
        )
        ax.add_patch(rect)

    # Start markers
    starts = trajectories[0]
    for i in range(n_agents):
        if is_swarm:
            verts = compute_triangle_vertices(starts[i, 0], starts[i, 1], 0.0, R_form)
            tri = Polygon(verts, closed=True, facecolor=colors[i],
                          edgecolor="white", linewidth=1.5, alpha=0.35, zorder=6)
            ax.add_patch(tri)
        ax.plot(starts[i, 0], starts[i, 1], marker="o", markersize=10 if is_swarm else 12,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=2, zorder=7)

    # Goal markers
    for i in range(n_agents):
        ax.plot(goals[i, 0], goals[i, 1], marker="*", markersize=18,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=1.2, zorder=7)

    # ── Dynamic elements ─────────────────────────────────────────────
    trail_lines = []
    for i in range(n_agents):
        (line,) = ax.plot([], [], color=colors[i], linewidth=1.8, alpha=0.75, zorder=4)
        trail_lines.append(line)

    # Current position: triangle + bounding circle (swarm) or dot (single)
    swarm_triangles = []
    bounding_circles = []
    current_dots = []

    if is_swarm:
        for i in range(n_agents):
            # Triangle formation
            tri = Polygon(np.zeros((3, 2)), closed=True,
                          facecolor=colors[i], edgecolor="white",
                          linewidth=1.5, alpha=0.6, zorder=8)
            ax.add_patch(tri)
            swarm_triangles.append(tri)
            # Drone dots at vertices
            for k in range(3):
                (d,) = ax.plot([], [], marker="o", markersize=5,
                               markerfacecolor="white",
                               markeredgecolor=colors[i],
                               markeredgewidth=1.2, zorder=9)
                current_dots.append(d)
            # Bounding circle (r_swarm) — the safety zone
            bc = plt.Circle((0, 0), r_swarm, fill=True,
                            facecolor=colors[i], edgecolor=colors[i],
                            linewidth=1.5, alpha=0.12, linestyle="-", zorder=6)
            ax.add_patch(bc)
            bounding_circles.append(bc)
    else:
        for i in range(n_agents):
            (dot,) = ax.plot([], [], marker="o", markersize=8,
                             markerfacecolor=colors[i],
                             markeredgecolor="white", markeredgewidth=1.5,
                             zorder=8)
            current_dots.append(dot)

    # Sensing radius (dashed)
    sensing_circles = []
    for i in range(n_agents):
        sc = plt.Circle((0, 0), comm_radius, fill=False, linestyle="--",
                         linewidth=1.0, edgecolor=colors[i], alpha=0.25, zorder=3)
        ax.add_patch(sc)
        sensing_circles.append(sc)

    entity_name = "swarms" if is_swarm else "agents"
    mode_label = "Trained Policy π(x)" if mode == "trained_policy" else "LQR Controller"
    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=11, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=15,
    )

    title_add = " w/ Heatmap (Agent 0)" if heatmap else ""
    ax.set_title(
        f"{'Swarm' if is_swarm else 'Agent'} Trajectories" + title_add + "\n"
        f"({n_agents} {entity_name} · {mode_label})",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    # Legend
    from matplotlib.lines import Line2D
    handles = []
    for i in range(n_agents):
        handles.append(Line2D([0], [0], color=colors[i], linewidth=2,
                              label=f"Swarm {i}" if is_swarm else f"Agent {i}"))
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                          markeredgecolor="white", markersize=14, label="Goal (★)"))
    handles.append(mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                                  alpha=0.65, label="Obstacle"))
    if is_swarm:
        handles.append(mpatches.Patch(facecolor="gray", edgecolor="white",
                                      alpha=0.5, label=f"Formation △ (R={R_form})"))
        handles.append(mpatches.Patch(facecolor="gray", edgecolor="gray",
                                      alpha=0.2, label=f"Safety ○ (r={r_swarm})"))
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#dee2e6")

    def init():
        for line in trail_lines:
            line.set_data([], [])
        for dot in current_dots:
            dot.set_data([], [])
        return []

    def update(frame_idx):
        sim_step = frame_indices[frame_idx]
        
        # ── Rendering Heatmap ──
        if heatmap and gcbf_net is not None and is_swarm:
            if contour_col[0] is not None:
                if hasattr(contour_col[0], 'collections'):
                    for coll in contour_col[0].collections:
                        coll.remove()
                else:
                    contour_col[0].remove()
                contour_col[0] = None
                
            if contour_col[1] is not None:
                if hasattr(contour_col[1], 'collections'):
                    for coll in contour_col[1].collections:
                        coll.remove()
                else:
                    contour_col[1].remove()
                contour_col[1] = None
                
            with torch.no_grad():
                states_t = trajectories[sim_step]  # (n_agents, 4)
                
                # Batched evaluation
                # Create a batch of size N_grid where each sample is the entire n_agents state
                states_batch = torch.tensor(states_t, dtype=torch.float32, device=dev).unsqueeze(0).expand(N_grid, n_agents, 4).clone()
                
                grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=dev)
                
                # For each grid point, we ONLY change the position of the focal agent (Agent 0)
                # The other agents remain exactly where they actually are in this timestep
                states_batch[:, 0, :2] = grid_tensor
                
                goals_batch = torch.tensor(goals, dtype=torch.float32, device=dev).unsqueeze(0).expand(N_grid, n_agents, 4)
                
                if len(obstacle_info) > 0:
                    obs_list = []
                    for (c, hs) in obstacle_info:
                        obs_list.append(np.concatenate([c, hs]))
                    obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=dev).unsqueeze(0).expand(N_grid, len(obstacle_info), 4)
                else:
                    obs_t = None
                
                graph = build_vectorized_swarm_graph(states_batch, goals_batch, obs_t, comm_radius, 3, 4)
                
                # GCBFNetwork.forward() normally slices out `[:self.n_agents]`. 
                # Since we passed a mega-graph with (N_grid * N_per) nodes, and we want 
                # the outputs for all `N_grid * n_agents` agent nodes, we temporarily override it.
                original_n_agents = gcbf_net.n_agents
                gcbf_net.n_agents = N_grid * n_agents
                h_out_agents = gcbf_net(graph).squeeze(-1) # (N_grid * n_agents)
                gcbf_net.n_agents = original_n_agents
                
                h_out_agents = h_out_agents.reshape(N_grid, n_agents)
                h_agent0 = h_out_agents[:, 0].cpu().numpy().reshape(grid_res, grid_res)
                
                contour = ax.contourf(X_grid, Y_grid, h_agent0, levels=np.linspace(-0.5, 0.5, 21), cmap="RdBu", extend="both", zorder=0, alpha=0.6)
                contour_line = ax.contour(X_grid, Y_grid, h_agent0, levels=[0.0], colors='black', linewidths=1.5, zorder=0)
                
                contour_col[0] = contour
                contour_col[1] = contour_line

        # ── Update Agents ──
        for i in range(n_agents):
            trail = trajectories[:sim_step + 1, i, :]
            trail_lines[i].set_data(trail[:, 0], trail[:, 1])
            cx = trajectories[sim_step, i, 0]
            cy = trajectories[sim_step, i, 1]
            sensing_circles[i].center = (cx, cy)

            if is_swarm:
                verts = compute_triangle_vertices(cx, cy, 0.0, R_form)
                swarm_triangles[i].set_xy(verts)
                for k in range(3):
                    current_dots[i * 3 + k].set_data([verts[k, 0]], [verts[k, 1]])
                bounding_circles[i].center = (cx, cy)
            else:
                current_dots[i].set_data([cx], [cy])

        step_text.set_text(f"Step {sim_step}/{total_frames - 1}  [{mode_label}]")
        return []

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_frames,
        interval=1000 // fps, blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Video saved to: {save_path}  ({n_frames} frames @ {fps} fps)")
    return save_path


# ═══════════════════════════════════════════════════════════════════════
# Static plot
# ═══════════════════════════════════════════════════════════════════════

def plot_trajectories(
    trajectories, goals, obstacle_info, area_size,
    save_path="trajectories.png", mode="lqr", comm_radius=1.5,
    is_swarm=False, R_form=0.3, r_swarm=0.4, gcbf_net=None, heatmap=False, grid_res=40
):
    n_agents = trajectories.shape[1]
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)]
    entity_name = "swarms" if is_swarm else "agents"
    mode_label = "Trained Policy π(x)" if mode == "trained_policy" else "LQR Controller"

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(-0.3, area_size + 0.3)
    ax.set_ylim(-0.3, area_size + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#adb5bd")

    # Static Heatmap (uses final position)
    if heatmap and gcbf_net is not None and is_swarm:
        dev = next(gcbf_net.parameters()).device
        grid_x = np.linspace(-0.3, area_size + 0.3, grid_res)
        grid_y = np.linspace(-0.3, area_size + 0.3, grid_res)
        X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
        grid_pts = np.c_[X_grid.ravel(), Y_grid.ravel()]
        N_grid = len(grid_pts)
        
            # Batched evaluation
            states_t = trajectories[-1]  # Final state
            states_batch = torch.tensor(states_t, dtype=torch.float32, device=dev).unsqueeze(0).expand(N_grid, n_agents, 4).clone()
            
            grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=dev)
            
            # For each grid point, change only the focal agent (Agent 0) position
            states_batch[:, 0, :2] = grid_tensor
            
            goals_batch = torch.tensor(goals, dtype=torch.float32, device=dev).unsqueeze(0).expand(N_grid, n_agents, 4)
            if len(obstacle_info) > 0:
                obs_list = []
                for (c, hs) in obstacle_info:
                    obs_list.append(np.concatenate([c, hs]))
                obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=dev).unsqueeze(0).expand(N_grid, len(obstacle_info), 4)
            else:
                obs_t = None
            
            graph = build_vectorized_swarm_graph(states_batch, goals_batch, obs_t, comm_radius, 3, 4)
            # Temporarily override gcbf_net.n_agents to get all batched agent outputs
            original_n_agents = gcbf_net.n_agents
            gcbf_net.n_agents = N_grid * n_agents
            h_out_agents = gcbf_net(graph).squeeze(-1)
            gcbf_net.n_agents = original_n_agents
            
            h_out_agents = h_out_agents.reshape(N_grid, n_agents)
            h_agent0 = h_out_agents[:, 0].cpu().numpy().reshape(grid_res, grid_res)
            
            ax.contourf(X_grid, Y_grid, h_agent0, levels=np.linspace(-0.5, 0.5, 21), cmap="RdBu", extend="both", zorder=0, alpha=0.6)
            ax.contour(X_grid, Y_grid, h_agent0, levels=[0.0], colors='black', linewidths=1.5, zorder=0)

    # Obstacles
    for center, half_size in obstacle_info:
        rect = mpatches.FancyBboxPatch(
            (center[0] - half_size[0], center[1] - half_size[1]),
            2 * half_size[0], 2 * half_size[1],
            boxstyle="round,pad=0.01",
            facecolor="#adb5bd", edgecolor="#6c757d",
            linewidth=1.2, alpha=0.65, zorder=1,
        )
        ax.add_patch(rect)

    # Trajectories
    for i in range(n_agents):
        path = trajectories[:, i, :]
        ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=1.8, alpha=0.75,
                zorder=2, label=f"Swarm {i}" if is_swarm else f"Agent {i}")

    # Start positions
    starts = trajectories[0]
    for i in range(n_agents):
        if is_swarm:
            verts = compute_triangle_vertices(starts[i, 0], starts[i, 1], 0.0, R_form)
            tri = Polygon(verts, closed=True, facecolor=colors[i],
                          edgecolor="white", linewidth=1.5, alpha=0.35, zorder=4)
            ax.add_patch(tri)
        ax.plot(starts[i, 0], starts[i, 1], marker="o", markersize=10,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=2, zorder=5)

    # Goals
    for i in range(n_agents):
        ax.plot(goals[i, 0], goals[i, 1], marker="*", markersize=18,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)

    # Final positions
    finals = trajectories[-1]
    for i in range(n_agents):
        if is_swarm:
            verts = compute_triangle_vertices(finals[i, 0], finals[i, 1], 0.0, R_form)
            tri = Polygon(verts, closed=True, facecolor=colors[i],
                          edgecolor="white", linewidth=1.5, alpha=0.7, zorder=6)
            ax.add_patch(tri)
            for k in range(3):
                ax.plot(verts[k, 0], verts[k, 1], marker="o", markersize=5,
                        markerfacecolor="white", markeredgecolor=colors[i],
                        markeredgewidth=1.2, zorder=7)
            bc = plt.Circle((finals[i, 0], finals[i, 1]), r_swarm,
                            fill=True, facecolor=colors[i], edgecolor=colors[i],
                            linewidth=1.5, alpha=0.12, zorder=3)
            ax.add_patch(bc)

    # Ghost formations along trajectory
    if is_swarm:
        n_ghosts = min(8, trajectories.shape[0] // 10)
        if n_ghosts > 0:
            ghost_indices = np.linspace(0, trajectories.shape[0] - 1,
                                        n_ghosts, dtype=int)[1:-1]
            for idx in ghost_indices:
                for i in range(n_agents):
                    verts = compute_triangle_vertices(
                        trajectories[idx, i, 0], trajectories[idx, i, 1], 0.0, R_form
                    )
                    tri = Polygon(verts, closed=True, facecolor=colors[i],
                                  edgecolor=colors[i], linewidth=0.5,
                                  alpha=0.12, zorder=3)
                    ax.add_patch(tri)

    # Sensing radius at final
    for i in range(n_agents):
        sc = plt.Circle((finals[i, 0], finals[i, 1]), comm_radius,
                        fill=False, linestyle="--", linewidth=1.0,
                        edgecolor=colors[i], alpha=0.25, zorder=1)
        ax.add_patch(sc)

    title_add = " w/ Heatmap (Agent 0)" if heatmap else ""
    ax.set_title(
        f"{'Swarm' if is_swarm else 'Agent'} Trajectories" + title_add + "\n"
        f"({n_agents} {entity_name}, {trajectories.shape[0]-1} steps · {mode_label})",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    from matplotlib.lines import Line2D
    handles = []
    for i in range(n_agents):
        handles.append(Line2D([0], [0], color=colors[i], linewidth=2,
                              label=f"Swarm {i}" if is_swarm else f"Agent {i}"))
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                          markeredgecolor="white", markersize=14, label="Goal (★)"))
    handles.append(mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                                  alpha=0.65, label="Obstacle"))
    if is_swarm:
        handles.append(mpatches.Patch(facecolor="gray", edgecolor="white",
                                      alpha=0.5, label=f"Formation △ (R={R_form})"))
        handles.append(mpatches.Patch(facecolor="gray", edgecolor="gray",
                                      alpha=0.2, label=f"Safety ○ (r={r_swarm})"))
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#dee2e6")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-agent trajectories (LQR or trained policy)"
    )
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--area_size", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--n_obs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="trajectories.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--swarm_lqr", action="store_true",
                        help="Run SwarmIntegrator with LQR only")
    parser.add_argument("--force_lqr", action="store_true",
                        help="Ignore trained policy, use LQR only")
    parser.add_argument("--heatmap", action="store_true",
                        help="Visualize GCBF values as a heatmap overlay")
    parser.add_argument("--grid_res", type=int, default=40,
                        help="Resolution of the heatmap grid (default: 40)")
    
    args = parser.parse_args()

    print("Running simulation...")
    result = run_simulation(
        num_agents=args.num_agents, area_size=args.area_size,
        max_steps=args.max_steps, dt=args.dt, n_obs=args.n_obs,
        seed=args.seed, checkpoint_path=args.checkpoint,
        force_lqr=args.force_lqr, swarm_lqr=args.swarm_lqr,
    )
    trajectories, goals, obstacle_info, area, comm_r, mode, is_swarm, R_form, r_swarm, gcbf_net = result
    entity = "swarms" if is_swarm else "agents"
    print(f"  Recorded {trajectories.shape[0]} frames for "
          f"{trajectories.shape[1]} {entity}.  Mode: {mode}")

    common = dict(
        trajectories=trajectories, goals=goals, obstacle_info=obstacle_info,
        area_size=area, mode=mode, comm_radius=comm_r,
        is_swarm=is_swarm, R_form=R_form, r_swarm=r_swarm,
        gcbf_net=gcbf_net, heatmap=args.heatmap, grid_res=args.grid_res
    )

    if args.save.endswith(".mp4"):
        print("Creating video...")
        create_video(**common, save_path=args.save, fps=args.fps, skip=args.skip)
    else:
        print("Plotting static image...")
        plot_trajectories(**common, save_path=args.save)

    if args.png:
        png_path = args.save.replace(".mp4", ".png") if args.save.endswith(".mp4") \
            else "trajectories.png"
        plot_trajectories(**common, save_path=png_path)

    print("Done!")


if __name__ == "__main__":
    main()
