#!/usr/bin/env python3
"""
Visualization: Animate multi-agent trajectories as an MP4 video.

Supports:
  1. DoubleIntegrator (single agent dots)
  2. SwarmIntegrator  (3-drone triangle formations per swarm)

Draws:
  - Start positions  (● circles / △ triangles)
  - Goal positions   (★ stars)
  - Trajectories     (colored growing paths of the CoM)
  - Obstacles        (gray rectangles)
  - Current swarm positions (filled triangles that move + rotate each frame)

Usage (Colab):
    # ── LQR mode (no training needed) ──
    !python visualize.py --num_agents 4 --seed 0

    # ── Trained policy mode (single agent) ──
    !python visualize.py --checkpoint gcbf_plus_checkpoint.pt --seed 0

    # ── Trained policy mode (swarm) ──
    !python visualize.py --checkpoint gcbf_swarm_checkpoint.pt --seed 0

    # Display in notebook:
    from IPython.display import Video
    Video("trajectories.mp4", embed=True)
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import numpy as np
import torch

from gcbf_plus.env import DoubleIntegrator, SwarmIntegrator
from gcbf_plus.nn import GCBFNetwork, PolicyNetwork


# ── Color palette ────────────────────────────────────────────────────────
AGENT_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
    "#dcbeff",  # lavender
    "#9A6324",  # brown
    "#fffac8",  # beige
    "#800000",  # maroon
    "#aaffc3",  # mint
    "#808000",  # olive
]


# ── Swarm geometry helper ────────────────────────────────────────────────

def compute_triangle_vertices(com_x, com_y, theta, R_form):
    """
    Compute the 3 vertices of the equilateral triangle formation.

    Parameters
    ----------
    com_x, com_y : float — CoM position
    theta : float — yaw angle (radians)
    R_form : float — circumradius

    Returns
    -------
    verts : (3, 2) numpy array
    """
    s32 = math.sqrt(3.0) / 2.0
    # Local offsets (same as in swarm_graph.py)
    local = np.array([
        [R_form, 0.0],
        [-R_form / 2.0,  R_form * s32],
        [-R_form / 2.0, -R_form * s32],
    ])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    rotated = local @ R.T  # (3, 2)
    verts = rotated + np.array([com_x, com_y])
    return verts


def load_trained_policy(checkpoint_path: str):
    """
    Load a trained PolicyNetwork from a checkpoint saved by train.py.

    Returns
    -------
    policy_net : PolicyNetwork  (eval mode, no grad)
    config     : dict           (num_agents, area_size, etc.)
    """
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

    print(f"  Loaded trained policy from: {checkpoint_path}")
    print(f"    agents={cfg['num_agents']}  area={cfg['area_size']}  "
          f"n_obs={cfg['n_obs']}  dt={cfg['dt']}  comm_radius={cfg['comm_radius']}")
    if 'state_dim' in cfg:
        print(f"    state_dim={cfg['state_dim']}  edge_dim={cfg['edge_dim']}"
              f"  R_form={cfg.get('R_form', 'N/A')}")
    return policy_net, cfg


def run_simulation(
    num_agents: int = 4,
    area_size: float = 2.0,
    max_steps: int = 256,
    dt: float = 0.03,
    n_obs: int = 2,
    seed: int = 0,
    checkpoint_path: Optional[str] = None,
    force_lqr: bool = False,
    swarm_lqr: bool = False,
):
    """
    Run a simulation and record trajectories.

    Returns
    -------
    trajectories : np.ndarray, shape (T+1, num_agents, 2) — CoM positions
    goals        : np.ndarray, shape (num_agents, 2)
    obstacle_info: list of (center, half_size)
    area_size    : float
    comm_radius  : float
    mode         : str — "trained_policy" or "lqr"
    is_swarm     : bool
    R_form       : float (only meaningful if is_swarm)
    thetas       : np.ndarray (T+1, num_agents) — yaw angles (only if swarm)
    """

    # ── Load trained policy if checkpoint given ──────────────────────
    policy_net = None
    mode = "lqr"
    is_swarm = swarm_lqr  # --swarm_lqr forces swarm mode even without checkpoint
    R_form = 0.3
    r_swarm = 0.4
    comm_radius = 2.0 if swarm_lqr else 1.5

    if checkpoint_path is not None:
        policy_net, cfg = load_trained_policy(checkpoint_path)
        num_agents = cfg["num_agents"]
        area_size = cfg["area_size"]
        n_obs = cfg["n_obs"]
        dt = cfg["dt"]
        comm_radius = cfg["comm_radius"]
        mode = "trained_policy"
        is_swarm = "r_swarm" in cfg  # detect swarm by r_swarm marker
        R_form = cfg.get("R_form", 0.3)
        r_swarm = cfg.get("r_swarm", 0.4)
        if force_lqr:
            print("  [force_lqr] Ignoring trained policy — using LQR only")
            policy_net = None
            mode = "lqr"

    # ── Create environment ───────────────────────────────────────────
    if is_swarm:
        env = SwarmIntegrator(
            num_agents=num_agents,
            area_size=area_size,
            dt=dt,
            max_steps=max_steps,
            params={"n_obs": n_obs, "comm_radius": comm_radius, "R_form": R_form},
        )
    else:
        env = DoubleIntegrator(
            num_agents=num_agents,
            area_size=area_size,
            dt=dt,
            max_steps=max_steps,
            params={"n_obs": n_obs, "comm_radius": comm_radius},
        )

    env.reset(seed=seed)

    # ── Record initial state ─────────────────────────────────────────
    trajectories: List[np.ndarray] = []
    thetas_list: List[np.ndarray] = []

    trajectories.append(env.agent_states[:, :2].detach().numpy().copy())
    if is_swarm:
        thetas_list.append(env.agent_states[:, 4].detach().numpy().copy())

    goals = env.goal_states[:, :2].detach().numpy().copy()

    obstacle_info = []
    for obs in env._obstacles:
        c = obs.center.numpy().copy()
        hs = obs.half_size.numpy().copy()
        obstacle_info.append((c, hs))

    # ── Simulate ─────────────────────────────────────────────────────
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
        trajectories.append(env.agent_states[:, :2].detach().numpy().copy())
        if is_swarm:
            thetas_list.append(env.agent_states[:, 4].detach().numpy().copy())
        if info["done"]:
            break

    trajectories = np.array(trajectories)  # (T+1, n, 2)
    thetas = np.array(thetas_list) if is_swarm else None  # (T+1, n) or None

    displacement = np.linalg.norm(trajectories[-1] - trajectories[0], axis=1)
    print(f"  Agent displacements: {displacement}")

    return trajectories, goals, obstacle_info, area_size, comm_radius, mode, is_swarm, R_form, thetas


def create_video(
    trajectories: np.ndarray,
    goals: np.ndarray,
    obstacle_info,
    area_size: float,
    save_path: str = "trajectories.mp4",
    fps: int = 30,
    skip: int = 1,
    mode: str = "lqr",
    comm_radius: float = 1.5,
    is_swarm: bool = False,
    R_form: float = 0.3,
    thetas: Optional[np.ndarray] = None,
):
    """
    Create an MP4 animation of agent trajectories.
    For swarm mode, draws 3-drone triangle formations instead of dots.
    """
    n_agents = trajectories.shape[1]
    total_frames = trajectories.shape[0]
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)]

    frame_indices = list(range(0, total_frames, skip))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)
    n_frames = len(frame_indices)

    # ── Set up the figure ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(-0.3, area_size + 0.3)
    ax.set_ylim(-0.3, area_size + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#adb5bd")

    # ── Static elements: obstacles ───────────────────────────────────
    for center, half_size in obstacle_info:
        rect = mpatches.FancyBboxPatch(
            (center[0] - half_size[0], center[1] - half_size[1]),
            2 * half_size[0],
            2 * half_size[1],
            boxstyle="round,pad=0.01",
            facecolor="#adb5bd",
            edgecolor="#6c757d",
            linewidth=1.2,
            alpha=0.65,
            zorder=1,
        )
        ax.add_patch(rect)

    # ── Static elements: start markers ───────────────────────────────
    starts = trajectories[0]
    if is_swarm and thetas is not None:
        for i in range(n_agents):
            verts = compute_triangle_vertices(
                starts[i, 0], starts[i, 1], thetas[0, i], R_form
            )
            tri = Polygon(verts, closed=True,
                          facecolor=colors[i], edgecolor="white",
                          linewidth=2, alpha=0.5, zorder=5)
            ax.add_patch(tri)
    else:
        for i in range(n_agents):
            ax.plot(
                starts[i, 0], starts[i, 1],
                marker="o", markersize=12,
                markerfacecolor=colors[i],
                markeredgecolor="white", markeredgewidth=2,
                zorder=5,
            )

    # ── Static elements: goal markers (stars) ────────────────────────
    for i in range(n_agents):
        ax.plot(
            goals[i, 0], goals[i, 1],
            marker="*", markersize=18,
            markerfacecolor=colors[i],
            markeredgecolor="white", markeredgewidth=1.2,
            zorder=5,
        )

    # ── Dynamic elements ─────────────────────────────────────────────
    # CoM trail lines
    trail_lines = []
    for i in range(n_agents):
        (line,) = ax.plot([], [], color=colors[i], linewidth=1.8, alpha=0.75, zorder=2)
        trail_lines.append(line)

    # Current position: dots (single agent) or triangles (swarm)
    if is_swarm and thetas is not None:
        # Create triangle patches for each swarm
        swarm_triangles = []
        swarm_drone_dots = []  # small dots for each drone
        for i in range(n_agents):
            tri = Polygon(
                np.zeros((3, 2)), closed=True,
                facecolor=colors[i], edgecolor="white",
                linewidth=1.5, alpha=0.7, zorder=6,
            )
            ax.add_patch(tri)
            swarm_triangles.append(tri)
            # 3 small dots for individual drones
            dots_i = []
            for k in range(3):
                (d,) = ax.plot([], [], marker="o", markersize=4,
                               markerfacecolor="white",
                               markeredgecolor=colors[i],
                               markeredgewidth=1.0, zorder=7)
                dots_i.append(d)
            swarm_drone_dots.append(dots_i)
        current_dots = []  # not used in swarm mode
    else:
        swarm_triangles = []
        swarm_drone_dots = []
        current_dots = []
        for i in range(n_agents):
            (dot,) = ax.plot(
                [], [],
                marker="o", markersize=8,
                markerfacecolor=colors[i],
                markeredgecolor="white", markeredgewidth=1.5,
                zorder=6,
            )
            current_dots.append(dot)

    # ── Sensing radius circles (move with agents) ────────────────────
    sensing_circles = []
    for i in range(n_agents):
        circle = plt.Circle(
            (0, 0), comm_radius,
            fill=False, linestyle="--", linewidth=1.0,
            edgecolor=colors[i], alpha=0.35, zorder=1,
        )
        ax.add_patch(circle)
        sensing_circles.append(circle)

    # Step counter + mode label
    entity_name = "swarms" if is_swarm else "agents"
    mode_label = "Trained Policy π(x)" if mode == "trained_policy" else "LQR Controller"
    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        zorder=10,
    )

    # Title
    ax.set_title(
        f"{'Swarm' if is_swarm else 'Agent'} Trajectories  "
        f"({n_agents} {entity_name} · {mode_label})",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    # ── Legend ────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    legend_handles = []
    for i in range(n_agents):
        legend_handles.append(
            Line2D([0], [0], color=colors[i], linewidth=2,
                   label=f"Swarm {i}" if is_swarm else f"Agent {i}")
        )
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=10, label="Start")
    )
    legend_handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=14, label="Goal (★)")
    )
    legend_handles.append(
        mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                       alpha=0.65, label="Obstacle")
    )
    if is_swarm:
        legend_handles.append(
            mpatches.Patch(facecolor="gray", edgecolor="white",
                           alpha=0.5, label="Formation △")
        )
    ax.legend(
        handles=legend_handles, loc="upper right",
        fontsize=9, framealpha=0.9, edgecolor="#dee2e6",
    )

    # ── Collect all dynamic artists for blit ──────────────────────────
    all_dynamic = trail_lines + current_dots + sensing_circles + [step_text]
    # Triangles and drone dots are updated via set_xy / set_data but
    # with blit=False they refresh automatically
    for dots_i in swarm_drone_dots:
        all_dynamic.extend(dots_i)

    def init():
        for line in trail_lines:
            line.set_data([], [])
        for dot in current_dots:
            dot.set_data([], [])
        for i, circ in enumerate(sensing_circles):
            circ.center = (starts[i, 0], starts[i, 1])
        step_text.set_text("")
        return all_dynamic

    def update(frame_idx):
        sim_step = frame_indices[frame_idx]
        for i in range(n_agents):
            trail = trajectories[: sim_step + 1, i, :]
            trail_lines[i].set_data(trail[:, 0], trail[:, 1])
            cx = trajectories[sim_step, i, 0]
            cy = trajectories[sim_step, i, 1]
            sensing_circles[i].center = (cx, cy)

            if is_swarm and thetas is not None:
                # Update triangle vertices
                verts = compute_triangle_vertices(
                    cx, cy, thetas[sim_step, i], R_form
                )
                swarm_triangles[i].set_xy(verts)
                # Update drone dots
                for k in range(3):
                    swarm_drone_dots[i][k].set_data([verts[k, 0]], [verts[k, 1]])
            else:
                current_dots[i].set_data([cx], [cy])

        step_text.set_text(f"Step {sim_step}/{total_frames - 1}  [{mode_label}]")
        return all_dynamic

    # ── Build animation ──────────────────────────────────────────────
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,  # blit=False needed for Polygon patch updates
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Video saved to: {save_path}  ({n_frames} frames @ {fps} fps)")
    return save_path


def plot_trajectories(
    trajectories: np.ndarray,
    goals: np.ndarray,
    obstacle_info,
    area_size: float,
    save_path: str = "trajectories.png",
    mode: str = "lqr",
    comm_radius: float = 1.5,
    is_swarm: bool = False,
    R_form: float = 0.3,
    thetas: Optional[np.ndarray] = None,
):
    """Save a static trajectory plot with triangle formations for swarm mode."""
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

    for center, half_size in obstacle_info:
        rect = mpatches.FancyBboxPatch(
            (center[0] - half_size[0], center[1] - half_size[1]),
            2 * half_size[0], 2 * half_size[1],
            boxstyle="round,pad=0.01",
            facecolor="#adb5bd", edgecolor="#6c757d",
            linewidth=1.2, alpha=0.65, zorder=1,
        )
        ax.add_patch(rect)

    # CoM trajectories
    for i in range(n_agents):
        path = trajectories[:, i, :]
        ax.plot(path[:, 0], path[:, 1], color=colors[i],
                linewidth=1.8, alpha=0.75, zorder=2,
                label=f"Swarm {i}" if is_swarm else f"Agent {i}")

    # Start positions
    starts = trajectories[0]
    if is_swarm and thetas is not None:
        for i in range(n_agents):
            verts = compute_triangle_vertices(
                starts[i, 0], starts[i, 1], thetas[0, i], R_form
            )
            tri = Polygon(verts, closed=True,
                          facecolor=colors[i], edgecolor="white",
                          linewidth=2, alpha=0.5, zorder=5)
            ax.add_patch(tri)
    else:
        for i in range(n_agents):
            ax.plot(starts[i, 0], starts[i, 1], marker="o", markersize=12,
                    markerfacecolor=colors[i], markeredgecolor="white",
                    markeredgewidth=2, zorder=5)

    # Goal positions
    for i in range(n_agents):
        ax.plot(goals[i, 0], goals[i, 1], marker="*", markersize=18,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)

    # Final positions — draw triangles or dots
    finals = trajectories[-1]
    if is_swarm and thetas is not None:
        for i in range(n_agents):
            verts = compute_triangle_vertices(
                finals[i, 0], finals[i, 1], thetas[-1, i], R_form
            )
            tri = Polygon(verts, closed=True,
                          facecolor=colors[i], edgecolor="white",
                          linewidth=1.5, alpha=0.8, zorder=6)
            ax.add_patch(tri)
            # Draw individual drone dots
            for k in range(3):
                ax.plot(verts[k, 0], verts[k, 1], marker="o", markersize=4,
                        markerfacecolor="white", markeredgecolor=colors[i],
                        markeredgewidth=1.0, zorder=7)
        # Also draw formation ghosts at evenly spaced intervals
        n_ghosts = min(8, trajectories.shape[0] // 10)
        if n_ghosts > 0:
            ghost_indices = np.linspace(0, trajectories.shape[0] - 1,
                                        n_ghosts, dtype=int)[1:-1]
            for idx in ghost_indices:
                for i in range(n_agents):
                    verts = compute_triangle_vertices(
                        trajectories[idx, i, 0],
                        trajectories[idx, i, 1],
                        thetas[idx, i], R_form
                    )
                    tri = Polygon(verts, closed=True,
                                  facecolor=colors[i], edgecolor=colors[i],
                                  linewidth=0.5, alpha=0.15, zorder=3)
                    ax.add_patch(tri)

    # Sensing radius at final position
    for i in range(n_agents):
        circle = plt.Circle(
            (finals[i, 0], finals[i, 1]), comm_radius,
            fill=False, linestyle="--", linewidth=1.0,
            edgecolor=colors[i], alpha=0.35, zorder=1,
        )
        ax.add_patch(circle)

    ax.set_title(
        f"{'Swarm' if is_swarm else 'Agent'} Trajectories  "
        f"({n_agents} {entity_name}, {trajectories.shape[0]-1} steps · {mode_label})",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    from matplotlib.lines import Line2D
    legend_handles = []
    for i in range(n_agents):
        legend_handles.append(
            Line2D([0], [0], color=colors[i], linewidth=2,
                   label=f"Swarm {i}" if is_swarm else f"Agent {i}"))
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=10, label="Start"))
    legend_handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=14, label="Goal (★)"))
    legend_handles.append(
        mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                       alpha=0.65, label="Obstacle"))
    if is_swarm:
        legend_handles.append(
            mpatches.Patch(facecolor="gray", edgecolor="white",
                           alpha=0.5, label="Formation △"))
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, framealpha=0.9, edgecolor="#dee2e6")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-agent trajectories (LQR or trained policy)"
    )
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--area_size", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--n_obs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="trajectories.mp4",
                        help="Output file (.mp4 for video, .png for static)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip", type=int, default=1,
                        help="Render every N-th step (video only)")
    parser.add_argument("--png", action="store_true",
                        help="Also save a static PNG snapshot")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (.pt). "
                             "If given, uses the trained policy; "
                             "otherwise uses the LQR controller.")
    parser.add_argument("--swarm_lqr", action="store_true",
                        help="Run SwarmIntegrator with LQR only (no checkpoint needed). "
                             "Useful for testing the swarm environment.")
    parser.add_argument("--force_lqr", action="store_true",
                        help="Load checkpoint config but ignore the trained policy; "
                             "use LQR only. For debugging.")
    args = parser.parse_args()

    print("Running simulation...")
    result = run_simulation(
        num_agents=args.num_agents,
        area_size=args.area_size,
        max_steps=args.max_steps,
        dt=args.dt,
        n_obs=args.n_obs,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        force_lqr=args.force_lqr,
        swarm_lqr=args.swarm_lqr,
    )
    trajectories, goals, obstacle_info, area, comm_r, mode, is_swarm, R_form, thetas = result
    entity = "swarms" if is_swarm else "agents"
    print(f"  Recorded {trajectories.shape[0]} frames for "
          f"{trajectories.shape[1]} {entity}.  Mode: {mode}")

    if args.save.endswith(".mp4"):
        print("Creating video...")
        create_video(
            trajectories=trajectories,
            goals=goals,
            obstacle_info=obstacle_info,
            area_size=area,
            save_path=args.save,
            fps=args.fps,
            skip=args.skip,
            mode=mode,
            comm_radius=comm_r,
            is_swarm=is_swarm,
            R_form=R_form,
            thetas=thetas,
        )
    else:
        print("Plotting static image...")
        plot_trajectories(
            trajectories=trajectories,
            goals=goals,
            obstacle_info=obstacle_info,
            area_size=area,
            save_path=args.save,
            mode=mode,
            comm_radius=comm_r,
            is_swarm=is_swarm,
            R_form=R_form,
            thetas=thetas,
        )

    if args.png:
        png_path = args.save.replace(".mp4", ".png") if args.save.endswith(".mp4") \
            else "trajectories.png"
        plot_trajectories(
            trajectories=trajectories,
            goals=goals,
            obstacle_info=obstacle_info,
            area_size=area,
            save_path=png_path,
            mode=mode,
            comm_radius=comm_r,
            is_swarm=is_swarm,
            R_form=R_form,
            thetas=thetas,
        )

    print("Done!")


if __name__ == "__main__":
    main()
