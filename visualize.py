#!/usr/bin/env python3
"""
Visualization: Animate multi-agent trajectories as an MP4 video.

Supports two modes:
  1. LQR nominal controller (default, no checkpoint needed)
  2. Trained GCBF+ policy   (pass --checkpoint gcbf_plus_checkpoint.pt)

Draws:
  - Start positions  (●  circles)
  - Goal positions   (★  stars)
  - Trajectories     (colored growing paths)
  - Obstacles        (gray rectangles)
  - Current agent positions (filled circles that move each frame)

Usage (Colab):
    # ── LQR mode (no training needed) ──
    !python visualize.py --num_agents 4 --seed 0

    # ── Trained policy mode ──
    !python -m gcbf_plus.train --num_agents 4 --num_steps 2000
    !python visualize.py --checkpoint gcbf_plus_checkpoint.pt --seed 0

    # Display in notebook:
    from IPython.display import Video
    Video("trajectories.mp4", embed=True)
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import torch

from gcbf_plus.env import DoubleIntegrator
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
    return policy_net, cfg


def run_simulation(
    num_agents: int = 4,
    area_size: float = 10.0,
    max_steps: int = 256,
    dt: float = 0.03,
    n_obs: int = 8,
    seed: int = 0,
    checkpoint_path: Optional[str] = None,
):
    """
    Run a simulation and record trajectories.

    If checkpoint_path is provided, uses the trained PolicyNetwork.
    Otherwise, uses the LQR nominal controller.

    Returns
    -------
    trajectories : np.ndarray, shape (T+1, num_agents, 2)
    goals        : np.ndarray, shape (num_agents, 2)
    obstacle_info: list of (center, half_size)
    area_size    : float
    mode         : str — "trained_policy" or "lqr"
    """

    # ── Load trained policy if checkpoint given ──────────────────────
    policy_net = None
    mode = "lqr"

    if checkpoint_path is not None:
        policy_net, cfg = load_trained_policy(checkpoint_path)
        # Override env params from the checkpoint config
        num_agents = cfg["num_agents"]
        area_size = cfg["area_size"]
        n_obs = cfg["n_obs"]
        dt = cfg["dt"]
        comm_radius = cfg["comm_radius"]
        mode = "trained_policy"
    else:
        comm_radius = 0.5  # default for LQR mode

    # ── Create environment ───────────────────────────────────────────
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
    trajectories.append(env.agent_states[:, :2].detach().numpy().copy())

    goals = env.goal_states[:, :2].detach().numpy().copy()

    obstacle_info = []
    for obs in env._obstacles:
        c = obs.center.numpy().copy()
        hs = obs.half_size.numpy().copy()
        obstacle_info.append((c, hs))

    # ── Simulate ─────────────────────────────────────────────────────
    for _ in range(max_steps):
        if policy_net is not None:
            # Use the trained policy π(x)
            with torch.no_grad():
                graph = env._get_graph()
                u = policy_net(graph)  # (n_agents, action_dim)
        else:
            # Use LQR nominal controller
            u = env.nominal_controller()

        _, info = env.step(u)
        trajectories.append(env.agent_states[:, :2].detach().numpy().copy())
        if info["done"]:
            break

    trajectories = np.array(trajectories)  # (T+1, n, 2)

    # Debug: print total displacement
    displacement = np.linalg.norm(trajectories[-1] - trajectories[0], axis=1)
    print(f"  Agent displacements: {displacement}")

    return trajectories, goals, obstacle_info, area_size, mode


def create_video(
    trajectories: np.ndarray,
    goals: np.ndarray,
    obstacle_info,
    area_size: float,
    save_path: str = "trajectories.mp4",
    fps: int = 30,
    skip: int = 1,
    mode: str = "lqr",
):
    """
    Create an MP4 animation of agent trajectories.
    """
    n_agents = trajectories.shape[1]
    total_frames = trajectories.shape[0]
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)]

    # Frame indices to render
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

    # ── Static elements: start markers (circles) ─────────────────────
    starts = trajectories[0]
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
    trail_lines = []
    for i in range(n_agents):
        (line,) = ax.plot([], [], color=colors[i], linewidth=1.8, alpha=0.75, zorder=2)
        trail_lines.append(line)

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

    # Step counter + mode label
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
        f"Multi-Agent Trajectories  ({n_agents} agents · {mode_label})",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    # ── Legend ────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    legend_handles = []
    for i in range(n_agents):
        legend_handles.append(
            Line2D([0], [0], color=colors[i], linewidth=2, label=f"Agent {i}")
        )
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=10, label="Start (●)")
    )
    legend_handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=14, label="Goal (★)")
    )
    legend_handles.append(
        mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                       alpha=0.65, label="Obstacle")
    )
    ax.legend(
        handles=legend_handles, loc="upper right",
        fontsize=9, framealpha=0.9, edgecolor="#dee2e6",
    )

    # ── Animation functions ──────────────────────────────────────────
    def init():
        for line in trail_lines:
            line.set_data([], [])
        for dot in current_dots:
            dot.set_data([], [])
        step_text.set_text("")
        return trail_lines + current_dots + [step_text]

    def update(frame_idx):
        sim_step = frame_indices[frame_idx]
        for i in range(n_agents):
            trail = trajectories[: sim_step + 1, i, :]
            trail_lines[i].set_data(trail[:, 0], trail[:, 1])
            current_dots[i].set_data([trajectories[sim_step, i, 0]],
                                     [trajectories[sim_step, i, 1]])
        step_text.set_text(f"Step {sim_step}/{total_frames - 1}  [{mode_label}]")
        return trail_lines + current_dots + [step_text]

    # ── Build animation ──────────────────────────────────────────────
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=n_frames,
        interval=1000 // fps,
        blit=True,
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
):
    """Save a static trajectory plot."""
    n_agents = trajectories.shape[1]
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)]
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

    for i in range(n_agents):
        path = trajectories[:, i, :]
        ax.plot(path[:, 0], path[:, 1], color=colors[i],
                linewidth=1.8, alpha=0.75, zorder=2, label=f"Agent {i}")
        n_dots = min(20, path.shape[0])
        indices = np.linspace(0, path.shape[0] - 1, n_dots, dtype=int)
        ax.scatter(path[indices, 0], path[indices, 1],
                   color=colors[i], s=8, alpha=0.35, zorder=3)

    starts = trajectories[0]
    for i in range(n_agents):
        ax.plot(starts[i, 0], starts[i, 1], marker="o", markersize=12,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=2, zorder=5)
    for i in range(n_agents):
        ax.plot(goals[i, 0], goals[i, 1], marker="*", markersize=18,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)

    ax.set_title(f"Multi-Agent Trajectories  ({n_agents} agents, "
                 f"{trajectories.shape[0]-1} steps · {mode_label})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    from matplotlib.lines import Line2D
    legend_handles = []
    for i in range(n_agents):
        legend_handles.append(
            Line2D([0], [0], color=colors[i], linewidth=2, label=f"Agent {i}"))
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=10, label="Start (●)"))
    legend_handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=14, label="Goal (★)"))
    legend_handles.append(
        mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                       alpha=0.65, label="Obstacle"))
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
    parser.add_argument("--area_size", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--n_obs", type=int, default=8)
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
    args = parser.parse_args()

    print("Running simulation...")
    trajectories, goals, obstacle_info, area, mode = run_simulation(
        num_agents=args.num_agents,
        area_size=args.area_size,
        max_steps=args.max_steps,
        dt=args.dt,
        n_obs=args.n_obs,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    print(f"  Recorded {trajectories.shape[0]} frames for "
          f"{trajectories.shape[1]} agents.  Mode: {mode}")

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
        )

    print("Done!")


if __name__ == "__main__":
    main()
