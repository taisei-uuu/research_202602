#!/usr/bin/env python3
"""
Visualization: Plot multi-agent trajectories using matplotlib.

Draws:
  - Start positions  (●  circles)
  - Goal positions   (★  stars)
  - Trajectories     (colored paths)
  - Obstacles        (gray rectangles)

Usage:
    python visualize.py
    python visualize.py --num_agents 6 --max_steps 300 --seed 42
"""

from __future__ import annotations

import argparse
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from gcbf_plus.env import DoubleIntegrator


# ── Color palette ────────────────────────────────────────────────────────
# A curated palette that looks good on both white and dark backgrounds.
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


def run_simulation(
    num_agents: int = 4,
    area_size: float = 10.0,
    max_steps: int = 256,
    dt: float = 0.03,
    n_obs: int = 8,
    seed: int = 0,
):
    """
    Run a simulation with the LQR nominal controller and record trajectories.

    Returns
    -------
    trajectories : np.ndarray, shape (max_steps+1, num_agents, 2)
        Position (px, py) at each time-step for every agent.
    goals : np.ndarray, shape (num_agents, 2)
        Goal positions.
    obstacles : list of (center, half_size)  — obstacle rectangles.
    area_size : float
    """
    env = DoubleIntegrator(
        num_agents=num_agents,
        area_size=area_size,
        dt=dt,
        max_steps=max_steps,
        params={"n_obs": n_obs},
    )

    env.reset(seed=seed)

    # Store trajectories — list of (num_agents, 2) position snapshots
    trajectories: List[np.ndarray] = []
    trajectories.append(env.agent_states[:, :2].detach().numpy().copy())

    goals = env.goal_states[:, :2].detach().numpy().copy()

    # Collect obstacle info for plotting
    obstacle_info = []
    for obs in env._obstacles:
        c = obs.center.numpy().copy()
        hs = obs.half_size.numpy().copy()
        obstacle_info.append((c, hs))

    # Run simulation
    for _ in range(max_steps):
        u = env.nominal_controller()
        _, info = env.step(u)
        trajectories.append(env.agent_states[:, :2].detach().numpy().copy())
        if info["done"]:
            break

    trajectories = np.array(trajectories)  # (T+1, n, 2)
    return trajectories, goals, obstacle_info, area_size


def plot_trajectories(
    trajectories: np.ndarray,
    goals: np.ndarray,
    obstacle_info,
    area_size: float,
    save_path: str = "trajectories.png",
    show: bool = True,
):
    """
    Plot agent trajectories, start/goal markers, and obstacles.

    Parameters
    ----------
    trajectories : (T+1, n_agents, 2)
    goals        : (n_agents, 2)
    obstacle_info: list of (center, half_size) tuples
    area_size    : float
    save_path    : str — path to save the figure
    show         : bool — whether to call plt.show()
    """
    n_agents = trajectories.shape[1]
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)]

    fig, ax = plt.subplots(figsize=(9, 9))

    # ── Background & grid ────────────────────────────────────────────
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(-0.3, area_size + 0.3)
    ax.set_ylim(-0.3, area_size + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#adb5bd")

    # ── Obstacles ────────────────────────────────────────────────────
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

    # ── Paths ────────────────────────────────────────────────────────
    for i in range(n_agents):
        path = trajectories[:, i, :]  # (T+1, 2)
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=colors[i],
            linewidth=1.8,
            alpha=0.75,
            zorder=2,
            label=f"Agent {i}",
        )
        # Fading trail effect: draw lighter intermediate dots
        n_dots = min(20, path.shape[0])
        indices = np.linspace(0, path.shape[0] - 1, n_dots, dtype=int)
        ax.scatter(
            path[indices, 0],
            path[indices, 1],
            color=colors[i],
            s=8,
            alpha=0.35,
            zorder=3,
        )

    # ── Start positions (circles) ────────────────────────────────────
    starts = trajectories[0]  # (n, 2)
    for i in range(n_agents):
        ax.plot(
            starts[i, 0],
            starts[i, 1],
            marker="o",
            markersize=12,
            markerfacecolor=colors[i],
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=5,
        )

    # ── Goal positions (stars) ───────────────────────────────────────
    for i in range(n_agents):
        ax.plot(
            goals[i, 0],
            goals[i, 1],
            marker="*",
            markersize=18,
            markerfacecolor=colors[i],
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=5,
        )

    # ── Title & labels ───────────────────────────────────────────────
    ax.set_title(
        f"Multi-Agent Trajectories  ({n_agents} agents, "
        f"{trajectories.shape[0]-1} steps)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    # ── Legend ────────────────────────────────────────────────────────
    # Build custom legend entries
    from matplotlib.lines import Line2D

    legend_handles = []
    for i in range(n_agents):
        legend_handles.append(
            Line2D([0], [0], color=colors[i], linewidth=2, label=f"Agent {i}")
        )
    legend_handles.append(
        Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor="gray",
            markeredgecolor="white", markersize=10,
            label="Start (●)",
        )
    )
    legend_handles.append(
        Line2D(
            [0], [0],
            marker="*", color="w", markerfacecolor="gray",
            markeredgecolor="white", markersize=14,
            label="Goal (★)",
        )
    )
    legend_handles.append(
        mpatches.Patch(
            facecolor="#adb5bd", edgecolor="#6c757d", alpha=0.65,
            label="Obstacle",
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#dee2e6",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-agent trajectories (LQR nominal controller)"
    )
    parser.add_argument("--num_agents", type=int, default=4,
                        help="Number of agents")
    parser.add_argument("--area_size", type=float, default=10.0,
                        help="Side length of the square arena")
    parser.add_argument("--max_steps", type=int, default=256,
                        help="Maximum simulation steps")
    parser.add_argument("--dt", type=float, default=0.03,
                        help="Simulation time-step")
    parser.add_argument("--n_obs", type=int, default=8,
                        help="Number of rectangular obstacles")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default="trajectories.png",
                        help="Output file path")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the plot (useful for headless)")
    args = parser.parse_args()

    print("Running simulation...")
    trajectories, goals, obstacle_info, area = run_simulation(
        num_agents=args.num_agents,
        area_size=args.area_size,
        max_steps=args.max_steps,
        dt=args.dt,
        n_obs=args.n_obs,
        seed=args.seed,
    )
    print(f"  Recorded {trajectories.shape[0]} frames for {args.num_agents} agents.")

    print("Plotting...")
    plot_trajectories(
        trajectories=trajectories,
        goals=goals,
        obstacle_info=obstacle_info,
        area_size=area,
        save_path=args.save,
        show=not args.no_show,
    )
    print("Done!")


if __name__ == "__main__":
    main()
