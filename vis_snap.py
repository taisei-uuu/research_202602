#!/usr/bin/env python3
"""
vis_snap.py — Static trajectory snapshot with formation-scale overlay.

Runs the same simulation as visualize.py but outputs a PNG showing:
  - Trajectory paths colored by formation scale s (RdYlGn colormap)
  - Bounding circles at regular intervals (radius = R_form·s + r_margin)
  - Formation triangles at start and end
  - Obstacles, start positions (○), goal positions (★)

Usage:
    python vis_snap.py --checkpoint ./checkpoints/affine_swarm_400.pt --scenario scenarios/crossing.json
    python vis_snap.py --checkpoint ./checkpoints/affine_swarm_400.pt --method hocbf_lqr --output hocbf_snap.png
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D
import numpy as np

from visualize import run_simulation, AGENT_COLORS, compute_triangle_vertices


# ═══════════════════════════════════════════════════════════════════════
# Snapshot plot
# ═══════════════════════════════════════════════════════════════════════

def plot_snapshot(
    trajectories: np.ndarray,   # (T, N, 4)
    goals: np.ndarray,          # (N, 2)
    obstacle_info: list,        # [(center, radius), ...]
    area_size: float,
    scale_traj: Optional[np.ndarray],  # (T, N, 2)  s, s_dot
    R_form: float,
    r_margin: float,
    s_min: float = 0.4,
    s_max: float = 1.5,
    mode: str = "trained_policy",
    save_path: str = "snapshot.png",
    stride: int = 20,
):
    """Plot trajectory snapshot with scale-colored paths and bounding circles."""
    T, N, _ = trajectories.shape
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(N)]

    _mode_labels = {
        "trained_policy": "Trained Policy π(x)",
        "hocbf_lqr":      "HOCBF+LQR",
        "lqr":            "LQR Controller",
    }
    mode_label = _mode_labels.get(mode, mode)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(-0.3, area_size + 0.3)
    ax.set_ylim(-0.3, area_size + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#adb5bd")

    # ── Obstacles ────────────────────────────────────────────────────────
    for center, radius in obstacle_info:
        ax.add_patch(Circle(
            (center[0], center[1]), radius,
            facecolor="#adb5bd", edgecolor="#6c757d",
            linewidth=1.2, alpha=0.65, zorder=1,
        ))

    # ── Trajectory paths colored by scale ───────────────────────────────
    scale_norm = plt.Normalize(vmin=s_min, vmax=s_max)
    cmap = cm.get_cmap("RdYlGn")

    for i in range(N):
        path = trajectories[:, i, :2]  # (T, 2)

        if scale_traj is not None:
            scales = scale_traj[:, i, 0]  # (T,)
        else:
            scales = np.ones(T)

        # LineCollection: each segment colored by midpoint scale
        pts = path.reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        scale_mid = (scales[:-1] + scales[1:]) / 2.0

        lc = LineCollection(segments, cmap=cmap, norm=scale_norm,
                            linewidth=2.5, zorder=3, alpha=0.9)
        lc.set_array(scale_mid)
        ax.add_collection(lc)

    # ── Bounding circles at stride intervals ─────────────────────────────
    for i in range(N):
        step_indices = list(range(0, T, stride))
        if (T - 1) not in step_indices:
            step_indices.append(T - 1)

        for t in step_indices:
            pos_t = trajectories[t, i, :2]
            s_t = float(scale_traj[t, i, 0]) if scale_traj is not None else 1.0
            r_t = R_form * s_t + r_margin

            # Alpha: faint at start, more visible toward end
            alpha = 0.08 + 0.18 * (t / max(T - 1, 1))
            ax.add_patch(Circle(
                (pos_t[0], pos_t[1]), r_t,
                facecolor=colors[i], edgecolor=colors[i],
                linewidth=0.5, alpha=alpha, zorder=2,
            ))

    # ── Formation triangles at start and end ─────────────────────────────
    for i in range(N):
        # Start
        s0 = float(scale_traj[0, i, 0]) if scale_traj is not None else 1.0
        verts0 = compute_triangle_vertices(
            trajectories[0, i, 0], trajectories[0, i, 1], 0.0, R_form * s0)
        ax.add_patch(Polygon(verts0, closed=True, facecolor=colors[i],
                             edgecolor="white", linewidth=1.5, alpha=0.35, zorder=5))

        # End
        sT = float(scale_traj[-1, i, 0]) if scale_traj is not None else 1.0
        vertsT = compute_triangle_vertices(
            trajectories[-1, i, 0], trajectories[-1, i, 1], 0.0, R_form * sT)
        ax.add_patch(Polygon(vertsT, closed=True, facecolor=colors[i],
                             edgecolor="white", linewidth=1.5, alpha=0.75, zorder=5))

    # ── Start positions ───────────────────────────────────────────────────
    for i in range(N):
        ax.plot(trajectories[0, i, 0], trajectories[0, i, 1],
                marker="o", markersize=10,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=2, zorder=6)

    # ── Goal positions ────────────────────────────────────────────────────
    for i in range(N):
        ax.plot(goals[i, 0], goals[i, 1],
                marker="*", markersize=18,
                markerfacecolor=colors[i], markeredgecolor="white",
                markeredgewidth=1.2, zorder=6)

    # ── Legend ────────────────────────────────────────────────────────────
    handles = []
    for i in range(N):
        handles.append(Line2D([0], [0], color=colors[i], linewidth=2.5,
                              label=f"Swarm {i}"))
    handles.append(Line2D([0], [0], marker="*", color="w",
                          markerfacecolor="gray", markeredgecolor="white",
                          markersize=14, label="Goal (★)"))
    handles.append(mpatches.Patch(facecolor="#adb5bd", edgecolor="#6c757d",
                                  alpha=0.65, label="Obstacle"))
    handles.append(mpatches.Patch(facecolor="gray", edgecolor="white",
                                  alpha=0.35, label="Formation △ start"))
    handles.append(mpatches.Patch(facecolor="gray", edgecolor="white",
                                  alpha=0.75, label="Formation △ end"))
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#dee2e6")

    ax.set_title(
        f"Swarm Trajectories  [{mode_label}]\n"
        f"({N} swarms · {T - 1} steps  |  path colored by scale s)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Snapshot saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Static trajectory snapshot with formation-scale overlay"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--method", type=str, default="affine_policy",
                        choices=["affine_policy", "hocbf_lqr", "lqr_only"])
    parser.add_argument("--scenario", type=str, default=None,
                        help="Path to scenario JSON")
    parser.add_argument("--n_obs", type=int, default=None)
    parser.add_argument("--no_scale", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--swarm_lqr", action="store_true",
                        help="Use SwarmIntegrator without checkpoint")
    parser.add_argument("--stride", type=int, default=20,
                        help="Draw bounding circle every N steps (default: 20)")
    parser.add_argument("--output", type=str, default="snapshot.png",
                        help="Output PNG path (default: snapshot.png)")
    parser.add_argument("--exact_qp", action="store_true", default=False,
                        help="Use exact QP solver (quadprog) instead of Dykstra projection")
    args = parser.parse_args()

    print("Running simulation...")
    result = run_simulation(
        max_steps=args.max_steps,
        n_obs=args.n_obs,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        swarm_lqr=args.swarm_lqr,
        scenario_path=args.scenario,
        no_scale=args.no_scale,
        method=args.method,
        use_exact_qp=args.exact_qp,
    )
    (trajectories, goals, obstacle_info, area, _comm_r, mode,
     _is_swarm, R_form, r_margin, _payload_traj, _cable_length,
     scale_traj, _edge_traj, _lidar_traj) = result

    print(f"  {trajectories.shape[0]} frames, {trajectories.shape[1]} swarms.  Mode: {mode}")

    # Recover s_min / s_max from checkpoint if available
    try:
        import torch
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = ckpt.get("config", {})
        s_min = cfg.get("s_min", 0.4)
        s_max = cfg.get("s_max", 1.5)
    except Exception:
        s_min, s_max = 0.4, 1.5

    print("Plotting snapshot...")
    plot_snapshot(
        trajectories=trajectories,
        goals=goals,
        obstacle_info=obstacle_info,
        area_size=area,
        scale_traj=scale_traj,
        R_form=R_form,
        r_margin=r_margin,
        s_min=s_min,
        s_max=s_max,
        mode=mode,
        save_path=args.output,
        stride=args.stride,
    )
    print("Done!")


if __name__ == "__main__":
    main()
