#!/usr/bin/env python3
"""
Training log visualizer.

Usage:
    # まずログをファイルに保存してから実行
    python -m gcbf_plus.train_swarm ... 2>&1 | tee train.log
    python plot_training_log.py train.log

    # 出力ファイル指定
    python plot_training_log.py train.log --output train_plot.png

    # 移動平均の窓サイズ変更
    python plot_training_log.py train.log --smooth 5
"""

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


# ── Parser ────────────────────────────────────────────────────────────

# Line 1: Step N | R: X (qp:X, pr:X, ar:X, av:X) | S: X (X-X) | GNN: X/X | grad: X[(clip)]
RE_STEP = re.compile(
    r"Step\s+(\d+)"
    r"\s*\|\s*R:\s*([-\d.]+)\s*\(qp:([-\d.]+),\s*pr:([-\d.]+),\s*ar:([-\d.]+),\s*av:([-\d.]+)\)"
    r"\s*\|\s*S:\s*([\d.]+)\s*\(([\d.]+)-([\d.]+)\)"
    r".*?GNN:\s*([\d.]+)/([\d.]+)"
    r"\s*\|\s*grad:\s*([\d.]+)(\(clip\))?"
)

# Line 2: Life: X (X-X) | Reset: X% [Goal:N(X%) Col:N(X%) TO:N(X%)] | Xs
RE_LIFE = re.compile(
    r"Life:\s*([\d.]+)\s*\(([\d.]+)-([\d.]+)\)"
    r"\s*\|\s*Reset:\s*([\d.]+)%"
    r"\s*\[Goal:(\d+)\(([\d.]+)%\)\s*Col:(\d+)\(([\d.]+)%\)\s*TO:(\d+)\(([\d.]+)%\)\]"
)


def parse_log(path: str) -> dict:
    data = {
        "step": [],
        "reward": [], "r_qp": [], "r_pr": [], "r_ar": [], "r_av": [],
        "scale_mean": [], "scale_min": [], "scale_max": [],
        "gnn_mean": [], "gnn_max": [],
        "grad": [], "grad_clip": [],
        "life_mean": [], "life_min": [], "life_max": [],
        "reset_rate": [],
        "goal_pct": [], "col_pct": [], "to_pct": [],
    }

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        m1 = RE_STEP.search(lines[i])
        if m1:
            # look ahead for life line
            m2 = None
            for j in range(i + 1, min(i + 4, len(lines))):
                m2 = RE_LIFE.search(lines[j])
                if m2:
                    break

            data["step"].append(int(m1.group(1)))
            data["reward"].append(float(m1.group(2)))
            data["r_qp"].append(float(m1.group(3)))
            data["r_pr"].append(float(m1.group(4)))
            data["r_ar"].append(float(m1.group(5)))
            data["r_av"].append(float(m1.group(6)))
            data["scale_mean"].append(float(m1.group(7)))
            data["scale_min"].append(float(m1.group(8)))
            data["scale_max"].append(float(m1.group(9)))
            data["gnn_mean"].append(float(m1.group(10)))
            data["gnn_max"].append(float(m1.group(11)))
            data["grad"].append(float(m1.group(12)))
            data["grad_clip"].append(m1.group(13) is not None)

            if m2:
                data["life_mean"].append(float(m2.group(1)))
                data["life_min"].append(float(m2.group(2)))
                data["life_max"].append(float(m2.group(3)))
                data["reset_rate"].append(float(m2.group(4)))
                data["goal_pct"].append(float(m2.group(6)))
                data["col_pct"].append(float(m2.group(8)))
                data["to_pct"].append(float(m2.group(10)))
            else:
                for k in ("life_mean", "life_min", "life_max", "reset_rate",
                          "goal_pct", "col_pct", "to_pct"):
                    data[k].append(float("nan"))
        i += 1

    return {k: np.array(v) for k, v in data.items()}


def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


# ── Plot ──────────────────────────────────────────────────────────────

def plot(data: dict, smooth_w: int, output: str):
    steps = data["step"]
    if len(steps) == 0:
        print("No data parsed — check log format.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Training Log", fontsize=14)

    def ax_plot(ax, y, label, color, fill_low=None, fill_high=None, ylabel=None, markers=None):
        ys = smooth(y, smooth_w)
        ax.plot(steps, ys, color=color, linewidth=1.5, label=label)
        if fill_low is not None and fill_high is not None:
            ax.fill_between(steps, smooth(fill_low, smooth_w), smooth(fill_high, smooth_w),
                            alpha=0.15, color=color)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel or label)
        ax.grid(True, alpha=0.3)
        if markers is not None:
            clip_steps = steps[markers]
            clip_vals  = ys[markers]
            ax.scatter(clip_steps, clip_vals, marker="x", color="red", s=20, zorder=5,
                       label="grad clip")
            ax.legend(fontsize=8)

    # (0,0) Total reward
    ax_plot(axes[0, 0], data["reward"], "Total Reward", "black")

    # (0,1) Reward components
    ax = axes[0, 1]
    for key, color, label in [
        ("r_pr",  "steelblue", "R_progress"),
        ("r_ar",  "green",     "R_arrival"),
        ("r_qp",  "red",       "R_qp"),
        ("r_av",  "orange",    "R_avoid"),
    ]:
        ax.plot(steps, smooth(data[key], smooth_w), color=color, linewidth=1.2, label=label)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Reward Components")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Formation scale
    ax_plot(axes[1, 0], data["scale_mean"], "Formation Scale",
            "purple", data["scale_min"], data["scale_max"], ylabel="s")

    # (1,1) GNN output magnitude
    ax = axes[1, 1]
    ax.plot(steps, smooth(data["gnn_mean"], smooth_w), color="teal", linewidth=1.5, label="mean")
    ax.plot(steps, smooth(data["gnn_max"],  smooth_w), color="teal", linewidth=1.0,
            linestyle="--", label="max")
    ax.fill_between(steps, smooth(data["gnn_mean"], smooth_w), smooth(data["gnn_max"], smooth_w),
                    alpha=0.1, color="teal")
    ax.set_title("GNN Output Magnitude")
    ax.set_xlabel("Step")
    ax.set_ylabel("|π|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,0) Gradient norm
    clip_mask = np.where(data["grad_clip"])[0]
    ax_plot(axes[2, 0], data["grad"], "Gradient Norm", "brown", ylabel="‖∇‖",
            markers=clip_mask if len(clip_mask) > 0 else None)

    # (2,1) Episode lifetime
    ax_plot(axes[2, 1], data["life_mean"], "Episode Life (steps)",
            "navy", data["life_min"], data["life_max"], ylabel="steps")

    # (3,0) Episode outcome %
    ax = axes[3, 0]
    for key, color, label in [
        ("goal_pct", "green",  "Goal %"),
        ("col_pct",  "red",    "Collision %"),
        ("to_pct",   "gray",   "Timeout %"),
    ]:
        valid = ~np.isnan(data[key])
        if valid.any():
            ax.plot(steps[valid], smooth(data[key][valid], smooth_w),
                    color=color, linewidth=1.2, label=label)
    ax.set_title("Episode Outcome")
    ax.set_xlabel("Step")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3,1) Reset rate
    ax_plot(axes[3, 1], data["reset_rate"], "Reset Rate (%)", "darkorange", ylabel="%")

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved: {output}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Path to training log file")
    parser.add_argument("--output", default="train_plot.png")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Moving average window (1 = no smoothing)")
    args = parser.parse_args()

    data = parse_log(args.log)
    print(f"Parsed {len(data['step'])} log entries "
          f"(step {data['step'][0] if len(data['step']) else '?'} "
          f"→ {data['step'][-1] if len(data['step']) else '?'})")
    plot(data, smooth_w=args.smooth, output=args.output)
