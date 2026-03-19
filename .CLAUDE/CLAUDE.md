# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of a hierarchical distributed multi-agent control system for drone swarms carrying payloads. The core innovation is the **affine-transform architecture**: 3-drone swarms are controlled via translation + formation-scale parameters (not individual positions), while a payload pendulum swings beneath. A GNN policy outputs control offsets, which a QP solver filters for safety.

Reference paper: "GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control"

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run integration smoke test (most useful)
python test_swarm.py

# Other smoke tests
python test_step1.py   # basic double integrator
python test_step2.py   # GCBF network
python test_h_vals.py  # CBF constraint values

# Training
python -m gcbf_plus.train_swarm --num_steps 2000 --batch_size 128 --horizon 128 --save_interval 100

# Evaluation
python -m evaluate --checkpoint ./checkpoints/affine_swarm_1000.pt --methods affine_policy,hocbf_lqr,lqr_only

# Visualization
python visualize.py --checkpoint ./checkpoints/affine_swarm_1000.pt
```

## Architecture

The control pipeline is 4-level hierarchical:

```
GNN π(x) → velocity offsets (tanh scaled)
  ↓
LQR nominal controller + PD tracking → u_nom (nominal acceleration)
  ↓
Analytical QP solver (Dykstra projection, 3-5 iterations)
  - HOCBF payload swing (soft, slack variable)
  - Scale-CBF (hard)
  - Obstacle-CBF (hard, 2nd order)
  - Box constraint (hard)
  ↓
Environment dynamics (double integrator + nonlinear pendulum)
```

**System state per swarm:** agent CoM `[px, py, vx, vy]`, formation scale `[s, ṡ]`, payload swing `[γx, γy, γ̇x, γ̇y]`. Control input: `[a_cx, a_cy, a_s]`.

### Training Loop (`gcbf_plus/train_swarm.py`)

1. **Data collection** (`no_grad`): Roll out B vectorized environments for `horizon` steps; build mega-graph, run GNN → QP → env step, record snapshots.
2. **Pool shuffle**: Treat `(horizon × batch_size)` time-series as i.i.d.
3. **Mini-batch training** (4 passes): GNN forward (with grad) → recompute u_nom → QP detached → loss → Adam.

**Loss** (`gcbf_plus/algo/loss.py`):
- `L_progress`: `E[v_target · goal_direction]` (coef=1.0)
- `L_qp`: `E[||u_AT - u_QP||²]` (coef=2.0, QP intervention penalty)
- `L_effort`: `E[||π(x)||²]` (coef=0.01, regularization)

### GNN Architecture (`gcbf_plus/nn/gnn.py`)

Attention-based message passing. For each layer:
- `ψ₁` encoder: `[node_feat+edge_feat → 256 → 256 → 128]`
- `ψ₂` attention: `[128 → 128 → 128 → 1]` + softmax per receiver
- `ψ₃` value: `[128 → 256 → 128]`
- `ψ₄` decoder: `[128 → 256 → 256 → action_dim]`

### Key Files

| File | Role |
|------|------|
| `gcbf_plus/train_swarm.py` | Main training entry point |
| `gcbf_plus/algo/affine_qp_solver.py` | GPU-accelerated analytical QP (Dykstra) |
| `gcbf_plus/algo/loss.py` | Training objective |
| `gcbf_plus/env/swarm_integrator.py` | Single swarm: state, dynamics, graph |
| `gcbf_plus/env/vectorized_swarm.py` | Batched env for training |
| `gcbf_plus/nn/gnn.py` | GNN message-passing layer |
| `gcbf_plus/nn/policy_net.py` | Wraps GNN, outputs 3D action |
| `gcbf_plus/utils/swarm_graph.py` | Builds GraphsTuple from agent/goal/obstacle state |
| `evaluate.py` | Pluggable method registry (affine_policy, hocbf_lqr, lqr_only) |

### System Parameters

Δt=0.05s, m=0.1kg, u_max=0.3N, v_max=1.0m/s, arena=15×15m, R_form=0.5m, s∈[0.4,1.5], ṡ_max=1.0, R_comm=3.0m, cable l=1.0m, g=9.81m/s².

Optimizer: Adam, lr=3e-4, gradient clip max_norm=2.0.

### Documentation

- `algorithm_spec.md` — full algorithm specification (9 sections)
- `implementation_differences.md.resolved` — deviations from the paper
