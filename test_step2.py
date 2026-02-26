#!/usr/bin/env python3
"""
Smoke tests for GCBF+ Step 2: Training Loop, Loss Functions, and QP Solver.

Run with:
    python test_step2.py
"""

import sys
import numpy as np
import torch

# ── Helpers ──────────────────────────────────────────────────────────────

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
n_passed = 0
n_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global n_passed, n_failed
    if condition:
        n_passed += 1
        print(f"  [{PASS}] {name}")
    else:
        n_failed += 1
        print(f"  [{FAIL}] {name}  — {detail}")


# ═══════════════════════════════════════════════════════════════════════
# 1. Environment Extensions (safe/unsafe masks, differentiable graph)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 1. Environment Extensions")
print("=" * 60)

from gcbf_plus.env import DoubleIntegrator

NUM_AGENTS = 4
env = DoubleIntegrator(
    num_agents=NUM_AGENTS,
    area_size=10.0,
    params={"comm_radius": 1.0, "n_obs": 4},
)
graph = env.reset(seed=42)

# 1a. Safe/unsafe masks
safe_m = env.safe_mask()
unsafe_m = env.unsafe_mask()
check(
    f"safe_mask shape == ({NUM_AGENTS},)",
    safe_m.shape == (NUM_AGENTS,),
    f"got {safe_m.shape}",
)
check(
    f"unsafe_mask shape == ({NUM_AGENTS},)",
    unsafe_m.shape == (NUM_AGENTS,),
    f"got {unsafe_m.shape}",
)
check(
    "safe ∩ unsafe == ∅ (mutually exclusive)",
    not (safe_m & unsafe_m).any().item(),
)
check(
    "safe ∪ unsafe == all (complete coverage)",
    (safe_m | unsafe_m).all().item(),
)

# 1b. Differentiable graph
agent_states = env.agent_states.clone().requires_grad_(True)
g = env.build_graph_differentiable(agent_states)
check("differentiable graph has edges", g.n_edge >= 0)

# 1c. state_dot
action = torch.zeros(NUM_AGENTS, 2)
x_dot = env.state_dot(agent_states, action)
check(
    f"state_dot shape == ({NUM_AGENTS}, 4)",
    x_dot.shape == (NUM_AGENTS, 4),
    f"got {x_dot.shape}",
)

# 1d. forward_step
next_states = env.forward_step(agent_states, action)
check(
    f"forward_step shape == ({NUM_AGENTS}, 4)",
    next_states.shape == (NUM_AGENTS, 4),
    f"got {next_states.shape}",
)


# ═══════════════════════════════════════════════════════════════════════
# 2. Autograd Lie Derivative
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 2. Autograd Lie Derivative")
print("=" * 60)

from gcbf_plus.nn import GCBFNetwork, PolicyNetwork
from gcbf_plus.algo.loss import compute_lie_derivative

# Set up networks
gcbf_net = GCBFNetwork(
    node_dim=env.node_dim, edge_dim=env.edge_dim, n_agents=NUM_AGENTS
)
policy_net = PolicyNetwork(
    node_dim=env.node_dim,
    edge_dim=env.edge_dim,
    action_dim=env.action_dim,
    n_agents=NUM_AGENTS,
)

# Build graph with grad-tracked states
agent_states = env.agent_states.clone().requires_grad_(True)
graph = env.build_graph_differentiable(agent_states)

h = gcbf_net(graph).squeeze(-1)          # (n_agents,)
pi_action = policy_net(graph)            # (n_agents, 2)
x_dot = env.state_dot(agent_states, pi_action)

h_dot, dh_dx = compute_lie_derivative(h, agent_states, x_dot)

check(
    f"h_dot shape == ({NUM_AGENTS},)",
    h_dot.shape == (NUM_AGENTS,),
    f"got {h_dot.shape}",
)
check(
    f"dh_dx shape == ({NUM_AGENTS}, 4)",
    dh_dx.shape == (NUM_AGENTS, 4),
    f"got {dh_dx.shape}",
)
check("h_dot is finite", torch.isfinite(h_dot).all().item())
check("dh_dx is finite", torch.isfinite(dh_dx).all().item())
check(
    "h_dot has grad_fn (autograd connected)",
    h_dot.grad_fn is not None,
)


# ═══════════════════════════════════════════════════════════════════════
# 3. QP Solver
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 3. CBF-QP Solver (cvxpy)")
print("=" * 60)

from gcbf_plus.algo.qp_solver import solve_cbf_qp

u_nom = env.nominal_controller()
x_dot_f = torch.zeros_like(env.agent_states)
x_dot_f[:, :2] = env.agent_states[:, 2:]

u_qp = solve_cbf_qp(
    u_nom=u_nom,
    h=h.detach(),
    dh_dx=dh_dx.detach(),
    x_dot_f=x_dot_f,
    B_mat=env._B,
    alpha=1.0,
)

check(
    f"u_qp shape == ({NUM_AGENTS}, 2)",
    u_qp.shape == (NUM_AGENTS, 2),
    f"got {u_qp.shape}",
)
check("u_qp is finite", torch.isfinite(u_qp).all().item())
check("u_qp dtype matches u_nom", u_qp.dtype == u_nom.dtype)


# ═══════════════════════════════════════════════════════════════════════
# 4. Loss Function
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 4. Loss Function")
print("=" * 60)

from gcbf_plus.algo.loss import compute_loss

total_loss, info = compute_loss(
    h=h,
    h_dot=h_dot,
    pi_action=pi_action,
    u_qp=u_qp,
    safe_mask=safe_m,
    unsafe_mask=unsafe_m,
)

check("total_loss is scalar", total_loss.dim() == 0)
check("total_loss is finite", torch.isfinite(total_loss).item())
check("total_loss has grad_fn", total_loss.grad_fn is not None)
check("info has loss/total", "loss/total" in info)
check("info has acc/h_dot", "acc/h_dot" in info)

# Print info
for k, v in sorted(info.items()):
    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════
# 5. End-to-End Training Step
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 5. End-to-End Training Step")
print("=" * 60)

optim_cbf = torch.optim.Adam(gcbf_net.parameters(), lr=3e-5)
optim_actor = torch.optim.Adam(policy_net.parameters(), lr=3e-5)

# Snapshot params before update
param_before = {n: p.clone() for n, p in gcbf_net.named_parameters()}

# Forward + backward
optim_cbf.zero_grad()
optim_actor.zero_grad()

# Rebuild graph (clean computation graph)
agent_states2 = env.agent_states.clone().requires_grad_(True)
g2 = env.build_graph_differentiable(agent_states2)
h2 = gcbf_net(g2).squeeze(-1)
pi2 = policy_net(g2)
xd2 = env.state_dot(agent_states2, pi2)
hd2, dhx2 = compute_lie_derivative(h2, agent_states2, xd2)

x_dot_f2 = torch.zeros_like(agent_states2.detach())
x_dot_f2[:, :2] = agent_states2[:, 2:].detach()
with torch.no_grad():
    uq2 = solve_cbf_qp(u_nom, h2.detach(), dhx2.detach(), x_dot_f2, env._B)

loss2, _ = compute_loss(h2, hd2, pi2, uq2, safe_m, unsafe_m)
loss2.backward()

# Check gradients exist
grad_ok = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in gcbf_net.parameters()
    if p.requires_grad
)
check("GCBF gradients are finite after backward", grad_ok)

grad_ok_a = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in policy_net.parameters()
    if p.requires_grad
)
check("Policy gradients are finite after backward", grad_ok_a)

# Clip and step
torch.nn.utils.clip_grad_norm_(gcbf_net.parameters(), 2.0)
torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 2.0)
optim_cbf.step()
optim_actor.step()

# Check params changed
any_changed = any(
    not torch.equal(p, param_before[n])
    for n, p in gcbf_net.named_parameters()
)
check("GCBF parameters updated after optimizer step", any_changed)


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = n_passed + n_failed
print(f" Results: {n_passed}/{total} passed, {n_failed} failed")
print("=" * 60)

sys.exit(0 if n_failed == 0 else 1)
