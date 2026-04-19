#!/usr/bin/env python3
"""
Smoke tests for Affine-Transform Swarm architecture.

Tests:
  1. SwarmIntegrator with scale state and 3D affine action
  2. VectorizedSwarmEnv with batched scale dynamics
  3. PolicyNetwork with action_dim=3
  4. Affine QP solver constraints
  5. Training loop (5 steps)

Run with:
    python test_swarm.py
"""

import sys
import torch
import numpy as np

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
# 1. SwarmIntegrator Environment Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 1. SwarmIntegrator (Affine Transform)")
print("=" * 60)

from gcbf_plus.env import SwarmIntegrator

NUM_AGENTS = 3
env = SwarmIntegrator(num_agents=NUM_AGENTS, area_size=10.0, dt=0.03)

# Properties
check("state_dim == 4", env.state_dim == 4)
check("action_dim == 3", env.action_dim == 3, f"got {env.action_dim}")
check("node_dim == 3", env.node_dim == 3)
check("edge_dim == 5", env.edge_dim == 5)

# Reset
env.reset(seed=42)
check(f"agent_states shape == ({NUM_AGENTS}, 4)",
      env.agent_states.shape == (NUM_AGENTS, 4),
      f"got {env.agent_states.shape}")
check(f"scale_states shape == ({NUM_AGENTS}, 2)",
      env.scale_states.shape == (NUM_AGENTS, 2),
      f"got {env.scale_states.shape}")
check("scale_states initialized to [1.0, 0.0]",
      torch.allclose(env.scale_states, torch.tensor([[1.0, 0.0]] * NUM_AGENTS)),
      f"got {env.scale_states}")
check(f"payload_states shape == ({NUM_AGENTS}, 4)",
      env.payload_states.shape == (NUM_AGENTS, 4))

# Step with 3D action
action = torch.zeros(NUM_AGENTS, 3)
_, info = env.step(action)
check("step with 3D action works", True)
check("step returns info dict", "done" in info)

# Nominal controller shape
u_ref = env.nominal_controller()
check(f"nominal_controller shape == ({NUM_AGENTS}, 3)",
      u_ref.shape == (NUM_AGENTS, 3),
      f"got {u_ref.shape}")
check("nominal a_s == 0", (u_ref[:, 2] == 0).all().item())

# Scale dynamics
env.reset(seed=42)
action = torch.zeros(NUM_AGENTS, 3)
action[:, 2] = 1.0  # positive scale acceleration
env.step(action)
check("scale increases with positive a_s",
      (env.scale_states[:, 0] > 1.0).all().item(),
      f"got s={env.scale_states[:, 0].tolist()}")
check("s_dot > 0 after positive a_s",
      (env.scale_states[:, 1] > 0).all().item())

# Scale clamping
env.reset(seed=42)
action = torch.zeros(NUM_AGENTS, 3)
action[:, 2] = -100.0  # extreme negative
for _ in range(100):
    env.step(action)
check(f"scale clamped at s_min={env.params['s_min']}",
      (env.scale_states[:, 0] >= env.params["s_min"]).all().item(),
      f"got s={env.scale_states[:, 0].tolist()}")

# Dynamic r_swarm
s_test = torch.tensor([0.5, 1.0, 1.5])
r = env.r_swarm(s_test)
check("r_swarm(0.5) < r_swarm(1.0) < r_swarm(1.5)",
      r[0] < r[1] < r[2], f"got {r.tolist()}")


# ═══════════════════════════════════════════════════════════════════════
# 2. VectorizedSwarmEnv Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 2. VectorizedSwarmEnv (Batched Affine)")
print("=" * 60)

from gcbf_plus.env.vectorized_swarm import VectorizedSwarmEnv

B = 4
vec_env = VectorizedSwarmEnv(num_agents=NUM_AGENTS, batch_size=B, area_size=10.0,
                              params={"n_obs": 2})
vec_env.reset(torch.device("cpu"))

check(f"action_dim == 3", vec_env.action_dim == 3)
check(f"_agent_states shape == ({B}, {NUM_AGENTS}, 4)",
      vec_env._agent_states.shape == (B, NUM_AGENTS, 4),
      f"got {vec_env._agent_states.shape}")
check(f"_scale_states shape == ({B}, {NUM_AGENTS}, 2)",
      vec_env._scale_states.shape == (B, NUM_AGENTS, 2),
      f"got {vec_env._scale_states.shape}")
check("scale init s=1.0",
      (vec_env._scale_states[:, :, 0] == 1.0).all().item())

# Batched step
action = torch.zeros(B, NUM_AGENTS, 3)
action[:, :, 2] = 0.5  # scale accel
vec_env.step(action)
check("batched step with 3D action works", True)
check("batched scale increases",
      (vec_env._scale_states[:, :, 0] > 1.0).all().item())

# Nominal controller
u_nom = vec_env.nominal_controller()
check(f"nominal shape == ({B}, {NUM_AGENTS}, 3)",
      u_nom.shape == (B, NUM_AGENTS, 3),
      f"got {u_nom.shape}")

# Unsafe mask with dynamic radii
mask = vec_env.unsafe_mask()
check(f"unsafe_mask shape == ({B}, {NUM_AGENTS})",
      mask.shape == (B, NUM_AGENTS),
      f"got {mask.shape}")

# Graph builder
graph = vec_env.build_batch_graph()
check("graph building works", graph is not None)
check(f"graph n_node == {B * (NUM_AGENTS * 2 + 2)}",
      graph.n_node == B * (NUM_AGENTS * 2 + 2),
      f"got {graph.n_node}")


# ═══════════════════════════════════════════════════════════════════════
# 3. PolicyNetwork Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 3. PolicyNetwork (3D Affine Output)")
print("=" * 60)

from gcbf_plus.nn import PolicyNetwork

policy = PolicyNetwork(node_dim=3, edge_dim=5, action_dim=3, n_agents=NUM_AGENTS)
check(f"policy.action_dim == 3", policy.action_dim == 3)

# Forward pass on single-instance graph
env_test = SwarmIntegrator(num_agents=NUM_AGENTS, area_size=10.0,
                           params={"n_obs": 2, "comm_radius": 100.0})
env_test.reset(seed=100)
graph_test = env_test._get_graph()
u = policy(graph_test)
check(f"policy output shape == ({NUM_AGENTS}, 3)",
      u.shape == (NUM_AGENTS, 3),
      f"got {u.shape}")
check("policy output is finite", torch.isfinite(u).all().item())

# Gradient flow
loss = u.sum()
loss.backward()
grad_ok = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in policy.parameters()
    if p.requires_grad
)
check("policy backward() succeeds with finite grads", grad_ok)


# ═══════════════════════════════════════════════════════════════════════
# 4. Affine QP Solver Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 4. Affine QP Solver")
print("=" * 60)

from gcbf_plus.algo.affine_qp_solver import solve_affine_qp

N = 5
u_at = torch.randn(N, 3)
s_val = torch.ones(N)
s_dot_val = torch.zeros(N)

# a) Basic call (no obstacles)
u_qp = solve_affine_qp(
    u_at=u_at,
    s=s_val, s_dot=s_dot_val,
    s_min=0.4, s_max=1.5,
)
check("QP returns correct shape", u_qp.shape == (N, 3), f"got {u_qp.shape}")
check("QP output is finite", torch.isfinite(u_qp).all().item())

# b) Scale lower bound: if s is at s_min and s_dot < 0, a_s should be ≥ 0-ish
s_low = torch.full((N,), 0.4)  # at s_min
s_dot_neg = torch.full((N,), -0.5)  # falling
u_at_neg = torch.zeros(N, 3)
u_at_neg[:, 2] = -5.0  # big negative a_s
u_qp_low = solve_affine_qp(
    u_at=u_at_neg, s=s_low, s_dot=s_dot_neg,
    s_min=0.4, s_max=1.5,
)
# With s=s_min and s_dot=-0.5: lower bound = -α·(-0.5) - α·(0.4-0.4) = α·0.5 = 1.0
check("scale lower bound enforced: a_s ≥ lower_bound",
      (u_qp_low[:, 2] >= 0.9).all().item(),
      f"got a_s={u_qp_low[:, 2].tolist()}")

# c) Scale upper bound: if s is at s_max and s_dot > 0
s_high = torch.full((N,), 1.5)  # at s_max
s_dot_pos = torch.full((N,), 0.5)
u_at_pos = torch.zeros(N, 3)
u_at_pos[:, 2] = 5.0  # big positive a_s
u_qp_high = solve_affine_qp(
    u_at=u_at_pos, s=s_high, s_dot=s_dot_pos,
    s_min=0.4, s_max=1.5,
)
check("scale upper bound enforced: a_s <= upper_bound",
      (u_qp_high[:, 2] <= -0.9).all().item(),
      f"got a_s={u_qp_high[:, 2].tolist()}")

# d) With obstacle
obs_c = torch.tensor([[[5.0, 5.0]]]).expand(N, 1, 2).clone()
obs_hs = torch.tensor([[[0.3, 0.3]]]).expand(N, 1, 2).clone()
pos = torch.tensor([[5.0 + 1.0, 5.0]]).expand(N, 2).clone()  # 1m from obstacle
vel = torch.zeros(N, 2)

u_qp_obs = solve_affine_qp(
    u_at=torch.zeros(N, 3),
    obs_centers=obs_c, obs_half_sizes=obs_hs,
    agent_pos=pos, agent_vel=vel,
    s=s_val, s_dot=s_dot_val,
    R_form=0.5, r_margin=0.2,
    s_min=0.4, s_max=1.5,
)
check("QP with obstacles returns finite", torch.isfinite(u_qp_obs).all().item())


# ═══════════════════════════════════════════════════════════════════════
# 5. Loss Function Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 5. Affine Loss Function")
print("=" * 60)

from gcbf_plus.algo.reward import compute_reward

pi = torch.randn(N, 3, requires_grad=True)
u_nom_l = torch.randn(N, 3)
u_qp_l = u_nom_l + torch.randn(N, 3) * 0.1
dist_red = torch.rand(N)
dist_goal = torch.rand(N) * 5.0

loss, info = compute_reward(
    pi_action=pi, u_nom=u_nom_l, u_qp=u_qp_l,
    dist_reduction=dist_red, dist_to_goal=dist_goal,
)
check("loss is scalar", loss.dim() == 0)
check("loss is finite", torch.isfinite(loss).item())
check("info has reward/total", "reward/total" in info)
check("info has reward/progress", "reward/progress" in info)
check("info has reward/qp", "reward/qp" in info)
check("info has reward/avoid", "reward/avoid" in info)

loss.backward()
check("backprop through loss succeeds", pi.grad is not None)
check("gradient is finite", torch.isfinite(pi.grad).all().item())


# ═══════════════════════════════════════════════════════════════════════
# 6. Mini Training Loop (3 steps)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 6. Mini Training Loop (3 steps)")
print("=" * 60)

from gcbf_plus.train_swarm import train
try:
    history = train(
        num_agents=2,
        area_size=8.0,
        n_obs=1,
        num_steps=3,
        n_env_train=4,
        batch_size=8,
        horizon=4,
        lr_actor=1e-3,
        log_interval=1,
        seed=0,
        checkpoint_path="/tmp/test_affine_ckpt.pt",
        device="cpu",
    )
    check("training loop completes", True)
    check("history has reward/total", len(history.get("reward/total", [])) > 0)
    losses = history.get("reward/total", [])
    check("loss values are finite", all(np.isfinite(l) for l in losses),
          f"got {losses}")
except Exception as e:
    check("training loop completes", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = n_passed + n_failed
print(f" Results: {n_passed}/{total} passed, {n_failed} failed")
print("=" * 60)

sys.exit(0 if n_failed == 0 else 1)
