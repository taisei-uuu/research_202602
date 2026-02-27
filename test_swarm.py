#!/usr/bin/env python3
"""
Smoke tests for GCBF+ Swarm environment (3-drone rigid body per node).

Run with:
    python test_swarm.py
"""

import math
import sys
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
# 1. SwarmIntegrator Environment Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 1. SwarmIntegrator Environment")
print("=" * 60)

from gcbf_plus.env import SwarmIntegrator

NUM_AGENTS = 4
env = SwarmIntegrator(num_agents=NUM_AGENTS, area_size=10.0, dt=0.03)

# 1a. Basic properties
check("state_dim == 6", env.state_dim == 6)
check("action_dim == 3", env.action_dim == 3)
check("node_dim == 3", env.node_dim == 3)
check("edge_dim == 8", env.edge_dim == 8)

# 1b. g_x_matrix shape
g_x = env.g_x_matrix
check(
    f"g_x_matrix shape == (6, 3)",
    g_x.shape == (6, 3),
    f"got {g_x.shape}",
)

# 1c. Reset
graph = env.reset(seed=42)
check("reset returns GraphsTuple", graph is not None)
check(
    f"agent_states shape == ({NUM_AGENTS}, 6)",
    env.agent_states.shape == (NUM_AGENTS, 6),
    f"got {env.agent_states.shape}",
)
check(
    f"goal_states shape == ({NUM_AGENTS}, 6)",
    env.goal_states.shape == (NUM_AGENTS, 6),
    f"got {env.goal_states.shape}",
)

# 1d. Graph structure
n_obs = env.params["n_obs"]
expected_nodes = NUM_AGENTS + NUM_AGENTS + n_obs
check(
    f"n_node == {expected_nodes}",
    graph.n_node == expected_nodes,
    f"got {graph.n_node}",
)
check(
    f"nodes shape == ({expected_nodes}, 3)",
    graph.nodes.shape == (expected_nodes, 3),
    f"got {graph.nodes.shape}",
)
check("n_edge >= 0", graph.n_edge >= 0)
if graph.n_edge > 0:
    check(
        f"edges dim == 8",
        graph.edges.shape[1] == 8,
        f"got {graph.edges.shape[1]}",
    )

# 1e. Step
action = torch.zeros(NUM_AGENTS, 3)
graph_next, info = env.step(action)
check("step returns graph", graph_next is not None)
check("info has 'done'", "done" in info)
check("info has 'step'", "step" in info)

# 1f. Nominal controller
u_ref = env.nominal_controller()
check(
    f"u_ref shape == ({NUM_AGENTS}, 3)",
    u_ref.shape == (NUM_AGENTS, 3),
    f"got {u_ref.shape}",
)


# ═══════════════════════════════════════════════════════════════════════
# 2. Swarm Geometry Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 2. Swarm Geometry (Drone Positions)")
print("=" * 60)

env2 = SwarmIntegrator(num_agents=2, area_size=10.0, params={"n_obs": 0})
env2.reset(seed=100)

# 2a. Drone positions shape
drone_pos = env2.get_drone_positions()
check(
    f"drone_positions shape == (2, 3, 2)",
    drone_pos.shape == (2, 3, 2),
    f"got {drone_pos.shape}",
)

# 2b. Rotation correctness: θ=0 → drones at known offsets
R_form = env2.params["R_form"]
test_state = torch.tensor(
    [[5.0, 5.0, 0.0, 0.0, 0.0, 0.0]],  # CoM at (5,5), θ=0
    dtype=torch.float32,
)
drone_pos_test = env2.get_drone_positions(test_state)
expected_d1 = torch.tensor([5.0 + R_form, 5.0])
expected_d2 = torch.tensor([5.0 - R_form / 2, 5.0 + R_form * math.sqrt(3) / 2])
expected_d3 = torch.tensor([5.0 - R_form / 2, 5.0 - R_form * math.sqrt(3) / 2])

check(
    "drone 1 at θ=0 correct",
    torch.allclose(drone_pos_test[0, 0], expected_d1, atol=1e-5),
    f"got {drone_pos_test[0, 0].tolist()} expected {expected_d1.tolist()}",
)
check(
    "drone 2 at θ=0 correct",
    torch.allclose(drone_pos_test[0, 1], expected_d2, atol=1e-5),
    f"got {drone_pos_test[0, 1].tolist()} expected {expected_d2.tolist()}",
)
check(
    "drone 3 at θ=0 correct",
    torch.allclose(drone_pos_test[0, 2], expected_d3, atol=1e-5),
    f"got {drone_pos_test[0, 2].tolist()} expected {expected_d3.tolist()}",
)

# 2c. Rotation by π/2: d1 should rotate 90° CCW
test_state_rot = torch.tensor(
    [[5.0, 5.0, 0.0, 0.0, math.pi / 2, 0.0]],
    dtype=torch.float32,
)
drone_pos_rot = env2.get_drone_positions(test_state_rot)
expected_d1_rot = torch.tensor([5.0, 5.0 + R_form])  # rotated 90° CCW
check(
    "drone 1 at θ=π/2 correct",
    torch.allclose(drone_pos_rot[0, 0], expected_d1_rot, atol=1e-5),
    f"got {drone_pos_rot[0, 0].tolist()} expected {expected_d1_rot.tolist()}",
)


# ═══════════════════════════════════════════════════════════════════════
# 3. Safety Mask Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 3. Safety Masks (Drone-to-Drone Collision)")
print("=" * 60)

env3 = SwarmIntegrator(
    num_agents=2, area_size=10.0,
    params={"n_obs": 0, "drone_radius": 0.05, "R_form": 0.3},
)
env3.reset(seed=200)

# 3a. Far apart → safe
env3._agent_states = torch.tensor([
    [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [8.0, 8.0, 0.0, 0.0, 0.0, 0.0],
], dtype=torch.float32)
check("far apart → both safe", env3.safe_mask().all().item())

# 3b. Very close → unsafe (drone overlap)
env3._agent_states = torch.tensor([
    [5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    [5.0 + 0.01, 5.0, 0.0, 0.0, 0.0, 0.0],  # CoMs nearly overlap
], dtype=torch.float32)
unsafe = env3.unsafe_mask()
check("overlapping → both unsafe", unsafe.all().item())


# ═══════════════════════════════════════════════════════════════════════
# 4. Differentiable Dynamics Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 4. Differentiable Dynamics")
print("=" * 60)

env4 = SwarmIntegrator(num_agents=3, area_size=10.0, params={"n_obs": 0})
env4.reset(seed=300)

# 4a. forward_step output shape
states = env4.agent_states.clone().requires_grad_(True)
action = torch.randn(3, 3) * 0.1
next_states = env4.forward_step(states, action)
check(
    f"forward_step output shape == (3, 6)",
    next_states.shape == (3, 6),
    f"got {next_states.shape}",
)

# 4b. Autograd through forward_step
loss_test = next_states.sum()
grad = torch.autograd.grad(loss_test, states, retain_graph=True)[0]
check(
    "forward_step gradient exists",
    grad is not None and grad.shape == (3, 6),
    f"grad is None or wrong shape",
)
check("forward_step gradient is finite", torch.isfinite(grad).all().item())

# 4c. state_dot output shape
x_dot = env4.state_dot(states, action)
check(
    f"state_dot output shape == (3, 6)",
    x_dot.shape == (3, 6),
    f"got {x_dot.shape}",
)


# ═══════════════════════════════════════════════════════════════════════
# 5. GNN Forward Pass with Swarm Graph
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 5. GNN Models with Swarm Graph")
print("=" * 60)

from gcbf_plus.nn import GCBFNetwork, PolicyNetwork

env5 = SwarmIntegrator(
    num_agents=NUM_AGENTS, area_size=10.0,
    params={"comm_radius": 100.0, "n_obs": 2},
)
graph5 = env5.reset(seed=500)
check(f"swarm graph n_edge > 0", graph5.n_edge > 0, f"got {graph5.n_edge}")

NODE_DIM = env5.node_dim   # 3
EDGE_DIM = env5.edge_dim   # 8

# 5a. GCBF Network forward
gcbf = GCBFNetwork(node_dim=NODE_DIM, edge_dim=EDGE_DIM, n_agents=NUM_AGENTS)
h = gcbf(graph5)
check(
    f"GCBF output shape == ({NUM_AGENTS}, 1)",
    h.shape == (NUM_AGENTS, 1),
    f"got {h.shape}",
)
check("GCBF output is finite", torch.isfinite(h).all().item())

# 5b. Policy Network forward
policy = PolicyNetwork(
    node_dim=NODE_DIM, edge_dim=EDGE_DIM, action_dim=3, n_agents=NUM_AGENTS
)
u = policy(graph5)
check(
    f"Policy output shape == ({NUM_AGENTS}, 3)",
    u.shape == (NUM_AGENTS, 3),
    f"got {u.shape}",
)
check("Policy output is finite", torch.isfinite(u).all().item())

# 5c. Gradient flow
loss_h = h.sum()
loss_h.backward()
grad_ok = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in gcbf.parameters()
    if p.requires_grad
)
check("GCBF backward() succeeds with finite grads", grad_ok)

# 5d. ψ₁ input dim = node_dim*2 + edge_dim = 3*2 + 8 = 14
psi1_in = gcbf.gnn_layers[0].msg_net.net[0].in_features
expected_in = NODE_DIM * 2 + EDGE_DIM
check(
    f"ψ₁ input dim == {expected_in} (node_dim*2 + edge_dim)",
    psi1_in == expected_in,
    f"got {psi1_in}",
)


# ═══════════════════════════════════════════════════════════════════════
# 6. Lie Derivative through Swarm Graph (autograd test)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 6. Lie Derivative (autograd through swarm graph)")
print("=" * 60)

env6 = SwarmIntegrator(
    num_agents=3, area_size=10.0,
    params={"comm_radius": 100.0, "n_obs": 0},
)
env6.reset(seed=600)

states6 = env6.agent_states.clone().requires_grad_(True)
graph6 = env6.build_graph_differentiable(states6)
gcbf6 = GCBFNetwork(node_dim=3, edge_dim=8, n_agents=3)
h6 = gcbf6.gnn_layers[0](graph6)

# Extract agent outputs
h_agents6 = h6[:3].squeeze(-1)  # (3,)

# Compute dh/dx
dh_dx = torch.autograd.grad(
    outputs=h_agents6.sum(),
    inputs=states6,
    create_graph=True,
    retain_graph=True,
)[0]
check(
    f"dh/dx shape == (3, 6)",
    dh_dx.shape == (3, 6),
    f"got {dh_dx.shape}",
)
check("dh/dx is finite", torch.isfinite(dh_dx).all().item())

# Compute h_dot = (dh/dx) · x_dot
action6 = torch.randn(3, 3) * 0.01
x_dot6 = env6.state_dot(states6, action6)
h_dot6 = (dh_dx * x_dot6).sum(dim=-1)
check(
    f"h_dot shape == (3,)",
    h_dot6.shape == (3,),
    f"got {h_dot6.shape}",
)
check("h_dot is finite", torch.isfinite(h_dot6).all().item())


# ═══════════════════════════════════════════════════════════════════════
# 7. Edge Feature Content Sanity
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 7. Edge Feature Sanity")
print("=" * 60)

env7 = SwarmIntegrator(
    num_agents=2, area_size=10.0,
    params={"comm_radius": 100.0, "n_obs": 0},
)
env7.reset(seed=700)
graph7 = env7._get_graph()

if graph7.n_edge > 0:
    check("edge features dim == 8", graph7.edges.shape[1] == 8)
    # min_dist (column 5) should be positive
    min_dists = graph7.edges[:, 5]
    check("min_dist values are positive", (min_dists > 0).all().item())
    check("edge features are finite", torch.isfinite(graph7.edges).all().item())
else:
    print("  [INFO] No edges — skipping edge feature sanity checks")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = n_passed + n_failed
print(f" Results: {n_passed}/{total} passed, {n_failed} failed")
print("=" * 60)

sys.exit(0 if n_failed == 0 else 1)
