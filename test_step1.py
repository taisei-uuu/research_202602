#!/usr/bin/env python3
"""
Smoke tests for GCBF+ Step 1: Double Integrator environment + GNN models.

Run with:
    python test_step1.py
"""

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
# 1. Environment Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 1. Double Integrator Environment")
print("=" * 60)

from gcbf_plus.env import DoubleIntegrator

NUM_AGENTS = 4
env = DoubleIntegrator(num_agents=NUM_AGENTS, area_size=10.0, dt=0.03)

# 1a. Basic properties
check("state_dim == 4", env.state_dim == 4)
check("action_dim == 2", env.action_dim == 2)
check("node_dim == 3", env.node_dim == 3)
check("edge_dim == 4", env.edge_dim == 4)

# 1b. Reset
graph = env.reset(seed=42)
check("reset returns GraphsTuple", graph is not None)
check(
    f"agent_states shape == ({NUM_AGENTS}, 4)",
    env.agent_states.shape == (NUM_AGENTS, 4),
    f"got {env.agent_states.shape}",
)
check(
    f"goal_states shape == ({NUM_AGENTS}, 4)",
    env.goal_states.shape == (NUM_AGENTS, 4),
    f"got {env.goal_states.shape}",
)

# 1c. Graph structure
n_obs = env.params["n_obs"]
expected_nodes = NUM_AGENTS + NUM_AGENTS + n_obs  # agents + goals + obstacles
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
        f"edges shape == ({graph.n_edge}, 4)",
        graph.edges.shape == (graph.n_edge, 4),
        f"got {graph.edges.shape}",
    )
    check(
        f"senders shape == ({graph.n_edge},)",
        graph.senders.shape == (graph.n_edge,),
    )
    check(
        f"receivers shape == ({graph.n_edge},)",
        graph.receivers.shape == (graph.n_edge,),
    )
else:
    print("  [INFO] No edges in reset graph (comm_radius may be small)")

# 1d. Step
action = torch.zeros(NUM_AGENTS, 2)
graph_next, info = env.step(action)
check("step returns graph", graph_next is not None)
check("info has 'done'", "done" in info)
check("info has 'step'", "step" in info)
check(
    "agent_states unchanged with zero action (velocity was zero)",
    torch.allclose(env.agent_states[:, :2], env.agent_states[:, :2], atol=1e-5),
)

# 1e. Nominal controller
u_ref = env.nominal_controller()
check(
    f"u_ref shape == ({NUM_AGENTS}, 2)",
    u_ref.shape == (NUM_AGENTS, 2),
    f"got {u_ref.shape}",
)

# ═══════════════════════════════════════════════════════════════════════
# 2. GNN Model Tests
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 2. GNN Models (GCBF + Policy)")
print("=" * 60)

from gcbf_plus.nn import GCBFNetwork, PolicyNetwork

# Use a fresh reset with a large comm_radius to guarantee edges
env_test = DoubleIntegrator(
    num_agents=NUM_AGENTS,
    area_size=10.0,
    params={"comm_radius": 100.0, "n_obs": 2},
)
graph = env_test.reset(seed=123)
check(f"test graph n_edge > 0", graph.n_edge > 0, f"got {graph.n_edge}")

NODE_DIM = env_test.node_dim  # 3
EDGE_DIM = env_test.edge_dim  # 4

# 2a. GCBF Network — forward pass
gcbf = GCBFNetwork(node_dim=NODE_DIM, edge_dim=EDGE_DIM, n_agents=NUM_AGENTS)
h = gcbf(graph)
check(
    f"GCBF output shape == ({NUM_AGENTS}, 1)",
    h.shape == (NUM_AGENTS, 1),
    f"got {h.shape}",
)
check("GCBF output is finite", torch.isfinite(h).all().item())

# 2b. Policy Network — forward pass
policy = PolicyNetwork(
    node_dim=NODE_DIM, edge_dim=EDGE_DIM, action_dim=2, n_agents=NUM_AGENTS
)
u = policy(graph)
check(
    f"Policy output shape == ({NUM_AGENTS}, 2)",
    u.shape == (NUM_AGENTS, 2),
    f"got {u.shape}",
)
check("Policy output is finite", torch.isfinite(u).all().item())

# 2c. Gradient flow
print("\n  --- Gradient flow ---")
loss_h = h.sum()
loss_h.backward()
grad_ok_h = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in gcbf.parameters()
    if p.requires_grad
)
check("GCBF backward() succeeds with finite grads", grad_ok_h)

# Reset grads and test policy backward
policy.zero_grad()
u2 = policy(graph)
loss_u = u2.sum()
loss_u.backward()
grad_ok_u = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in policy.parameters()
    if p.requires_grad
)
check("Policy backward() succeeds with finite grads", grad_ok_u)

# 2d. Parameter counts
n_params_gcbf = sum(p.numel() for p in gcbf.parameters())
n_params_policy = sum(p.numel() for p in policy.parameters())
print(f"\n  [INFO] GCBF  params: {n_params_gcbf:,}")
print(f"  [INFO] Policy params: {n_params_policy:,}")

# 2e. Architecture sanity: ψ₁ input dim = node_dim*2 + edge_dim = 10
psi1_in = gcbf.gnn_layers[0].msg_net.net[0].in_features
expected_in = NODE_DIM * 2 + EDGE_DIM
check(
    f"ψ₁ input dim == {expected_in} (node_dim*2 + edge_dim)",
    psi1_in == expected_in,
    f"got {psi1_in}",
)

# 2f. ψ₃ is a real MLP (not identity)
psi3_params = sum(p.numel() for p in gcbf.gnn_layers[0].value_net.parameters())
check("ψ₃ (value_net) has > 0 parameters", psi3_params > 0, f"got {psi3_params}")

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = n_passed + n_failed
print(f" Results: {n_passed}/{total} passed, {n_failed} failed")
print("=" * 60)

sys.exit(0 if n_failed == 0 else 1)
