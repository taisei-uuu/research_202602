"""
Loss functions for GCBF+ training.

Implements:
    L_CBF  = c_safe · L_safe + c_unsafe · L_unsafe + c_hdot · L_hdot   (Eq. 19–21)
    L_ctrl = c_action · MSE(π(x), u_QP)                                 (Eq. 22)

Autograd Lie derivative:
    ḣ = (∂h/∂x) · ẋ
    where ẋ = f(x) + g(x)u  (continuous-time state derivative).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_lie_derivative(
    h_values: torch.Tensor,
    agent_states: torch.Tensor,
    x_dot: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ḣ = (∂h/∂x) · ẋ  using ``torch.autograd.grad``.

    Parameters
    ----------
    h_values : (n_agents,)
        CBF values — must be connected to *agent_states* in the computation
        graph (i.e. ``agent_states`` had ``requires_grad=True`` when h was
        computed).
    agent_states : (n_agents, state_dim)
        Agent states with gradients tracked.
    x_dot : (n_agents, state_dim)
        Continuous-time state derivative  ẋ = f(x) + g(x)u.

    Returns
    -------
    h_dot : (n_agents,)
        Time derivative of the CBF for each agent.
    dh_dx : (n_agents, state_dim)
        Gradient ∂h/∂x for each agent (useful for the QP solver).
    """
    # Compute ∂h/∂x  for every agent at once
    # h_values is (n,), agent_states is (n, 4)
    # We sum h_values to get a scalar, then grad gives (n, 4)
    dh_dx = torch.autograd.grad(
        outputs=h_values.sum(),
        inputs=agent_states,
        create_graph=True,       # allow second-order gradients for training
        retain_graph=True,
    )[0]  # (n_agents, state_dim)

    # ḣ = Σ_k  (∂h_i/∂x_k) · (ẋ_k)   per agent
    h_dot = (dh_dx * x_dot).sum(dim=-1)    # (n_agents,)

    return h_dot, dh_dx


def compute_loss(
    h: torch.Tensor,
    h_dot: torch.Tensor,
    pi_action: torch.Tensor,
    u_qp: torch.Tensor,
    safe_mask: torch.Tensor,
    unsafe_mask: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 0.02,
    coef_safe: float = 1.0,
    coef_unsafe: float = 1.0,
    coef_h_dot: float = 0.2,
    coef_action: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the total GCBF+ training loss.

    Parameters
    ----------
    h : (n_agents,)
        CBF values per agent.
    h_dot : (n_agents,)
        Time derivative of CBF per agent.
    pi_action : (n_agents, action_dim)
        Policy network output.
    u_qp : (n_agents, action_dim)
        QP-solved safe control target.
    safe_mask : (n_agents,) bool
        True where agent is in the safe region.
    unsafe_mask : (n_agents,) bool
        True where agent is in the unsafe region.
    alpha, eps : float
        CBF hyperparameters.
    coef_safe, coef_unsafe, coef_h_dot, coef_action : float
        Loss term coefficients.

    Returns
    -------
    total_loss : scalar tensor
    info : dict of scalar loss values for logging
    """
    # ---- L_safe  (Eq. 20a, safe region): h should be > 0 ----
    # Penalise  ReLU(-h + ε)  for safe agents
    h_safe = torch.where(safe_mask, h, torch.ones_like(h) * eps * 2)
    loss_safe = F.relu(-h_safe + eps)
    n_safe = safe_mask.sum().clamp(min=1)
    loss_safe = loss_safe.sum() / n_safe

    # ---- L_unsafe  (Eq. 20a, unsafe region): h should be < 0 ----
    # Penalise  ReLU(h + ε)  for unsafe agents
    h_unsafe = torch.where(unsafe_mask, h, -torch.ones_like(h) * eps * 2)
    loss_unsafe = F.relu(h_unsafe + eps)
    n_unsafe = unsafe_mask.sum().clamp(min=1)
    loss_unsafe = loss_unsafe.sum() / n_unsafe

    # ---- L_h_dot  (Eq. 21):  ḣ + α·h ≥ 0 ----
    # Penalise  ReLU(-ḣ - α·h + ε)
    loss_h_dot = F.relu(-h_dot - alpha * h + eps).mean()

    # ---- L_action  (Eq. 22):  MSE(π(x), u_QP) ----
    loss_action = F.mse_loss(pi_action, u_qp)

    # ---- Total (Eq. 19) ----
    total_loss = (
        coef_safe * loss_safe
        + coef_unsafe * loss_unsafe
        + coef_h_dot * loss_h_dot
        + coef_action * loss_action
    )

    # ---- Accuracy metrics ----
    with torch.no_grad():
        acc_safe = (h[safe_mask] > 0).float().mean() if safe_mask.any() else torch.tensor(0.0)
        acc_unsafe = (h[unsafe_mask] < 0).float().mean() if unsafe_mask.any() else torch.tensor(0.0)
        acc_h_dot = ((h_dot + alpha * h) > 0).float().mean()

    info = {
        "loss/total": total_loss.item(),
        "loss/safe": loss_safe.item(),
        "loss/unsafe": loss_unsafe.item(),
        "loss/h_dot": loss_h_dot.item(),
        "loss/action": loss_action.item(),
        "acc/safe": acc_safe.item(),
        "acc/unsafe": acc_unsafe.item(),
        "acc/h_dot": acc_h_dot.item(),
        "n_safe": int(n_safe.item()),
        "n_unsafe": int(n_unsafe.item()),
    }

    return total_loss, info
