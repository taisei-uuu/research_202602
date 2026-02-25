"""
Graph Neural Network layer with attention aggregation.

Implements Equation 18 from the GCBF+ paper using pure PyTorch tensor ops
(no torch_geometric dependency).

    h_θ(z_i) = ψ_θ4( Σ_{j∈Ñ_i}  softmax(ψ_θ2(q_ij)) · ψ_θ3(q_ij) )

where   q_ij = ψ_θ1(z_ij)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP
from ..utils.graph import GraphsTuple


def _segment_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Compute softmax over *logits* grouped by *segment_ids*.

    Parameters
    ----------
    logits : (E,) — raw attention scores.
    segment_ids : (E,) — which segment (receiver node) each edge belongs to.
    num_segments : int — total number of segments (nodes).

    Returns
    -------
    (E,) — per-segment softmax weights.
    """
    # Numerical stability: subtract segment max
    max_vals = torch.full((num_segments,), -1e9, device=logits.device, dtype=logits.dtype)
    max_vals.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    logits = logits - max_vals[segment_ids]

    exp_logits = torch.exp(logits)

    # Sum per segment
    sum_exp = torch.zeros(num_segments, device=logits.device, dtype=logits.dtype)
    sum_exp.scatter_add_(0, segment_ids, exp_logits)

    # Normalise
    attn = exp_logits / (sum_exp[segment_ids] + 1e-12)
    return attn


class GNNLayer(nn.Module):
    """
    One GNN message-passing layer with attention aggregation (Eq. 18).

    Parameters
    ----------
    edge_dim : int
        Dimension of raw edge features z_ij.
    msg_hid_sizes : sequence of int
        Hidden sizes for the encoder (ψ₁) MLP.
    msg_out_dim : int
        Output dim of ψ₁ (the feature space dimension, e.g. 128).
    attn_hid_sizes : sequence of int
        Hidden sizes for the attention (ψ₂) MLP.
        The final output is always 1 (scalar attention logit).
    update_hid_sizes : sequence of int
        Hidden sizes for the decoder (ψ₄) MLP.
    out_dim : int
        Final output dimension of ψ₄.
    """

    def __init__(
        self,
        edge_dim: int,
        msg_hid_sizes: Sequence[int] = (256, 256),
        msg_out_dim: int = 128,
        attn_hid_sizes: Sequence[int] = (128, 128),
        update_hid_sizes: Sequence[int] = (256, 256),
        out_dim: int = 1,
    ):
        super().__init__()

        # ψ₁  — encoder: edge features → message features
        self.msg_net = MLP(
            in_dim=edge_dim,
            hid_sizes=list(msg_hid_sizes),
            out_dim=msg_out_dim,
            act_final=True,          # ReLU after last layer (intermediate repr)
        )

        # ψ₂  — attention: message features → scalar logit
        self.attn_net = MLP(
            in_dim=msg_out_dim,
            hid_sizes=list(attn_hid_sizes),
            out_dim=1,
            act_final=False,
        )

        # ψ₄  — decoder / update: aggregated features → output
        self.update_net = MLP(
            in_dim=msg_out_dim,
            hid_sizes=list(update_hid_sizes),
            out_dim=out_dim,
            act_final=False,
        )

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Parameters
        ----------
        graph : GraphsTuple
            Must have .edges (E, edge_dim), .senders, .receivers, .n_node.

        Returns
        -------
        out : (n_node, out_dim) — per-node output features.
        """
        n_node = graph.n_node
        device = graph.edges.device

        if graph.n_edge == 0:
            # No edges → all zeros
            return torch.zeros(n_node, self.update_net.net[-1].out_features
                               if not isinstance(self.update_net.net[-1], nn.ReLU)
                               else self.update_net.net[-2].out_features,
                               device=device)

        # ---------- 1. Encode edge features  (ψ₁) ----------
        q = self.msg_net(graph.edges)          # (E, msg_out_dim)

        # ---------- 2. Attention logits      (ψ₂) ----------
        w = self.attn_net(q).squeeze(-1)       # (E,)

        # ---------- 3. Segment softmax over receivers ----------
        attn = _segment_softmax(w, graph.receivers, n_node)   # (E,)

        # ---------- 4. Weighted aggregation ----------
        # ψ₃  is identity (reuse q)  →  aggregate = Σ attn_j · q_j
        weighted = attn.unsqueeze(-1) * q      # (E, msg_out_dim)

        aggr = torch.zeros(n_node, q.shape[-1], device=device)
        aggr.scatter_add_(
            0,
            graph.receivers.unsqueeze(-1).expand_as(weighted),
            weighted,
        )                                       # (n_node, msg_out_dim)

        # ---------- 5. Decode / update       (ψ₄) ----------
        out = self.update_net(aggr)             # (n_node, out_dim)
        return out
