"""
Graph Neural Network layer with attention aggregation.

Implements Equation 18 from the GCBF+ paper using pure PyTorch tensor ops
(no torch_geometric dependency).

    h_θ(z_i) = ψ_θ4( Σ_{j∈Ñ_i}  softmax(ψ_θ2(q_ij)) · ψ_θ3(q_ij) )

where   q_ij = ψ_θ1(z_ij)
        z_ij = cat(v_sender, v_receiver, e_ij)

Corrections vs. initial implementation
---------------------------------------
1. ψ₁ now receives the CONCATENATION of sender node features, receiver node
   features, and edge features — not edge features alone.
2. ψ₃ is a separate MLP (hidden 256, output 128), not an identity.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .mlp import MLP
from ..utils.graph import GraphsTuple


# ------------------------------------------------------------------
# Numerically-stable segment softmax  (replaces jraph.segment_softmax)
# ------------------------------------------------------------------

def _segment_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Compute softmax over *logits* grouped by *segment_ids*.

    Parameters
    ----------
    logits : (E,)  — raw attention scores.
    segment_ids : (E,)  — which segment (receiver node) each edge belongs to.
    num_segments : int  — total number of segments (nodes).

    Returns
    -------
    (E,)  — per-segment softmax weights.
    """
    # Numerical stability: subtract per-segment max
    max_vals = torch.full(
        (num_segments,), -1e9, device=logits.device, dtype=logits.dtype
    )
    max_vals.scatter_reduce_(
        0, segment_ids, logits, reduce="amax", include_self=True
    )
    logits = logits - max_vals[segment_ids]

    exp_logits = torch.exp(logits)

    # Sum per segment
    sum_exp = torch.zeros(num_segments, device=logits.device, dtype=logits.dtype)
    sum_exp.scatter_add_(0, segment_ids, exp_logits)

    # Normalise
    attn = exp_logits / (sum_exp[segment_ids] + 1e-12)
    return attn


# ------------------------------------------------------------------
# GNN Layer
# ------------------------------------------------------------------

class GNNLayer(nn.Module):
    """
    One GNN message-passing layer with attention aggregation (Eq. 18).

    Architecture (Table I)
    ----------------------
    ψ₁  Encoder   : [node_dim*2 + edge_dim → 256 → 256 → 128]
    ψ₂  Attention : [128 → 128 → 128 → 1]
    ψ₃  Value     : [128 → 256 → 128]
    ψ₄  Decoder   : [128 → 256 → 256 → out_dim]

    Parameters
    ----------
    node_dim : int
        Dimension of per-node features (e.g. 3 for one-hot type indicator).
    edge_dim : int
        Dimension of raw edge features (e.g. 4 for [Δpx, Δpy, Δvx, Δvy]).
    msg_hid_sizes : sequence of int
        Hidden sizes for the encoder ψ₁.
    msg_out_dim : int
        Output dimension of ψ₁ (feature space, default 128).
    attn_hid_sizes : sequence of int
        Hidden sizes for the attention ψ₂. Output is always 1.
    value_hid_sizes : sequence of int
        Hidden sizes for ψ₃.
    value_out_dim : int
        Output dimension of ψ₃ (default 128).
    update_hid_sizes : sequence of int
        Hidden sizes for the decoder ψ₄.
    out_dim : int
        Final output dimension of ψ₄.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        msg_hid_sizes: Sequence[int] = (256, 256),
        msg_out_dim: int = 128,
        attn_hid_sizes: Sequence[int] = (128, 128),
        value_hid_sizes: Sequence[int] = (256,),
        value_out_dim: int = 128,
        update_hid_sizes: Sequence[int] = (256, 256),
        out_dim: int = 1,
    ):
        super().__init__()

        # ψ₁  — encoder: cat(v_sender, v_receiver, e_ij) → q_ij
        psi1_in_dim = node_dim * 2 + edge_dim
        self.msg_net = MLP(
            in_dim=psi1_in_dim,
            hid_sizes=list(msg_hid_sizes),
            out_dim=msg_out_dim,
            act_final=True,   # ReLU after last layer (intermediate repr)
        )

        # ψ₂  — attention: q_ij → scalar logit
        self.attn_net = MLP(
            in_dim=msg_out_dim,
            hid_sizes=list(attn_hid_sizes),
            out_dim=1,
            act_final=False,
        )

        # ψ₃  — value transform: q_ij → value features
        self.value_net = MLP(
            in_dim=msg_out_dim,
            hid_sizes=list(value_hid_sizes),
            out_dim=value_out_dim,
            act_final=True,   # ReLU (intermediate repr fed into ψ₄)
        )

        # ψ₄  — decoder / update: aggregated features → output
        self.update_net = MLP(
            in_dim=value_out_dim,
            hid_sizes=list(update_hid_sizes),
            out_dim=out_dim,
            act_final=False,
        )

        # Store out_dim for zero-edge fallback
        self._out_dim = out_dim

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Parameters
        ----------
        graph : GraphsTuple
            Must have ``.nodes`` (N, node_dim), ``.edges`` (E, edge_dim),
            ``.senders``, ``.receivers``, ``.n_node``.

        Returns
        -------
        out : (n_node, out_dim) — per-node output features.
        """
        n_node = graph.n_node
        device = graph.nodes.device

        if graph.n_edge == 0:
            # No edges → output zeros
            return torch.zeros(n_node, self._out_dim, device=device)

        # ---- Gather sender / receiver node features ----
        v_senders = graph.nodes[graph.senders]      # (E, node_dim)
        v_receivers = graph.nodes[graph.receivers]   # (E, node_dim)

        # ---- 1. Build z_ij and encode (ψ₁) ----
        z_ij = torch.cat([v_senders, v_receivers, graph.edges], dim=-1)
        q = self.msg_net(z_ij)                       # (E, msg_out_dim)

        # ---- 2. Attention logits (ψ₂) ----
        w = self.attn_net(q).squeeze(-1)             # (E,)

        # ---- 3. Per-receiver softmax ----
        attn = _segment_softmax(w, graph.receivers, n_node)  # (E,)

        # ---- 4. Value transform (ψ₃) ----
        v = self.value_net(q)                        # (E, value_out_dim)

        # ---- 5. Weighted aggregation ----
        weighted = attn.unsqueeze(-1) * v            # (E, value_out_dim)
        aggr = torch.zeros(n_node, v.shape[-1], device=device)
        aggr.scatter_add_(
            0,
            graph.receivers.unsqueeze(-1).expand_as(weighted),
            weighted,
        )                                            # (n_node, value_out_dim)

        # ---- 6. Decode (ψ₄) ----
        out = self.update_net(aggr)                  # (n_node, out_dim)
        return out
