"""
Mixture of Experts — Chapter 20.

Everything the chapter claims about MoE is computed from this file and checked
in tests/test_tfs.py. Shapes are annotated on every line:

    N  tokens in the flattened batch (B*T)        C  d_model
    E  routed experts                             k  experts chosen per token
    F  one expert's hidden width (d_ff / granularity)

Three routers, because real models use three different ones:
    softmax_topk   Switch, GShard, DeepSeek-V2   softmax over ALL experts, keep the top k
    topk_softmax   Mixtral, gpt-oss              top k of the LOGITS, softmax over those k
    sigmoid_bias   DeepSeek-V3, Kimi K2          independent sigmoid scores; a balancing
                                                 bias decides WHO is picked, never HOW MUCH
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ one expert


class Expert(nn.Module):
    """A SwiGLU MLP (Chapter 8), just narrower: F = d_ff of ONE expert."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)     # (C -> F)
        self.up = nn.Linear(d_model, d_ff, bias=False)       # (C -> F)
        self.down = nn.Linear(d_ff, d_model, bias=False)     # (F -> C)

    def forward(self, x):                                    # (n, C)
        return self.down(F.silu(self.gate(x)) * self.up(x))  # (n, C)


# ============================================================ routers


def route_softmax_topk(logits: torch.Tensor, k: int, renorm: bool = True):
    """Switch / GShard style. logits (N, E) -> weights (N, k), idx (N, k), probs (N, E)."""
    probs = logits.softmax(-1)                               # (N, E)  over ALL experts
    w, idx = probs.topk(k, dim=-1)                           # (N, k), (N, k)
    if renorm:
        w = w / w.sum(-1, keepdim=True)                      # (N, k)  sums to 1
    return w, idx, probs


def route_topk_softmax(logits: torch.Tensor, k: int):
    """Mixtral / gpt-oss style: choose on logits, softmax over the chosen k only."""
    top, idx = logits.topk(k, dim=-1)                        # (N, k), (N, k)
    w = top.softmax(-1)                                      # (N, k)
    return w, idx, logits.softmax(-1)                        # probs (N, E) for the aux loss


def route_sigmoid_bias(logits: torch.Tensor, k: int, bias: torch.Tensor):
    """DeepSeek-V3. Affinity s = sigmoid(logit), one per expert, independent.
    The balancing bias is added ONLY to choose the experts; the gate weights
    are the unbiased affinities of the chosen ones, normalised to sum to 1."""
    s = logits.sigmoid()                                     # (N, E)  each in (0, 1)
    idx = (s + bias).topk(k, dim=-1).indices                 # (N, k)  bias decides WHO
    w = s.gather(-1, idx)                                    # (N, k)  but not HOW MUCH
    w = w / w.sum(-1, keepdim=True)                          # (N, k)
    return w, idx, s / s.sum(-1, keepdim=True)               # probs (N, E), for reporting


# ============================================================ balancing


def expert_load(idx: torch.Tensor, n_expert: int) -> torch.Tensor:
    """Tokens routed to each expert. idx (N, k) -> counts (E,)."""
    return torch.bincount(idx.reshape(-1), minlength=n_expert)   # (E,)


def load_balance_loss(probs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Switch Transformer auxiliary loss, E * sum_i f_i * P_i.

    f_i: fraction of routed slots that went to expert i (counts, NOT differentiable)
    P_i: mean router probability of expert i (differentiable: this carries the gradient)
    = 1 exactly when routing is uniform; = E when everything goes to one expert
    with probability 1. Multiply by alpha (~1e-2) before adding to the LM loss."""
    N, E = probs.shape
    k = idx.shape[-1]
    f = expert_load(idx, E).float() / (N * k)                # (E,)  sums to 1
    P = probs.mean(0)                                        # (E,)  sums to 1
    return E * (f * P).sum()


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE (Zoph et al. 2022): penalise large router logits, which overflow
    in low precision and make routing brittle. mean( logsumexp(logits)^2 )."""
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


@torch.no_grad()
def update_bias(bias: torch.Tensor, idx: torch.Tensor, gamma: float = 1e-3) -> torch.Tensor:
    """DeepSeek-V3's auxiliary-loss-free rule: after each step, raise the bias of
    under-loaded experts and lower it for over-loaded ones by a fixed step.
    No gradient touches the LM loss."""
    load = expert_load(idx, bias.numel()).float()            # (E,)
    bias += gamma * torch.sign(load.mean() - load)           # under-used -> up
    return bias


def load_cv(idx: torch.Tensor, n_expert: int) -> float:
    """Coefficient of variation of expert load: 0 = perfectly balanced."""
    load = expert_load(idx, n_expert).float()
    return (load.std(unbiased=False) / load.mean()).item()


# ============================================================ the layer


class MoE(nn.Module):
    """A drop-in replacement for the MLP of a block (Chapter 8).

    gating: "softmax" (softmax then top-k, renormalised), "topk_softmax"
            (Mixtral), or "sigmoid" (DeepSeek-V3, with a balancing bias).
    n_shared: experts every token passes through (DeepSeekMoE)."""

    def __init__(self, d_model: int, d_ff: int, n_expert: int, top_k: int,
                 n_shared: int = 0, gating: str = "softmax"):
        super().__init__()
        assert gating in ("softmax", "topk_softmax", "sigmoid")
        self.E, self.k, self.gating = n_expert, top_k, gating
        self.router = nn.Linear(d_model, n_expert, bias=False)        # (C -> E)  tiny
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_expert))
        self.shared = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_shared))
        self.register_buffer("bias", torch.zeros(n_expert))           # sigmoid gating only

    def route(self, x):                                      # x (N, C)
        logits = self.router(x).float()                      # (N, E)  fp32: see 20.8
        if self.gating == "softmax":
            w, idx, probs = route_softmax_topk(logits, self.k)
        elif self.gating == "topk_softmax":
            w, idx, probs = route_topk_softmax(logits, self.k)
        else:
            w, idx, probs = route_sigmoid_bias(logits, self.k, self.bias)
        return logits, w.to(x.dtype), idx, probs             # (N,E) (N,k) (N,k) (N,E)

    def forward_loop(self, x):
        """The readable version: for each expert, gather the tokens that chose it."""
        shape = x.shape
        x = x.reshape(-1, shape[-1])                         # (N, C)  routing is PER TOKEN
        logits, w, idx, probs = self.route(x)
        out = torch.zeros_like(x)                            # (N, C)
        for e, expert in enumerate(self.experts):
            hit = idx == e                                   # (N, k)  bool
            rows = hit.any(-1)                               # (N,)    tokens that chose e
            if not rows.any():
                continue
            gate = (w * hit).sum(-1)[rows].unsqueeze(-1)     # (n_e, 1)
            out[rows] += gate * expert(x[rows])              # (n_e, C)
        for s in self.shared:
            out += s(x)                                      # (N, C)  every token
        return out.reshape(shape), (logits, idx, probs)

    def forward(self, x):
        """The production shape: SORT the N*k (token, expert) slots by expert, so
        each expert sees one contiguous block. That block-per-expert layout is
        exactly what a grouped GEMM kernel (MegaBlocks, grouped_gemm, DeepEP)
        consumes; here a Python loop over the blocks stands in for the kernel."""
        shape = x.shape
        x = x.reshape(-1, shape[-1])                         # (N, C)
        N, C = x.shape
        logits, w, idx, probs = self.route(x)
        flat_e = idx.reshape(-1)                             # (N*k,)  expert of each slot
        order = torch.argsort(flat_e, stable=True)           # (N*k,)  slots grouped by expert
        token = order // self.k                              # (N*k,)  which token each slot is
        counts = torch.bincount(flat_e, minlength=self.E).tolist()   # (E,)  block sizes
        xs = x[token]                                        # (N*k, C)  "dispatch"
        ys = torch.cat([self.experts[e](blk) if blk.shape[0] else blk
                        for e, blk in enumerate(torch.split(xs, counts))])   # (N*k, C)
        ys = ys * w.reshape(-1)[order].unsqueeze(-1)         # (N*k, C)  gate weights
        out = torch.zeros_like(x).index_add_(0, token, ys)   # (N, C)  "combine"
        for s in self.shared:
            out = out + s(x)                                 # (N, C)
        return out.reshape(shape), (logits, idx, probs)


# ============================================================ arithmetic


def mlp_params(d_model: int, d_ff: int) -> int:
    """SwiGLU: three d_model x d_ff matrices."""
    return 3 * d_model * d_ff


def moe_params(d_model: int, d_ff_expert: int, n_routed: int, top_k: int, n_shared: int = 0):
    """(total, active) MLP parameters of ONE MoE layer (router included)."""
    per = mlp_params(d_model, d_ff_expert)
    router = d_model * n_routed
    return (n_routed + n_shared) * per + router, (top_k + n_shared) * per + router


def expected_distinct_experts(n_expert: int, top_k: int, n_tokens: int) -> float:
    """How many different experts a batch of tokens touches, if each token's k
    experts were a uniform random draw. Each expert is missed by one token with
    probability (1 - k/E), so by all of them with (1 - k/E)^n.
    This is the number that decides MoE DECODE cost: every touched expert's
    weights must be read from memory once per step."""
    return n_expert * (1.0 - (1.0 - top_k / n_expert) ** n_tokens)


def expert_capacity(n_tokens: int, n_expert: int, top_k: int, capacity_factor: float) -> int:
    """Fixed per-expert buffer in capacity-based (token-dropping) MoE."""
    return math.ceil(capacity_factor * n_tokens * top_k / n_expert)


def all_to_all_bytes(n_tokens: int, top_k: int, d_model: int, bytes_per: int) -> int:
    """Expert parallelism moves every routed slot to its expert's device and back:
    dispatch + combine. Upper bound (assumes no slot stays on its own device)."""
    return 2 * n_tokens * top_k * d_model * bytes_per
