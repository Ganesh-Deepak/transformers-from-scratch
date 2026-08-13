"""
A modern small transformer — Chapters 6-9, 12, 13.

Architecture (matches the Chapter 22 capstone spec):
    RMSNorm (pre-norm)  ·  RoPE  ·  Grouped-Query Attention  ·  SwiGLU
    + KV cache for efficient generation

Shapes are annotated on every line. That is the point of this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ config


@dataclass
class Config:
    vocab_size: int = 8192
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6          # query heads
    n_kv_head: int = 2       # GQA: key/value heads (== n_head gives MHA, 1 gives MQA)
    d_head: int = 64
    d_ff: int = 1024
    max_T: int = 512
    rope_base: float = 10000.0
    tie_embed: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

    def n_params(self) -> int:
        """Analytic parameter count — Chapter 9. Must match the real model."""
        C, H, KV, D = self.d_model, self.n_head, self.n_kv_head, self.d_head
        embed = self.vocab_size * C
        attn = C * (H * D) + 2 * C * (KV * D) + (H * D) * C     # Wq, Wk, Wv, Wo
        mlp = 3 * C * self.d_ff                                  # SwiGLU
        norms = 2 * C                                            # two RMSNorms
        total = embed + self.n_layer * (attn + mlp + norms) + C  # + final norm
        if not self.tie_embed:
            total += self.vocab_size * C
        return total


# ============================================================ components


class RMSNorm(nn.Module):
    """Chapter 8. No mean subtraction, no bias."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (..., d)
        # Reduce in fp32 even when running in bf16 — this matters for stability.
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.g


def build_rope_cache(T: int, d_head: int, base: float = 10000.0, device=None):
    """
    Chapter 7. Returns cos, sin of shape (T, d_head).

    Uses the HuggingFace/Llama convention: the head dim is split in HALF
    (pairs are i and i+d/2), not into adjacent pairs. Must match rotate_half.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(T, device=device).float()               # (T,)
    freqs = torch.outer(t, inv_freq)                         # (T, d_head/2)
    emb = torch.cat((freqs, freqs), dim=-1)                  # (T, d_head)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, n_head, T, d_head);  cos, sin: (T, d_head)."""
    cos = cos[None, None]                                    # (1, 1, T, d_head)
    sin = sin[None, None]
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    """Chapters 6 + 13, with the Chapter 12 KV cache."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.nh, self.nkv, self.hd = cfg.n_head, cfg.n_kv_head, cfg.d_head
        self.rep = cfg.n_head // cfg.n_kv_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.d_head, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape                                    # (B, T, C)

        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)    # (B, nh,  T, hd)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)   # (B, nkv, T, hd)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)   # (B, nkv, T, hd)

        q, k = apply_rope(q, k, cos, sin)

        if cache is not None and cache[0] is not None:
            k = torch.cat([cache[0], k], dim=2)              # (B, nkv, T_past+T, hd)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        # Expand KV heads to match query heads. repeat_interleave, NOT repeat —
        # grouping must be contiguous: [k0,k0,k0, k1,k1,k1] for rep=3.
        kx = k.repeat_interleave(self.rep, dim=1)            # (B, nh, T_tot, hd)
        vx = v.repeat_interleave(self.rep, dim=1)

        # is_causal only when T > 1. During decode (T == 1) the single query is
        # the newest token and may attend to everything — passing is_causal=True
        # there would align the mask top-left and hide the entire prompt.
        y = F.scaled_dot_product_attention(
            q, kx, vx,
            is_causal=(T > 1),
            dropout_p=self.dropout if self.training else 0.0,
        )                                                    # (B, nh, T, hd)

        y = y.transpose(1, 2).reshape(B, T, self.nh * self.hd)   # (B, T, nh*hd)
        return self.o_proj(y), new_cache                     # (B, T, C)


class SwiGLU(nn.Module):
    """Chapter 8. Three matrices; d_ff is usually ~(8/3)*d_model."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):                                    # (B, T, C)
        return self.down(F.silu(self.gate(x)) * self.up(x))  # (B, T, C)


class Block(nn.Module):
    """Chapter 8. Pre-norm; the residual stream is never overwritten."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        a, new_cache = self.attn(self.n1(x), cos, sin, cache)
        x = x + a                                            # (B, T, C)
        x = x + self.mlp(self.n2(x))                         # (B, T, C)
        return x, new_cache


# ============================================================ model


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embed:
            self.head.weight = self.embed.weight             # Chapter 4

        cos, sin = build_rope_cache(cfg.max_T, cfg.d_head, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Chapter 9: scale residual-output projections so the stream's variance
        # does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None, caches=None, pos_offset=0):
        """
        ids:     (B, T) int64
        targets: (B, T) int64 or None
        caches:  list of per-layer (k, v) or None
        returns  logits (B, T, V), loss or None, new_caches
        """
        B, T = ids.shape
        assert pos_offset + T <= self.cfg.max_T, (
            f"sequence position {pos_offset + T} exceeds max_T={self.cfg.max_T}"
        )

        x = self.embed(ids)                                  # (B, T, C)
        cos = self.cos[pos_offset:pos_offset + T]            # (T, d_head)
        sin = self.sin[pos_offset:pos_offset + T]

        if caches is None:
            caches = [None] * self.cfg.n_layer
        new_caches = []
        for blk, cache in zip(self.blocks, caches):
            x, nc = blk(x, cos, sin, cache)                  # (B, T, C)
            new_caches.append(nc)

        x = self.norm(x)                                     # (B, T, C)
        logits = self.head(x)                                # (B, T, V)

        loss = None
        if targets is not None:
            # Chapter 9: shift. Predict token t+1 from tokens <= t.
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, new_caches

    # ------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=100, temperature=1.0, top_k=0,
                 top_p=1.0, min_p=0.0, use_cache=True, eos_id=None):
        """Chapters 11 + 12."""
        self.eval()
        caches = None
        pos = 0
        cur = ids

        for _ in range(max_new_tokens):
            if ids.size(1) >= self.cfg.max_T:
                break
            logits, _, caches = self.forward(cur, caches=caches, pos_offset=pos)
            if use_cache:
                pos = ids.size(1)
                # next iteration feeds ONLY the new token
            else:
                caches, pos = None, 0

            next_tok = sample(logits[:, -1, :], temperature, top_k, top_p, min_p)
            ids = torch.cat([ids, next_tok], dim=1)
            cur = next_tok if use_cache else ids

            if eos_id is not None and (next_tok == eos_id).all():
                break
        return ids


def sample(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    """
    Chapter 11. Order matters: temperature -> top-k -> top-p -> min-p.
    logits: (B, V)  ->  returns (B, 1)
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sl, si = logits.sort(descending=True, dim=-1)
        probs = sl.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        drop = (cum - probs) > top_p          # keep the token that crosses p
        drop_orig = drop.scatter(-1, si, drop)
        logits = logits.masked_fill(drop_orig, float("-inf"))

    probs = logits.softmax(dim=-1)

    if min_p > 0:
        probs = probs.masked_fill(probs < min_p * probs.amax(-1, keepdim=True), 0.0)
        probs = probs / probs.sum(-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)
