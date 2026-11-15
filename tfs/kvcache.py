"""
Chapters 15 and 16 reference code: the KV cache beyond its size formula.

    units                       kv_bytes, GIB, users_that_fit
    chunked prefill, exactly    forward_offset, prefill_chunked
    timing the cache            time_generation
    eviction                    streaming_keep, h2o_keep, snapkv_keep, mass_kept
    sharing across layers       cla_kv_bytes, yoco_kv_bytes

tfs.model.Model.forward is correct for training, for one-shot prefill and for
one-token decode. It is NOT correct for a multi-token chunk on top of a cache:
it passes is_causal=True, and PyTorch aligns that mask to the top-left, so the
chunk's first query is treated as position 0 and cannot see the cache.
forward_offset runs the same weights with the mask aligned bottom-right, which
is what chunked prefill needs. tests/test_inference.py checks both facts.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from tfs.model import apply_rope

GIB = 1024 ** 3
KIB = 1024


def kv_bytes(n_layer, n_kv_head, d_head, seq=1, batch=1, dtype_bytes=2):
    """2 (K and V) x layers x kv heads x head dim x tokens x sequences x bytes."""
    return 2 * n_layer * n_kv_head * d_head * seq * batch * dtype_bytes


def users_that_fit(budget_bytes, per_user_bytes):
    return int(budget_bytes // per_user_bytes)


# ============================================================ chunked prefill


def _attn_offset(attn, x, cos, sin, cache):
    """GroupedQueryAttention.forward, with a bottom-right causal mask."""
    B, T, _ = x.shape
    q = attn.q_proj(x).view(B, T, attn.nh, attn.hd).transpose(1, 2)    # (B, nh,  T, hd)
    k = attn.k_proj(x).view(B, T, attn.nkv, attn.hd).transpose(1, 2)   # (B, nkv, T, hd)
    v = attn.v_proj(x).view(B, T, attn.nkv, attn.hd).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)                            # (B, nkv, P+T, hd)
        v = torch.cat([cache[1], v], dim=2)
    new_cache = (k, v)
    P = k.size(2) - T                                                  # tokens already cached
    # query i sits at absolute position P+i and may see keys 0 .. P+i
    qpos = torch.arange(P, P + T)[:, None]                             # (T, 1)
    kpos = torch.arange(P + T)[None, :]                                # (1, P+T)
    mask = kpos <= qpos                                                # (T, P+T)  bool
    kx = k.repeat_interleave(attn.rep, dim=1)
    vx = v.repeat_interleave(attn.rep, dim=1)
    y = F.scaled_dot_product_attention(q, kx, vx, attn_mask=mask)      # (B, nh, T, hd)
    y = y.transpose(1, 2).reshape(B, T, attn.nh * attn.hd)
    return attn.o_proj(y), new_cache


@torch.no_grad()
def forward_offset(model, ids, caches=None, pos=0):
    """Model.forward for a chunk of T >= 1 tokens appended to an existing cache."""
    B, T = ids.shape
    x = model.embed(ids)                                               # (B, T, C)
    cos, sin = model.cos[pos:pos + T], model.sin[pos:pos + T]
    caches = caches or [None] * len(model.blocks)
    out = []
    for blk, c in zip(model.blocks, caches):
        a, nc = _attn_offset(blk.attn, blk.n1(x), cos, sin, c)
        x = x + a
        x = x + blk.mlp(blk.n2(x))
        out.append(nc)
    return model.head(model.norm(x)), out                              # (B, T, V)


@torch.no_grad()
def prefill_chunked(model, ids, chunk):
    """Process a prompt `chunk` tokens at a time. Returns last-chunk logits and the cache."""
    caches, logits = None, None
    for s in range(0, ids.size(1), chunk):
        logits, caches = forward_offset(model, ids[:, s:s + chunk], caches, pos=s)
    return logits, caches


# ============================================================ timing


@torch.no_grad()
def time_generation(model, prompt_len=16, n_new=200, use_cache=True, seed=0, repeats=3):
    """
    Seconds per generated token at each position, greedy, best of `repeats`
    runs (the minimum is the least noisy estimate of the true cost).
    """
    torch.manual_seed(seed)
    model.eval()
    ids0 = torch.randint(0, model.cfg.vocab_size, (1, prompt_len))
    best = None
    for _ in range(repeats):
        ids, caches, cur, pos, ts = ids0, None, ids0, 0, []
        for _ in range(n_new):
            t0 = time.perf_counter()
            if use_cache:
                logits, _, caches = model(cur, caches=caches, pos_offset=pos)
                pos = ids.size(1)
            else:
                logits, _, _ = model(ids)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ts.append(time.perf_counter() - t0)
            ids = torch.cat([ids, nxt], 1)
            cur = nxt
        best = ts if best is None else [min(a, b) for a, b in zip(best, ts)]
    return best


# ============================================================ eviction


def streaming_keep(T, n_sink=4, window=1020):
    """StreamingLLM: keep the first n_sink tokens (attention sinks) and the last `window`."""
    if T <= n_sink + window:
        return list(range(T))
    return list(range(n_sink)) + list(range(T - window, T))


def h2o_keep(acc_attention, budget, n_recent):
    """
    H2O: keep the n_recent newest tokens plus the 'heavy hitters' -- the older
    tokens with the largest attention accumulated so far.  acc_attention: (T,)
    """
    T = acc_attention.numel()
    if T <= budget:
        return list(range(T))
    recent = list(range(T - n_recent, T))
    old = acc_attention[:T - n_recent]
    heavy = old.topk(budget - n_recent).indices.sort().values.tolist()
    return heavy + recent


def snapkv_keep(q_window, k, budget, kernel=7):
    """
    SnapKV: before decoding starts, let the last w prompt queries vote on which
    earlier positions matter, smooth the votes with a max-pool (keep whole
    neighbourhoods, not single tokens), keep the top `budget` plus the window.
    q_window: (w, d)  k: (T, d) -- one head; real use repeats this per head.
    """
    w, d = q_window.shape
    T = k.size(0)
    prefix = T - w
    if prefix <= budget:
        return list(range(T))
    att = (q_window @ k[:prefix].T / d ** 0.5).softmax(-1)          # (w, prefix)
    votes = att.sum(0)                                               # (prefix,)
    votes = F.max_pool1d(votes[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]
    chosen = votes.topk(budget).indices.sort().values.tolist()
    return chosen + list(range(prefix, T))


def mass_kept(probs, keep):
    """Fraction of one query's attention mass that lands on the kept positions."""
    return probs[keep].sum().item() / probs.sum().item()


# ============================================================ cross-layer sharing


def cla_kv_bytes(n_layer, n_kv, d_head, share=2, dtype_bytes=2):
    """Cross-Layer Attention: every `share` adjacent layers use one K/V. Bytes per token."""
    return kv_bytes(-(-n_layer // share), n_kv, d_head, dtype_bytes=dtype_bytes)


def yoco_kv_bytes(n_layer, n_kv, d_head, T, window, dtype_bytes=2):
    """
    YOCO: the first half of the layers (self-decoder) use a sliding window and
    cache only `window` tokens each; they produce ONE global K/V that every
    layer of the second half (cross-decoder) reads. Total bytes at context T.
    """
    half = n_layer // 2
    self_dec = kv_bytes(half, n_kv, d_head, seq=min(T, window), dtype_bytes=dtype_bytes)
    global_kv = kv_bytes(1, n_kv, d_head, seq=T, dtype_bytes=dtype_bytes)
    return self_dec + global_kv


def synthetic_attention(T=1024, n_heavy=24, seed=0):
    """
    A stand-in for one head's attention pattern with the structure the
    eviction papers measured: a sink on token 0, a few 'heavy hitter' tokens
    scattered through the context, and a recency bias. Returns (accumulated
    attention over the prefix, the newest query's attention), each (T,).
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(T, generator=g) * 0.5
    heavy = torch.randperm(T - 64, generator=g)[:n_heavy] + 4
    base[heavy] += 4.0
    base[0] += 6.0                                                   # the sink
    recency = torch.linspace(-2.0, 2.0, T)
    probs = (base + recency).softmax(-1)
    acc = (base + 0.5 * torch.randn(T, generator=g)).softmax(-1)    # a noisy history of the same pattern
    return acc, probs
