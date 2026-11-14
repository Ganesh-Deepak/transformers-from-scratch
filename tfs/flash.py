"""
Chapter 17 reference code.

    flash_attention     the tiled forward pass of §17.4, with the in-tile causal mask
    split_kv_decode     Flash-Decoding: split the KEYS across workers, merge by log-sum-exp
    decode_occupancy    why decode needs the split at all
    mask mods           FlexAttention-style predicates, as plain functions

All exact against naive attention; tests/test_inference.py checks it.
"""

from __future__ import annotations

import math

import torch


def naive_attention(Q, K, V, causal=True):
    """Q: (N, d)  K, V: (M, d). Causal aligns the LAST query with the LAST key."""
    N, M = Q.size(0), K.size(0)
    S = Q @ K.T / Q.size(-1) ** 0.5                                    # (N, M)
    if causal:
        qpos = torch.arange(M - N, M)[:, None]
        S = S.masked_fill(torch.arange(M)[None, :] > qpos, float("-inf"))
    return S.softmax(-1) @ V


def flash_attention(Q, K, V, Br=16, Bc=16, causal=True):
    """
    §17.4, runnable. Only a (Br, Bc) score tile ever exists. Blocks entirely
    above the diagonal are skipped; the diagonal blocks still need the
    element-wise causal mask inside the tile.
    """
    N, d = Q.shape
    O = torch.zeros_like(Q)
    for i in range(0, N, Br):
        Qi = Q[i:i + Br]                                               # (br, d)
        br = Qi.size(0)
        Oi = torch.zeros(br, d)
        mi = torch.full((br,), float("-inf"))
        li = torch.zeros(br)
        for j in range(0, N, Bc):
            if causal and j > i + br - 1:
                break                                                  # every key in this block is in the future
            Kj, Vj = K[j:j + Bc], V[j:j + Bc]
            Sij = Qi @ Kj.T / math.sqrt(d)                             # (br, bc)
            if causal and j + Kj.size(0) - 1 > i:                      # block straddles the diagonal
                qi = torch.arange(i, i + br)[:, None]
                kj = torch.arange(j, j + Kj.size(0))[None, :]
                Sij = Sij.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(mi, Sij.max(-1).values)
            corr = torch.exp(mi - m_new)
            Pij = torch.exp(Sij - m_new[:, None])
            li = corr * li + Pij.sum(-1)
            Oi = corr[:, None] * Oi + Pij @ Vj
            mi = m_new
        O[i:i + br] = Oi / li[:, None]
    return O


def attend_partial(q, K, V):
    """One worker's share: unnormalised output and the log-sum-exp of its scores."""
    s = K @ q / q.numel() ** 0.5                                       # (m,)
    lse = torch.logsumexp(s, 0)
    return torch.exp(s - lse) @ V, lse                                 # (d,), ()


def split_kv_decode(q, K, V, n_splits):
    """
    Flash-Decoding for one query token. Each split attends to its own slice of
    the cache and returns a NORMALISED partial output plus its LSE; the merge
    weights each partial by exp(lse_i - lse_total). Exact.
    """
    T = K.size(0)
    size = -(-T // n_splits)
    outs, lses = [], []
    for s in range(0, T, size):
        o, l = attend_partial(q, K[s:s + size], V[s:s + size])
        outs.append(o)
        lses.append(l)
    return merge_partials(torch.stack(outs), torch.stack(lses))


def merge_partials(outs, lses):
    """
    The Flash-Decoding reduction. outs: (S, d) normalised partial outputs,
    lses: (S,) their log-sum-exps. Weight each by exp(lse_i - lse_total).
    """
    w = torch.exp(lses - torch.logsumexp(lses, 0))                     # (S,)  sums to 1
    return (w[:, None] * outs).sum(0)                                  # (d,)


def scores_partial(scores, values):
    """attend_partial when the scores are given: (normalised output, lse)."""
    lse = torch.logsumexp(scores, 0)
    return torch.exp(scores - lse) @ values, lse


def decode_occupancy(batch, heads, n_sm, splits=1):
    """Work items (thread blocks) per SM. Below 1, SMs sit idle."""
    return batch * heads * splits / n_sm


# ---------------------------------------------------------------- FlexAttention-style masks
# A mask_mod answers "may query q_idx see key kv_idx?" -- the same signature
# torch.nn.attention.flex_attention uses (b, h, q_idx, kv_idx).

def causal(b, h, q, k):
    return q >= k


def sliding_window(w):
    def mod(b, h, q, k):
        return (q >= k) & (q - k < w)
    return mod


def document(doc_id):
    """Packed sequences: attend only within your own document, causally."""
    def mod(b, h, q, k):
        return (doc_id[q] == doc_id[k]) & (q >= k)
    return mod


def prefix_lm(prefix_len):
    """Bidirectional over the prefix, causal after it."""
    def mod(b, h, q, k):
        return (k < prefix_len) | (q >= k)
    return mod


def dense_mask(mod, T):
    q = torch.arange(T)[:, None]
    k = torch.arange(T)[None, :]
    return mod(0, 0, q, k)


def block_sparsity(mask, block):
    """Fraction of (block x block) tiles a kernel can skip outright: entirely False."""
    T = mask.size(0)
    n = skip = 0
    for i in range(0, T, block):
        for j in range(0, T, block):
            n += 1
            skip += int(not mask[i:i + block, j:j + block].any())
    return skip / n
