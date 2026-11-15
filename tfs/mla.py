"""
Chapter 16 reference code: MLA decode, naive and absorbed.

The absorbed form folds W_UK into the query and W_UV into the output, so every
head attends directly against the cached latent [c_kv ; k_rope]. All heads read
the SAME cached vector: decode attention becomes multi-QUERY attention with one
"KV head" whose key is 576 wide (512 latent + 64 RoPE) and whose value is the
512-wide latent. tests/test_inference.py checks the two forms agree.
"""

from __future__ import annotations

import torch


def mla_decode_naive(q_nope, q_rope, c_kv, k_rope, W_uk, W_uv):
    """
    Decompress every cached token into per-head keys and values, then attend.
      q_nope (H, dn)  q_rope (H, dr)    one new token's query, per head
      c_kv   (T, r)   k_rope (T, dr)    the cache: latent + shared RoPE key
      W_uk   (H, dn, r)  W_uv (H, dv, r)
    returns (H, dv)
    """
    k_nope = torch.einsum("hdr,tr->htd", W_uk, c_kv)                 # (H, T, dn)
    v = torch.einsum("hdr,tr->htd", W_uv, c_kv)                      # (H, T, dv)
    scale = (q_nope.size(-1) + q_rope.size(-1)) ** -0.5
    s = torch.einsum("hd,htd->ht", q_nope, k_nope) + q_rope @ k_rope.T   # (H, T)
    p = (s * scale).softmax(-1)
    return torch.einsum("ht,htd->hd", p, v)                          # (H, dv)


def mla_decode_absorbed(q_nope, q_rope, c_kv, k_rope, W_uk, W_uv):
    """
    Same function, never decompressing the cache.
      q_lat = W_uk^T q_nope                (H, r)    query moved into latent space
      score = [q_lat ; q_rope] . [c_kv ; k_rope]     one 576-wide dot per head and token
      o_lat = p @ c_kv                     (H, r)    values stay latent too
      out   = W_uv o_lat                   (H, dv)   (in a real model folded into W_o)
    """
    q_lat = torch.einsum("hdr,hd->hr", W_uk, q_nope)                 # (H, r)
    q = torch.cat([q_lat, q_rope], -1)                               # (H, r + dr)
    kv = torch.cat([c_kv, k_rope], -1)                               # (T, r + dr)   ONE shared "head"
    scale = (q_nope.size(-1) + q_rope.size(-1)) ** -0.5
    p = (q @ kv.T * scale).softmax(-1)                               # (H, T)
    o_lat = p @ c_kv                                                 # (H, r)
    return torch.einsum("hdr,hr->hd", W_uv, o_lat)                   # (H, dv)


def decode_attention_intensity(n_q_heads, k_dim, v_dim, cached_values, dtype_bytes=2):
    """
    FLOPs per byte of cache read, for one new query token attending to one
    cached token: each query head does a k_dim dot (2*k_dim FLOPs) and a
    v_dim weighted add (2*v_dim FLOPs); the bytes are what is cached per token.
    """
    flops = n_q_heads * (2 * k_dim + 2 * v_dim)
    return flops / (cached_values * dtype_bytes)
