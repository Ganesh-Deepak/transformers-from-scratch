"""
Linear attention, the delta rule, and gated delta networks — Chapter 22.

Every function works on ONE head, shapes written on every line:

    q, k : (T, dk)        queries and keys            (T = time, dk = key width)
    v    : (T, dv)        values                      (dv = value width)
    beta : (T,)           write strength in [0, 1]
    alpha: (T,)           forget gate in (0, 1]
    S    : (dv, dk)       the state: a matrix that maps a key to a value

The state is a *fast weight*: a small linear map S, rewritten at every token,
read by S @ q. Softmax attention keeps every (k, v) pair; these layers keep
only S, so memory is constant in T.

    linear attention     S_t = S_{t-1}                       + v_t k_t^T
    delta rule           S_t = S_{t-1} (I - b_t k_t k_t^T)   + b_t v_t k_t^T
    gated delta rule     S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T
    output               o_t = S_t q_t

The recurrent forms are the definitions. `chunked_delta_rule` is how the layer
is trained (Yang et al. 2024, "Parallelizing Linear Transformers with the Delta
Rule over Sequence Length"): within a chunk of C tokens everything is a matmul,
and only the chunk-to-chunk state is sequential. Tests assert that it equals
the recurrence to float64 precision.
"""

from __future__ import annotations

import math

import torch


# ============================================================ recurrent forms


def linear_attention_recurrent(q, k, v):
    """Causal linear attention (Katharopoulos et al. 2020) as a recurrence."""
    T, dk = k.shape
    dv = v.shape[1]
    S = q.new_zeros(dv, dk)                                  # (dv, dk)
    out = []
    for t in range(T):
        S = S + torch.outer(v[t], k[t])                      # (dv, dk)  write: add v k^T
        out.append(S @ q[t])                                 # (dv,)     read:  S q
    return torch.stack(out), S                               # (T, dv), (dv, dk)


def linear_attention_parallel(q, k, v):
    """Same thing as one masked matmul: o = ((Q K^T) * causal) V. No softmax."""
    T = q.shape[0]
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    A = (q @ k.T) * mask                                     # (T, T)
    return A @ v                                             # (T, dv)


def gated_delta_rule_recurrent(q, k, v, beta, alpha=None, S0=None):
    """
    The gated delta rule (Yang, Kautz & Hatamizadeh 2025, "Gated Delta Networks").
    alpha=None gives the plain delta rule (DeltaNet, Schlag et al. 2021).

    Each step is one gradient-descent step on the regression loss
    0.5 * ||S k_t - v_t||^2 with learning rate beta_t, after decaying S by alpha_t:
        S <- alpha S
        S <- S - beta (S k - v) k^T
    """
    T, dk = k.shape
    dv = v.shape[1]
    S = q.new_zeros(dv, dk) if S0 is None else S0.clone()   # (dv, dk)
    out = []
    for t in range(T):
        if alpha is not None:
            S = alpha[t] * S                                 # (dv, dk)  forget
        err = S @ k[t] - v[t]                                # (dv,)     what S predicts for k_t, minus the truth
        S = S - beta[t] * torch.outer(err, k[t])             # (dv, dk)  correct only along k_t
        out.append(S @ q[t])                                 # (dv,)
    return torch.stack(out), S                               # (T, dv), (dv, dk)


def delta_rule_recurrent(q, k, v, beta, S0=None):
    return gated_delta_rule_recurrent(q, k, v, beta, None, S0)


# ============================================================ chunked (training) form


def chunked_delta_rule(q, k, v, beta, alpha=None, chunk=64, S0=None):
    """
    The chunkwise-parallel gated delta rule.

    Inside a chunk, write the state as S_t = g_t S0 + sum_{i<=t} (g_t/g_i) n_i k_i^T,
    where g_t = alpha_1 ... alpha_t (cumulative decay inside the chunk) and
    n_i = beta_i (v_i - alpha_i S_{i-1} k_i) is the "corrected value" each step
    writes. Substituting gives a LOWER-TRIANGULAR linear system for all n_i at once:

        (I + diag(beta) * strict_lower(K K^T * G)) N = diag(beta) (V - diag(g) K S0^T)

    (G[t, j] = g_t / g_j <= 1). Solving it is the only sequential piece, and it is
    C x C. Everything else is a matmul:

        O      = diag(g) Q S0^T + (Q K^T * G * causal) N         (C, dv)
        S_next = g_C S0 + (N * (g_C / g))^T K                    (dv, dk)

    Returns (T, dv) outputs and the final (dv, dk) state.
    """
    T, dk = k.shape
    dv = v.shape[1]
    S = q.new_zeros(dv, dk) if S0 is None else S0.clone()    # (dv, dk)
    outs = []
    for s in range(0, T, chunk):
        Q, K, V, b = q[s:s + chunk], k[s:s + chunk], v[s:s + chunk], beta[s:s + chunk]
        C = K.shape[0]                                       # last chunk may be short
        if alpha is None:
            logg = q.new_zeros(C)
        else:
            logg = torch.cumsum(torch.log(alpha[s:s + chunk]), 0)   # (C,)  log g_t
        g = logg.exp()                                       # (C,)
        G = (logg[:, None] - logg[None, :]).exp()            # (C, C)   g_t / g_j
        strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
        incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()

        A = torch.eye(C, dtype=q.dtype, device=q.device) + \
            b[:, None] * ((K @ K.T) * G * strict)            # (C, C)   unit lower-triangular
        rhs = b[:, None] * (V - g[:, None] * (K @ S.T))      # (C, dv)
        N = torch.linalg.solve_triangular(A, rhs, upper=False)   # (C, dv)  the UT transform

        O = g[:, None] * (Q @ S.T) + ((Q @ K.T) * G * incl) @ N  # (C, dv)
        S = g[-1] * S + (N * (g[-1] / g)[:, None]).T @ K     # (dv, dk)
        outs.append(O)
    return torch.cat(outs), S                                # (T, dv), (dv, dk)


# ============================================================ what the state can remember


def recall_error(n_pairs, dk=64, rule="linear", keys="random", seed=0):
    """
    Associative recall with one head: write n (key, value) pairs, then query every
    key and compare what comes back with the value that was written.

    rule:  "linear"  (add v k^T), "delta" (beta = 1), "softmax" (keep every pair)
    keys:  "orthogonal" (needs n <= dk) or "random" (unit vectors)
    Returns the mean relative error ||read - v|| / ||v||.
    """
    g = torch.Generator().manual_seed(seed)
    dv = dk
    if keys == "orthogonal":
        assert n_pairs <= dk
        Qm, _ = torch.linalg.qr(torch.randn(dk, dk, generator=g, dtype=torch.float64))
        K = Qm[:n_pairs]                                     # (n, dk)  orthonormal rows
    else:
        K = torch.randn(n_pairs, dk, generator=g, dtype=torch.float64)
        K = K / K.norm(dim=1, keepdim=True)                  # (n, dk)  unit rows
    V = torch.randn(n_pairs, dv, generator=g, dtype=torch.float64)   # (n, dv)

    if rule == "softmax":
        scores = (K @ K.T) * 1e3                             # sharp softmax: exact lookup
        read = scores.softmax(-1) @ V                        # (n, dv)
    else:
        beta = torch.ones(n_pairs, dtype=torch.float64)
        if rule == "linear":
            _, S = linear_attention_recurrent(K, K, V)
        else:
            _, S = delta_rule_recurrent(K, K, V, beta)
        read = K @ S.T                                       # (n, dv)  query every key
    return ((read - V).norm(dim=1) / V.norm(dim=1)).mean().item()


def recall_error_by_age(n_pairs, dk=32, rule="delta", seed=0, frac=0.1):
    """Mean relative recall error for the OLDEST and the NEWEST `frac` of the pairs written."""
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(n_pairs, dk, generator=g, dtype=torch.float64)
    K = K / K.norm(dim=1, keepdim=True)                      # (n, dk)
    V = torch.randn(n_pairs, dk, generator=g, dtype=torch.float64)
    if rule == "linear":
        _, S = linear_attention_recurrent(K, K, V)
    else:
        _, S = delta_rule_recurrent(K, K, V, torch.ones(n_pairs, dtype=torch.float64))
    err = (K @ S.T - V).norm(dim=1) / V.norm(dim=1)          # (n,)
    m = max(1, int(round(frac * n_pairs)))
    return err[:m].mean().item(), err[-m:].mean().item()


def overwrite_demo():
    """
    The 2-D worked example in Ch 22.2: write (k1, v1), (k2, v2), then write a NEW
    value for k1. Linear attention returns the sum of both k1 values; the delta
    rule returns the new one.
    """
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    v = torch.tensor([[3.0, 1.0], [0.0, 2.0], [-1.0, 4.0]], dtype=torch.float64)
    beta = torch.ones(3, dtype=torch.float64)
    _, S_lin = linear_attention_recurrent(k, k, v)
    _, S_delta = delta_rule_recurrent(k, k, v, beta)
    q = k[0]
    return (S_lin @ q).tolist(), (S_delta @ q).tolist(), S_lin.tolist(), S_delta.tolist()


# ============================================================ the systems arithmetic


def kv_cache_bytes_per_token(n_attn_layers, n_kv_heads, d_head, bytes_per=2):
    """Softmax-attention layers: K and V for every past token."""
    return n_attn_layers * 2 * n_kv_heads * d_head * bytes_per


def linear_state_bytes(n_linear_layers, n_heads, dk, dv, bytes_per=4):
    """Delta-rule layers: one (dv, dk) matrix per head, independent of T."""
    return n_linear_layers * n_heads * dk * dv * bytes_per


def qwen3_next_memory(T, state_bytes=4):
    """
    Qwen3-Next-80B-A3B (config.json): 48 layers, full_attention_interval = 4, so
    12 gated-attention layers (2 KV heads x 256) and 36 Gated DeltaNet layers
    (32 value heads x 128, keys 128 wide). Returns (hybrid_bytes, all_attention_bytes)
    for a T-token context, KV cache in bf16, delta state in `state_bytes`.
    (The short causal-convolution state of each linear layer is ignored: it is
    3 x the projection width per layer, a few hundred KB in total.)
    """
    kv_hybrid = kv_cache_bytes_per_token(12, 2, 256) * T
    state = linear_state_bytes(36, 32, 128, 128, state_bytes)
    kv_dense = kv_cache_bytes_per_token(48, 2, 256) * T
    return kv_hybrid + state, kv_dense


def softmax_attention_flops_per_token(T, dk, dv):
    """Causal: query t scores t keys and mixes t values; average t = T/2."""
    return 2 * (T / 2) * dk + 2 * (T / 2) * dv


def chunked_delta_flops_per_token(C, dk, dv):
    """
    Leading-order matmul FLOPs per token of the chunked delta rule, chunk size C:
      K K^T (2C dk), triangular solve (~C (dk + dv)... counted as C dv), Q K^T (2C dk),
      (QK^T) N (2C dv), Q S^T, K S^T, N^T K (2 dk dv each).
    Independent of T: that is the point.
    """
    return 2 * C * dk + C * dv + 2 * C * dk + 2 * C * dv + 3 * 2 * dk * dv


def interference_std(n_pairs, dk):
    """Random unit keys: each stored pair leaks ~1/sqrt(dk) of its value into a read."""
    return math.sqrt((n_pairs - 1) / dk)
