"""
Scaling-law arithmetic — Chapter 21.

    chinchilla_loss / fitted_optimum   the Hoffmann et al. parametric fit, solved exactly
    rule_of_20                         the 20-tokens-per-parameter heuristic
    effective_data                     Muennighoff et al. 2023: what repeated data is worth
    coverage / pass_at_k               test-time compute: what k samples buy
    attention_share                    why long context is trained last
"""

from __future__ import annotations

import math

HOFFMANN = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)          # rounded, as published
BESIROGLU = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)  # 2024 replication


def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    return E + A / N ** alpha + B / D ** beta


def fitted_optimum(C, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """
    Minimise L(N, D) subject to C = 6 N D. Setting dL/dN = 0 along D = C/(6N):
        alpha A / N^alpha = beta B / D^beta
    which solves to
        N* = G (C/6)^a,  D* = (C/6)^b / G,  a = beta/(alpha+beta), b = alpha/(alpha+beta)
        G  = (alpha A / (beta B))^(1/(alpha+beta))
    With alpha != beta the ratio D*/N* is NOT constant: it drifts with C.
    """
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** a
    D = (C / 6) / N
    return N, D, a, b


def rule_of_20(C, ratio=20):
    """C = 6 N D with D = ratio * N  =>  N = sqrt(C / (6 * ratio))."""
    N = math.sqrt(C / (6 * ratio))
    return N, ratio * N


R_D_STAR = 15.3878     # Muennighoff et al. 2023, Table 1
R_N_STAR = 5.3097


def effective_data(unique, epochs, r_star=R_D_STAR):
    """
    D' = U + U * R* * (1 - exp(-R/R*)),  R = epochs - 1 repetitions.
    Each repetition is worth less than the last; the total can never exceed
    U * (1 + R*), about 16.4 unique-data equivalents.
    """
    R = epochs - 1
    return unique + unique * r_star * (1 - math.exp(-R / r_star))


def coverage(p, k):
    """P(at least one of k independent samples is correct) for per-sample accuracy p."""
    return 1 - (1 - p) ** k


def pass_at_k(n, c, k):
    """Chen et al. 2021 unbiased estimator from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def attention_share(N, n_layer, d_attn, T):
    """
    Fraction of training FLOPs per token spent on attention scores at length T:
        12 * L * d_attn * T  /  (6 N + 12 * L * d_attn * T)
    """
    attn = 12 * n_layer * d_attn * T
    return attn / (6 * N + attn)
