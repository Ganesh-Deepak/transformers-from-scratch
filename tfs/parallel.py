"""
Distributed-training arithmetic — Chapter 19.

Bytes on the wire for context parallelism (ring attention, all-gather CP,
DeepSpeed-Ulysses), FSDP vs HSDP traffic, and the checkpoint-interval maths
of fault tolerance. Activation memory per layer lives in tfs/memory.py.

Conventions: T is the full sequence length, cp the context-parallel degree,
`bytes_per` the element size (2 for bf16). Volumes are what ONE rank SENDS
for ONE layer's attention in the FORWARD pass unless stated otherwise.
"""

from __future__ import annotations

import math


# ============================================================ context parallelism

def ring_attention_bytes(T, cp, n_kv, d_head, bytes_per=2):
    """
    Ring attention: each rank holds T/cp tokens of Q, K, V. For cp-1 steps it
    passes its current K,V block to the next rank and receives one from the
    previous, computing attention of its local Q against each block as it
    arrives. Per rank:  (cp - 1) * 2 * (T/cp) * n_kv * d_head * bytes.
    The volume does NOT shrink as cp grows: it approaches 2*T*n_kv*d_head*bytes.
    """
    return (cp - 1) * 2 * (T // cp) * n_kv * d_head * bytes_per


def allgather_cp_bytes(T, cp, n_kv, d_head, bytes_per=2):
    """
    Llama 3's variant: all-gather the full K and V, then attend locally. Same
    bytes as the ring (an all-gather sends (cp-1)/cp of the result per rank);
    it gives up overlap for simplicity and for arbitrary masks.
    """
    return (cp - 1) * 2 * (T // cp) * n_kv * d_head * bytes_per


def ulysses_bytes(T, cp, n_head, n_kv, d_head, bytes_per=2):
    """
    DeepSpeed-Ulysses: an all-to-all turns sequence-sharded q, k, v into
    head-sharded q, k, v (each rank then holds ALL T tokens for H/cp heads),
    attention runs locally, and a second all-to-all turns the output back.
    An all-to-all sends (cp-1)/cp of the local tensor, and the local tensor is
    only T/cp tokens, so per rank:
        (cp-1)/cp * (T/cp) * (n_head [q] + 2 n_kv [k, v] + n_head [out]) * d_head * bytes
    which FALLS as 1/cp. Requires cp <= n_kv (or replicating K/V heads).
    """
    local = (T // cp) * (2 * n_head + 2 * n_kv) * d_head * bytes_per
    return local * (cp - 1) // cp


def ring_overlap_min_block(peak_flops, link_bytes_per_s, n_head, n_kv, bytes_per=2,
                           causal=True):
    """
    Smallest block c = T/cp for which a ring step's attention compute hides its
    K,V transfer:
        compute  4 * c^2 * n_head * d_head  FLOPs   (QK^T and PV, forward)
        transfer 2 * c * n_kv * d_head * bytes
        hidden when compute/F >= transfer/BW  <=>  c >= F/BW * n_kv*bytes/(2*n_head)
    With a causal mask half the block pairs are empty, which doubles the need
    unless the sequence is load-balanced (Llama 3's 2*cp chunks, 'zigzag').
    With n_kv = n_head and the paper's accounting this is Liu et al.'s c >= F/B.
    """
    c = peak_flops / link_bytes_per_s * n_kv * bytes_per / (2 * n_head)
    return c * (2 if causal else 1)


def zigzag_chunks(cp):
    """Rank i gets chunks i and 2*cp-1-i of 2*cp: equal causal work per rank."""
    return [(i, 2 * cp - 1 - i) for i in range(cp)]


def causal_work(chunks, n_chunks):
    """Lower-triangle block count a rank owning `chunks` (query chunks) must compute."""
    return sum(c + 1 for c in chunks)


# ============================================================ FSDP vs HSDP

def fsdp_traffic(param_bytes, n, grad_bytes=None):
    """
    ZeRO-3 / FSDP over n ranks, per rank per step: all-gather params for the
    forward, again for the backward, reduce-scatter the gradients.
    Each collective moves (n-1)/n of the full tensor per rank.
    """
    grad_bytes = param_bytes if grad_bytes is None else grad_bytes
    f = (n - 1) / n
    return f * (2 * param_bytes + grad_bytes)


def hsdp_traffic(param_bytes, shard, replicas, grad_bytes=None):
    """
    Hybrid sharding: FSDP inside a group of `shard` ranks (one node, NVLink),
    plain data-parallel all-reduce of each rank's gradient SHARD across the
    `replicas` groups. Returns (intra_node_bytes, inter_node_bytes) per rank.
    """
    grad_bytes = param_bytes if grad_bytes is None else grad_bytes
    intra = fsdp_traffic(param_bytes, shard, grad_bytes)
    inter = 2 * (replicas - 1) / replicas * grad_bytes / shard
    return intra, inter


def hsdp_bytes_per_param(shard):
    """Persistent state per rank under HSDP: 16 bytes/param sharded over the group only."""
    return 16 / shard


# ============================================================ fault tolerance

def young_daly_interval(checkpoint_s, mtbf_s):
    """Young (1974) / Daly (2006): checkpoint every sqrt(2 * delta * MTBF) seconds."""
    return math.sqrt(2 * checkpoint_s * mtbf_s)


def expected_waste(interval_s, checkpoint_s, mtbf_s, restart_s=0.0):
    """
    Fraction of wall-clock lost, first order: writing checkpoints (delta/tau)
    plus, per failure, half an interval of lost work and the restart time.
    """
    return checkpoint_s / interval_s + (interval_s / 2 + restart_s) / mtbf_s


def cluster_mtbf(per_device_mtbf_s, n_devices):
    """Independent failures: the cluster fails n times as often as one device."""
    return per_device_mtbf_s / n_devices
