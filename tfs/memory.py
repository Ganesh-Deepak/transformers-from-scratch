"""
The memory of the backward pass, and the gradients autograd hides — Chapter 10
(and the activation formulas Chapter 19 reuses).

Everything here is either an exact count or a hand-written backward that the
tests check against autograd:

    saved_elements_per_token(cfg)   what one tfs Block keeps for backward, per token
    measure_saved(fn, *inputs)      the same thing, measured with autograd hooks
    activation_bytes(...)           whole-model activations for a real config
    checkpoint_memory(...)          memory with a checkpoint every k layers
    logits_bytes(...)               the vocabulary-sized tensors at the very end
    chunked_cross_entropy(...)      the fix: never materialise (N, V) logits
    korthikanti_bytes_per_layer     the published GPT-style formula (Ch 19)
    embedding_backward / swiglu_backward / rope_backward
    accumulate(grads, dtype)        why reductions run in fp32
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ===================================================== what backward keeps

def saved_elements_per_token(d_model, n_head, d_head, d_ff, expanded_kv=True,
                             n_kv_head=None):
    """
    Elements one tfs.model.Block saves for backward, PER TOKEN (weights excluded).

        2 RMSNorms  : input x, 1/rms, x*(1/rms), output        -> 2 * (3C + 1)
        attention   : q, k, v as SDPA sees them, its output (which IS the
                      o_proj input: the transpose+reshape is a view of the
                      kernel's (B, T, H, dh) layout), one logsumexp per
                      head per token. k and v are already repeat_interleaved
                      to n_head, so GQA saves nothing here     -> 4*H*dh + H
        SwiGLU      : gate(x), silu(gate), up(x), their product -> 4 * d_ff

    `test_saved_tensor_count_matches_autograd` measures this on the real Block.
    With expanded_kv=False (SDPA's enable_gqa, or a GQA-aware kernel) k and v
    are stored at n_kv_head width instead.
    """
    C, H, dh = d_model, n_head, d_head
    kv = H if expanded_kv else n_kv_head
    norms = 2 * (3 * C + 1)
    attn = 2 * H * dh + 2 * kv * dh + H        # q, out (= o_proj input); k, v; lse
    mlp = 4 * d_ff
    return norms + attn + mlp


def measure_saved(fn, *inputs, exclude=()):
    """
    Run fn(*inputs) under saved_tensors_hooks and return the number of BYTES
    autograd keeps for backward, counting each storage once and skipping the
    storages in `exclude` (pass the parameters and buffers: they are not
    activations, they exist anyway).
    """
    skip = {t.untyped_storage().data_ptr() for t in exclude}
    seen = {}

    def pack(t):
        p = t.untyped_storage().data_ptr()
        if p not in skip:
            seen.setdefault(p, t.untyped_storage().nbytes())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(*inputs)
    return sum(seen.values()), out


def activation_bytes(d_model, n_head, d_head, d_ff, n_layer, B, T,
                     bytes_per=2, expanded_kv=True, n_kv_head=None):
    """All layers' saved activations (no checkpointing)."""
    per_tok = saved_elements_per_token(d_model, n_head, d_head, d_ff,
                                       expanded_kv, n_kv_head)
    return per_tok * B * T * n_layer * bytes_per


def checkpoint_memory(n_layer, a, b, k):
    """
    Activation memory with a checkpoint every k layers, in units of per-token
    elements. a = what one layer saves internally, b = one layer boundary (C).

        stored boundaries          (n_layer / k) * b
        one segment re-run        k * (a + b)       (its internals, live in backward)

    Minimised at k* = sqrt(n_layer * b / (a + b)).  When a >> b — as in every
    transformer, a ~ 24 C vs b = C — k* rounds to 1: checkpoint every layer.
    """
    return math.ceil(n_layer / k) * b + k * (a + b)


def best_checkpoint_interval(n_layer, a, b):
    return math.sqrt(n_layer * b / (a + b))


def korthikanti_bytes_per_layer(s, b, h, a, t=1, sp=False, recompute="none"):
    """
    Korthikanti et al. 2022, activation bytes per GPT layer (2-byte activations,
    1-byte dropout masks, 4h GeLU MLP, materialised attention scores):

        no parallelism                 s*b*h*(34 + 5*a*s/h)
        tensor parallel t              s*b*h*(10 + 24/t + 5*a*s/(h*t))
        tensor + sequence parallel     s*b*h*(34 + 5*a*s/h) / t
        selective recompute (+SP)      s*b*h*34 / t        (attention core re-run)
        full recompute                 2*s*b*h             (only the layer input)
    """
    if recompute == "full":
        return 2 * s * b * h
    attn_core = 0 if recompute == "selective" else 5 * a * s / h
    if sp:
        return s * b * h * (34 + attn_core) / t
    return s * b * h * (10 + 24 / t + attn_core / t)


# ===================================================== the logits problem

def logits_bytes(B, T, V, logit_bytes=2, fp32_copies=2):
    """
    The tensors at the end of the model, for N = B*T positions:
        the head output (N, V) in the compute dtype
        + fp32_copies x (N, V) in fp32: log-softmax output saved for backward,
          and the fp32 gradient of the logits that backward creates.
    """
    N = B * T
    return N * V * logit_bytes + fp32_copies * N * V * 4


class _ChunkedLinearCE(torch.autograd.Function):
    """
    Loss = mean cross-entropy of (h @ W.T) against targets, computed chunk by
    chunk so the (N, V) logits never exist at once. Like Liger's fused linear
    cross-entropy, the gradient is computed IN THE FORWARD pass (it is p - y,
    Ch 10.3), so nothing vocabulary-sized is saved for backward.
    """

    @staticmethod
    def forward(ctx, h, W, targets, chunk):
        N = h.shape[0]
        acc = torch.promote_types(h.dtype, torch.float32)   # fp32 at least
        loss = h.new_zeros((), dtype=acc)
        gh = torch.zeros_like(h, dtype=acc)
        gW = torch.zeros_like(W, dtype=acc)
        for s in range(0, N, chunk):
            hc, tc = h[s:s + chunk], targets[s:s + chunk]
            z = (hc @ W.T).to(acc)                          # (n, V)  only one chunk alive
            lse = torch.logsumexp(z, dim=-1)                # (n,)
            loss += (lse - z.gather(1, tc[:, None]).squeeze(1)).sum()
            g = z.sub_(lse[:, None]).exp_()                 # softmax, in place
            g[torch.arange(len(tc)), tc] -= 1.0             # p - y
            g /= N
            gh[s:s + chunk] = g @ W.to(acc)                 # (n, C)
            gW += g.T @ hc.to(acc)                          # (V, C)
        ctx.save_for_backward(gh.to(h.dtype), gW.to(W.dtype))
        return loss / N

    @staticmethod
    def backward(ctx, grad_out):
        gh, gW = ctx.saved_tensors
        return gh * grad_out, gW * grad_out, None, None


def chunked_cross_entropy(h, W, targets, chunk=1024):
    """h: (N, C) final hidden states, W: (V, C) LM head, targets: (N,) int64."""
    return _ChunkedLinearCE.apply(h, W, targets, chunk)


# ===================================================== backward, by hand

def embedding_backward(ids, grad_out, vocab_size):
    """
    The embedding is a gather, so its backward is a SCATTER-ADD: every
    position's gradient is added into the row of the token it looked up.
    A token that appears 37 times gets the sum of 37 gradients.
    ids: (N,)  grad_out: (N, C)  ->  (V, C), all-zero rows for unseen tokens.
    """
    g = torch.zeros(vocab_size, grad_out.shape[-1], dtype=grad_out.dtype)
    return g.index_add_(0, ids, grad_out)


def swiglu_backward(a, b, g):
    """
    y = silu(a) * b, with a = x W_gate, b = x W_up.  g = dL/dy.
        dL/db = g * silu(a)
        dL/da = g * b * silu'(a),   silu'(a) = s(a) * (1 + a * (1 - s(a))),  s = sigmoid
    """
    s = torch.sigmoid(a)
    silu = a * s
    return g * b * s * (1 + a * (1 - s)), g * silu


def rope_backward(g, cos, sin):
    """
    RoPE multiplies each (i, i + d/2) pair by a rotation matrix R(theta).
    R is orthogonal, so the backward is R^T = R(-theta): rotate the gradient
    back by the same angle. Same code as the forward with sin negated.
    """
    from tfs.model import rotate_half
    return g * cos - rotate_half(g) * sin


# ===================================================== precision of reductions

def accumulate(grads, dtype):
    """Sum a list of gradient tensors with the running total held in `dtype`."""
    acc = torch.zeros_like(grads[0], dtype=dtype)
    for g in grads:
        acc = acc + g.to(dtype)
    return acc
