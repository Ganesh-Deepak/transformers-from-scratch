"""
Scaled low-precision arithmetic — Chapter 12.6-12.7.

    quantize(x, fmt, block)    scale -> cast -> unscale, per tensor or per block
    delayed_scale(history)     Transformer Engine's amax-history recipe
    bf16_accumulate(...)       adding small updates to a bf16 weight three ways:
                               round-to-nearest, stochastic rounding, Kahan
"""

from __future__ import annotations

import torch

FMT = {"e4m3": (torch.float8_e4m3fn, 448.0), "e5m2": (torch.float8_e5m2, 57344.0)}


def quantize(x, fmt="e4m3", block=None):
    """
    Round-trip x through fp8 with a scale chosen so the largest |value| in each
    scaling group lands on the format's max. block=None: one scale for the whole
    tensor. block=128: one scale per 128 consecutive elements of the last dim
    (DeepSeek-V3's 1x128 activation tiles). Returns the dequantised tensor.
    """
    dt, fmax = FMT[fmt]
    shp = x.shape
    g = x.reshape(-1) if block is None else x.reshape(-1, block)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = fmax / amax
    q = (g * scale).to(dt).to(torch.float32) / scale
    return q.reshape(shp)


def rel_error(x, xq):
    return ((x - xq).norm() / x.norm()).item()


def delayed_scale(amax_history, fmt="e4m3", margin=0):
    """
    'Delayed scaling': choose this step's scale from the max amax of the last few
    steps instead of the current tensor, so no extra pass over the data is needed.
    If the current tensor exceeds that history, values saturate at fmax.
    """
    _, fmax = FMT[fmt]
    return fmax / max(amax_history) / 2 ** margin


def bf16_accumulate(w0, update, steps, mode="nearest", seed=0):
    """
    Add `update` to a bf16 weight `steps` times.
      nearest     plain bf16 arithmetic: every add rounds to nearest
      stochastic  round up with probability equal to the fractional position
                  between the two neighbouring bf16 values (unbiased on average)
      kahan       keep a bf16 compensation term holding what rounding threw away
    Returns the final weight as a Python float.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.tensor(w0, dtype=torch.bfloat16)
    u = torch.tensor(update, dtype=torch.float32)
    comp = torch.tensor(0.0, dtype=torch.bfloat16)
    for _ in range(steps):
        if mode == "nearest":
            w = (w.float() + u).to(torch.bfloat16)
        elif mode == "stochastic":
            # bf16 is the top 16 bits of fp32. Adding a uniform random integer to
            # the low 16 bits and truncating rounds up with probability equal to
            # the discarded fraction: E[result] = exact (Chapter 0, section 0.8).
            bits = (w.float() + u).view(torch.int32)
            r = torch.randint(0, 1 << 16, (), generator=g, dtype=torch.int32)
            w = ((bits + r) & -65536).view(torch.float32).to(torch.bfloat16)
        elif mode == "kahan":
            y = (u + comp.float())
            t = (w.float() + y).to(torch.bfloat16)
            comp = (y - (t.float() - w.float())).to(torch.bfloat16)
            w = t
    return w.float().item()
