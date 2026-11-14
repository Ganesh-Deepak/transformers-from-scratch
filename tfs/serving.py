"""
Chapter 18 reference code: serving economics.

    SLO metrics                      percentile, slo_attainment, goodput
    a roofline model of one step     step_time, decode_profile, max_batch_for_tpot
    the chunked-prefill trade        chunked_prefill_trade
    $ per million tokens             cost_per_mtok
    MoE with expert parallelism      tokens_per_expert, batch_for_compute_bound_experts
    multi-LoRA                       lora_params, lora_batch_bytes
    KV offload tiers                 prefill_flops, reload_vs_recompute
    weight formats                   bits_per_weight, weight_bytes

The model is deliberately the Chapter 14 roofline and nothing more: a step
takes max(bytes / bandwidth, FLOPs / peak). Real engines reach 60-80% of it.
Every derived number in Chapter 18 is asserted in tests/test_inference.py.
"""

from __future__ import annotations

import math

from tfs.gpu import GPUS

GIB = 1024 ** 3

# Llama-3-8B (config.json): 32 layers, d_model 4096, 32 q heads, 8 kv heads, d_head 128,
# d_ff 14336, vocab 128256.  Parameter count 8.03 B.
LLAMA3_8B = dict(n_params=8.03e9, n_layer=32, d_model=4096, n_head=32, n_kv=8, d_head=128,
                 d_ff=14336)


def kv_bytes_per_token(n_layer, n_kv, d_head, dtype_bytes=2):
    return 2 * n_layer * n_kv * d_head * dtype_bytes


# ================================================================ SLO metrics


def percentile(xs, q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q% of samples at or below it."""
    s = sorted(xs)
    k = max(1, math.ceil(q / 100 * len(s)))
    return s[k - 1]


def slo_attainment(requests, ttft_slo: float, tpot_slo: float) -> float:
    """Fraction of requests whose TTFT AND mean TPOT both meet their targets."""
    ok = sum(1 for r in requests if r["ttft"] <= ttft_slo and r["tpot"] <= tpot_slo)
    return ok / len(requests)


def goodput(requests, ttft_slo: float, tpot_slo: float, seconds: float) -> float:
    """Requests per second that met their SLO. Throughput counts the rest too."""
    ok = sum(1 for r in requests if r["ttft"] <= ttft_slo and r["tpot"] <= tpot_slo)
    return ok / seconds


# ================================================================ one step


def step_time(n_params, weight_bytes, kv_read_bytes, tokens, gpu="H100", fmt="bf16"):
    """
    One forward pass over `tokens` tokens (decodes + prefill-chunk tokens),
    reading all weights once plus `kv_read_bytes` of cache.
    """
    g = GPUS[gpu]
    mem = (weight_bytes + kv_read_bytes) / g["hbm_bw"]
    comp = 2 * n_params * tokens / g[fmt]
    return dict(memory=mem, compute=comp, t=max(mem, comp),
                bound="compute" if comp > mem else "memory")


def decode_profile(b, ctx, m=LLAMA3_8B, gpu="H100", weight_bytes_per_param=2, kv_dtype_bytes=2):
    """Decode at batch b, each sequence holding ctx tokens of cache."""
    W = m["n_params"] * weight_bytes_per_param
    kv = kv_bytes_per_token(m["n_layer"], m["n_kv"], m["d_head"], kv_dtype_bytes)
    s = step_time(m["n_params"], W, b * ctx * kv, b, gpu)
    s["tpot"] = s["t"]
    s["tok_per_s"] = b / s["t"]
    return s


def max_batch_for_tpot(tpot_slo, ctx, m=LLAMA3_8B, gpu="H100"):
    b = 1
    while decode_profile(b + 1, ctx, m, gpu)["tpot"] <= tpot_slo:
        b += 1
    return b


def max_batch_for_memory(hbm_bytes, ctx, reserve_bytes, m=LLAMA3_8B, weight_bytes_per_param=2,
                         kv_dtype_bytes=2):
    W = m["n_params"] * weight_bytes_per_param
    kv = kv_bytes_per_token(m["n_layer"], m["n_kv"], m["d_head"], kv_dtype_bytes)
    return int((hbm_bytes - W - reserve_bytes) // (ctx * kv))


# ================================================================ chunked prefill


def chunked_prefill_trade(chunk, prompt, b, ctx, m=LLAMA3_8B, gpu="H100"):
    """
    Every step runs b decodes plus up to `chunk` prompt tokens of one waiting
    prefill. Larger chunks finish the prompt in fewer steps (TTFT) but make
    every step longer for the b users already decoding (TPOT).
    Attention FLOPs of the chunk are ignored (small next to 2N per token at
    these lengths); the cache the chunk reads is not.
    """
    W = m["n_params"] * 2
    kv = kv_bytes_per_token(m["n_layer"], m["n_kv"], m["d_head"])
    steps = math.ceil(prompt / chunk)
    s = step_time(m["n_params"], W, b * ctx * kv, b + chunk, gpu)
    return dict(steps=steps, tpot=s["t"], ttft=steps * s["t"], bound=s["bound"])


def balanced_chunk(b, ctx, m=LLAMA3_8B, gpu="H100"):
    """The chunk size at which compute time equals memory time: free prefill tokens."""
    g = GPUS[gpu]
    W = m["n_params"] * 2
    kv = kv_bytes_per_token(m["n_layer"], m["n_kv"], m["d_head"])
    mem = (W + b * ctx * kv) / g["hbm_bw"]
    return int(mem * g["bf16"] / (2 * m["n_params"])) - b


# ================================================================ cost


def cost_per_mtok(gpu_hour_usd: float, tokens_per_s: float, utilisation: float = 1.0) -> float:
    """Dollars per million tokens. Utilisation = fraction of paid hours spent at this throughput."""
    return gpu_hour_usd / (tokens_per_s * 3600 * utilisation) * 1e6


# ================================================================ MoE serving


def tokens_per_expert(global_tokens: int, top_k: int, n_expert: int) -> float:
    """With uniform routing, each routed expert sees B*k/E tokens per step."""
    return global_tokens * top_k / n_expert


def batch_for_compute_bound_experts(n_expert, top_k, gpu="H100", fmt="fp8", weight_bytes=1):
    """
    An expert's matmul with n tokens does 2n FLOPs per weight, reading
    weight_bytes per weight. Compute-bound when 2n / weight_bytes >= ridge.
    Returns (tokens each expert needs, global decode batch that delivers it).
    """
    g = GPUS[gpu]
    n = g[fmt] / g["hbm_bw"] * weight_bytes / 2
    return n, n * n_expert / top_k


# ================================================================ multi-LoRA


def lora_params(shapes, r: int) -> int:
    """LoRA on W (d_out x d_in) adds B (d_out x r) and A (r x d_in): r(d_in + d_out)."""
    return sum(r * (din + dout) for din, dout in shapes)


def llama_linear_shapes(m=LLAMA3_8B):
    """(d_in, d_out) of the seven adapted matrices in one Llama block."""
    d, kvd, f = m["d_model"], m["n_kv"] * m["d_head"], m["d_ff"]
    return [(d, d), (d, kvd), (d, kvd), (d, d), (d, f), (d, f), (f, d)]


def lora_batch_bytes(adapter_bytes: float, n_distinct: int) -> float:
    """Extra bytes read per decode step: each DISTINCT adapter in the batch is read once."""
    return adapter_bytes * n_distinct


# ================================================================ KV offload


def prefill_flops(n_params, n_layer, d_attn, T):
    """2N per token for the weights + causal attention: QK^T and PV, 2 FLOPs each, over T^2/2 pairs."""
    return 2 * n_params * T + 2 * n_layer * d_attn * T * T


def reload_vs_recompute(T, link_bw, m=LLAMA3_8B, gpu="H100", util=0.5):
    """Seconds to bring back T tokens of cache over a link vs recomputing them by prefill."""
    kv = kv_bytes_per_token(m["n_layer"], m["n_kv"], m["d_head"])
    reload = T * kv / link_bw
    fl = prefill_flops(m["n_params"], m["n_layer"], m["n_head"] * m["d_head"], T)
    recompute = fl / (GPUS[gpu]["bf16"] * util)
    return dict(kv_bytes=T * kv, reload=reload, recompute=recompute, flops=fl)


# ================================================================ weight formats


def bits_per_weight(fmt: str) -> float:
    """
    Storage cost including block scales (Chapter 0 §0.6):
      bf16 16 · fp8 with one fp32 scale per 128x128 block (DeepSeek-V3) 8 + 32/16384
      MXFP4: 4-bit e2m1 + one 8-bit E8M0 scale per 32 values  -> 4.25
      NVFP4: 4-bit e2m1 + one 8-bit e4m3 scale per 16 values  -> 4.5 (+ one fp32 per tensor)
    """
    return {"bf16": 16.0, "fp8": 8 + 32 / (128 * 128), "mxfp4": 4 + 8 / 32, "nvfp4": 4 + 8 / 16}[fmt]


def weight_bytes(n_params: float, fmt: str) -> float:
    return n_params * bits_per_weight(fmt) / 8
