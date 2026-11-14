"""
Chapter 14 reference code: the roofline, MFU, launch overhead, interconnects.

Hardware figures are vendor datasheet numbers, DENSE (no 2:4 sparsity), and
are the only inputs here; every derived number in the chapter comes from the
functions below and is asserted in tests/test_inference.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------- hardware
# FLOP/s per GPU, dense. Bandwidth in bytes/s.  Sources: NVIDIA H100 and
# Blackwell (HGX B200) datasheets. H100 has no fp4 tensor cores.
GPUS = {
    "A100": dict(bf16=312e12, fp8=None, fp4=None, hbm_bw=2.04e12),
    "H100": dict(bf16=989.4e12, fp8=1978.9e12, fp4=None, hbm_bw=3.35e12),
    "H200": dict(bf16=989.4e12, fp8=1978.9e12, fp4=None, hbm_bw=4.8e12),
    "B200": dict(bf16=2250e12, fp8=4500e12, fp4=9000e12, hbm_bw=8.0e12),
}

# Links, bytes/s in ONE direction (vendor figures are often quoted bidirectional).
LINKS = {
    "NVLink 4 (H100), per GPU": 450e9,        # 900 GB/s bidirectional, 18 links
    "NVLink 5 (B200), per GPU": 900e9,        # 1.8 TB/s bidirectional
    "PCIe 5.0 x16": 64e9,                     # 32 GT/s x 16 lanes, ~63 GB/s
    "InfiniBand NDR, one 400 Gb/s port": 50e9,
}


def ridge(gpu: str, fmt: str = "bf16") -> float:
    """FLOP per byte of HBM traffic at which an op stops being memory-bound."""
    g = GPUS[gpu]
    return g[fmt] / g["hbm_bw"]


def decode_batch_to_saturate(gpu: str, fmt: str = "bf16", weight_bytes: float = 2.0) -> float:
    """
    Decode with batch B does 2*B FLOPs per weight, and reads weight_bytes per
    weight, so AI = 2B / weight_bytes. Saturation when that equals the ridge.
    """
    return ridge(gpu, fmt) * weight_bytes / 2


# ---------------------------------------------------------------- MFU


def mfu(n_params: float, tokens_per_s: float, peak: float, train: bool = True) -> float:
    """
    Model FLOPs Utilisation: the FLOPs the MODEL needs (6N per token to train,
    2N to run, attention ignored) divided by what the chip could do.
    Recomputation does not count -- that is hardware FLOPs utilisation (HFU).
    """
    per_token = (6 if train else 2) * n_params
    return per_token * tokens_per_s / peak


def tokens_per_s_at_mfu(n_params: float, util: float, peak: float, train: bool = True) -> float:
    return util * peak / ((6 if train else 2) * n_params)


# ---------------------------------------------------------------- overhead


def decode_step_floor(weight_bytes: float, bw: float, n_kernels: int, launch_s: float):
    """
    Three floors on one decode step. The step can be no faster than the
    largest of them; when the launch floor wins, the GPU is OVERHEAD-bound:
    it is waiting for the CPU to hand it work, not for memory or maths.
    """
    mem = weight_bytes / bw
    launch = n_kernels * launch_s
    return dict(memory=mem, launch=launch, bound="overhead" if launch > mem else "memory")


def kernels_per_decode_step(n_layer: int, per_layer: int = 11, extra: int = 3) -> int:
    """
    A Llama-style block launched op by op: norm, qkv, rope, attention, o-proj,
    add, norm, gate+up, silu*mul, down, add  -> 11 kernels; plus embed, final
    norm, lm-head. Fusion changes the count; this is the unfused baseline.
    """
    return n_layer * per_layer + extra


# ---------------------------------------------------------------- collectives


def ring_allreduce_seconds(nbytes: float, n_gpus: int, link_bw: float) -> float:
    """A ring all-reduce moves 2(n-1)/n of the buffer over each GPU's link."""
    return 2 * (n_gpus - 1) / n_gpus * nbytes / link_bw
