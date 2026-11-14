"""
LoRA — Chapter 23.5. Parameter-efficient fine-tuning, and the arithmetic that
makes it the default.

    W' = W + s * B A        W: (out, in) frozen     A: (r, in)     B: (out, r)
                            s = alpha / r           (LoRA, Hu et al. 2021)
                            s = alpha / sqrt(r)     (rsLoRA, Kalajdzievski 2023)

Facts this file lets you check (every one is asserted in tests/test_posttraining.py):
  1. B starts at zero, so the wrapped model is EXACTLY the pretrained model at step 0,
     and at step 0 only B receives a gradient (A's gradient is s * B^T g x^T = 0).
  2. Zero-initialising BOTH A and B is a saddle: neither ever gets a gradient.
  3. Only A and B are trainable; the base weight gets no gradient at all.
  4. merge() folds s*BA into W; outputs are identical, and unmerge() restores W.
  5. Many adapters can share one base weight inside one batch (multi-LoRA serving).
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap an nn.Linear with a low-rank update. The base layer is frozen."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 rslora: bool = False, dropout: float = 0.0, init_B_zero: bool = True):
        super().__init__()
        assert r > 0
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # the base never trains
        d_out, d_in = base.weight.shape                   # nn.Linear stores (out, in)
        self.r, self.alpha, self.rslora = r, alpha, rslora
        self.scaling = alpha / math.sqrt(r) if rslora else alpha / r
        self.A = nn.Parameter(torch.empty(r, d_in, dtype=base.weight.dtype))     # (r, in)
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=base.weight.dtype))    # (out, r)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same init as nn.Linear / PEFT
        if not init_B_zero:
            nn.init.normal_(self.B, std=0.02)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.B @ self.A)           # (out, r) @ (r, in) -> (out, in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (..., in)
        y = self.base(x)                                  # (..., out)
        if self.merged:
            return y
        # Never materialise B @ A: two thin matmuls cost r*(in+out) per token, not in*out.
        return y + self.scaling * (self.drop(x) @ self.A.T) @ self.B.T   # (..., r) -> (..., out)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False


def apply_lora(model: nn.Module, targets: Iterable[str], r: int = 8, alpha: float = 16.0,
               rslora: bool = False) -> list[str]:
    """
    Replace every nn.Linear whose attribute name is in `targets` with a LoRALinear,
    and freeze everything else. Returns the dotted names that were wrapped.

    For tfs.model the names are q_proj, k_proj, v_proj, o_proj (attention) and
    gate, up, down (SwiGLU). The tied embedding / LM head is never wrapped.
    """
    targets = set(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, rslora=rslora))
                wrapped.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return wrapped


def lora_modules(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def merge_all(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.merge()


def unmerge_all(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.unmerge()


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================ DoRA (one function)

def dora_weight(W0: torch.Tensor, B: torch.Tensor, A: torch.Tensor, m: torch.Tensor,
                scaling: float) -> torch.Tensor:
    """
    DoRA (Liu et al. 2024): W' = m * (W0 + sBA) / ||W0 + sBA||, the norm taken per
    output row of the (out, in) matrix here (the paper's "column" in its in x out layout).
    m (out,) is a trainable magnitude, initialised to ||W0|| so W' == W0 at step 0.
    """
    V = W0 + scaling * (B @ A)                             # (out, in) direction
    return m.unsqueeze(1) * V / V.norm(dim=1, keepdim=True)


# ============================================================ multi-LoRA serving

def multi_lora_linear(x: torch.Tensor, W: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor,
                      adapter_ids: torch.Tensor, scaling: float) -> torch.Tensor:
    """
    One shared base weight, a different adapter per request, ONE batch (the
    Punica / S-LoRA layout, Ch 18). x: (N, in), W: (out, in),
    As: (n_adapters, r, in), Bs: (n_adapters, out, r), adapter_ids: (N,).

        y_i = W x_i + s * B[id_i] (A[id_i] x_i)

    The base matmul is batched across every request; only the thin r-wide
    part is gathered per request.
    """
    base = x @ W.T                                         # (N, out)  shared
    A = As[adapter_ids]                                    # (N, r, in)
    B = Bs[adapter_ids]                                    # (N, out, r)
    low = torch.einsum("nri,ni->nr", A, x)                 # (N, r)
    return base + scaling * torch.einsum("nor,nr->no", B, low)   # (N, out)


# ============================================================ the arithmetic

def lora_params_per_matrix(d_in: int, d_out: int, r: int) -> int:
    """A is (r, in), B is (out, r): r*(in + out) parameters, versus in*out for full."""
    return r * (d_in + d_out)


def llama_linear_shapes(d_model: int, n_head: int, n_kv_head: int, d_head: int, d_ff: int):
    """(name, d_in, d_out) for every Linear in one Llama-style block."""
    return [("q_proj", d_model, n_head * d_head), ("k_proj", d_model, n_kv_head * d_head),
            ("v_proj", d_model, n_kv_head * d_head), ("o_proj", n_head * d_head, d_model),
            ("gate", d_model, d_ff), ("up", d_model, d_ff), ("down", d_ff, d_model)]


def llama_param_counts(vocab: int, d_model: int, n_layer: int, n_head: int, n_kv_head: int,
                       d_head: int, d_ff: int, tied: bool = False) -> dict:
    """Total, and the part that lives in the block Linears (what QLoRA quantises)."""
    linear = n_layer * sum(i * o for _, i, o in llama_linear_shapes(d_model, n_head, n_kv_head, d_head, d_ff))
    embed = vocab * d_model * (1 if tied else 2)
    norms = n_layer * 2 * d_model + d_model
    return {"linear": linear, "embed": embed, "norms": norms, "total": linear + embed + norms}


def lora_adapter_params(d_model, n_layer, n_head, n_kv_head, d_head, d_ff, r,
                        targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down")) -> int:
    shapes = llama_linear_shapes(d_model, n_head, n_kv_head, d_head, d_ff)
    return n_layer * sum(lora_params_per_matrix(i, o, r) for n, i, o in shapes if n in targets)


# bytes per parameter for the trained parameters with AdamW in mixed precision:
#   bf16 weight 2 + bf16 grad 2 + fp32 master copy 4 + Adam m 4 + Adam v 4 = 16
TRAINED_BYTES = 16
# NF4 with double quantisation (Dettmers et al. 2023): 4 bits + 8/64 + 32/(64*256) bits
NF4_BITS = 4 + 8 / 64 + 32 / (64 * 256)


def finetune_memory_gb(counts: dict, n_adapter: int, method: str) -> float:
    """
    Weights + gradients + optimizer state, in GB (1e9 bytes). Activations excluded:
    they depend on batch and sequence, and LoRA does NOT remove them (the backward
    pass still runs through every frozen layer to reach the adapters).
      full  : every parameter trained, 16 B each
      lora  : frozen base in bf16 (2 B) + adapters at 16 B
      qlora : block Linears in NF4, embeddings and norms in bf16, adapters at 16 B
    """
    total = counts["total"]
    if method == "full":
        b = total * TRAINED_BYTES
    elif method == "lora":
        b = total * 2 + n_adapter * TRAINED_BYTES
    elif method == "qlora":
        b = counts["linear"] * NF4_BITS / 8 + (counts["embed"] + counts["norms"]) * 2 \
            + n_adapter * TRAINED_BYTES
    else:
        raise ValueError(method)
    return b / 1e9
