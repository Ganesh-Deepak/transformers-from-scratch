"""
Reference code for Part I (Chapters 1-9): every number the foundation chapters
quote that is not already computed in tfs/model.py or tfs/tokenizer.py.

Grouped by chapter. Tested in tests/test_foundations.py.
"""

from __future__ import annotations

import codecs
import math
import re
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfs.model import Config, Model, RMSNorm, build_rope_cache, rotate_half


# ==================================================================== Ch 1


def repeat_kv_bytes(B, n_kv, n_rep, T, d_head, bytes_per=2):
    """K for one layer, before and after making n_kv heads look like n_kv*n_rep."""
    stored = B * n_kv * T * d_head * bytes_per
    copied = stored * n_rep
    return stored, copied


def expand_vs_repeat():
    """What expand and repeat actually do to memory. Used by Ch 1.5's worked example."""
    k = torch.arange(2 * 3 * 4.).view(1, 2, 3, 4)            # (B=1, n_kv=2, T=3, d=4)
    e = k[:, :, None].expand(1, 2, 4, 3, 4)                  # (1, 2, n_rep=4, 3, 4)  a VIEW
    r = k.repeat_interleave(4, dim=1)                        # (1, 8, 3, 4)            a COPY
    merged = e.reshape(1, 8, 3, 4)                           # (1, 8, 3, 4)            reshape of a stride-0 axis copies
    return {
        "k_stride": k.stride(),
        "expand_stride": e.stride(),
        "expand_shares_storage": e.data_ptr() == k.data_ptr(),
        "expand_storage_elems": e.untyped_storage().nbytes() // 4,
        "repeat_storage_elems": r.untyped_storage().nbytes() // 4,
        "merged_is_copy": merged.data_ptr() != k.data_ptr(),
        "same_values": torch.equal(merged, r),
    }


def inplace_errors():
    """The two autograd errors in-place ops produce. Returns their messages."""
    msgs = {}
    w = torch.randn(3, requires_grad=True)
    try:
        w.add_(1.0)
    except RuntimeError as e:
        msgs["leaf"] = str(e)
    x = torch.randn(3, requires_grad=True)
    y = x.sigmoid()                                          # backward of sigmoid needs its OUTPUT y
    y.mul_(2)                                                # ...which we just overwrote
    try:
        y.sum().backward()
    except RuntimeError as e:
        msgs["saved"] = str(e)
    x = torch.randn(4, requires_grad=True)
    h = x * 1.0
    view = h[:2]                                             # a view into h
    view.exp_()                                              # in-place on the view bumps h's version too
    msgs["view_version"] = h._version
    return msgs


def promotion_table():
    """torch.result_type for the cases Ch 1.6 lists. (Tensor, tensor) unless noted."""
    i64 = torch.tensor([1])
    f32 = torch.tensor([1.0])
    bf16 = torch.tensor([1.0], dtype=torch.bfloat16)
    f16 = torch.tensor([1.0], dtype=torch.float16)
    u8 = torch.tensor([1], dtype=torch.uint8)
    i8 = torch.tensor([1], dtype=torch.int8)
    bf16_scalar0d = torch.tensor(1.0, dtype=torch.float32)  # 0-dim tensor
    return {
        "int64 + float32": torch.result_type(i64, f32),
        "bf16 + float32": torch.result_type(bf16, f32),
        "bf16 + fp16": torch.result_type(bf16, f16),
        "bf16 + python float": torch.result_type(bf16, 2.5),
        "bf16 + 0-dim fp32 tensor": torch.result_type(bf16, bf16_scalar0d),
        "int64 + python float": torch.result_type(i64, 2.5),
        "uint8 + int8": torch.result_type(u8, i8),
        "int64 / int64": (i64 / i64).dtype,
        "int64 // int64": (i64 // i64).dtype,
    }


def llama3_8b_config():
    """Llama-3-8B's published config.json, in tfs.model.Config terms."""
    return Config(vocab_size=128256, d_model=4096, n_layer=32, n_head=32, n_kv_head=8,
                  d_head=128, d_ff=14336, max_T=8192, rope_base=500000.0, tie_embed=False)


def shape_trace(cfg=None, B=2, T=16):
    """
    Run a forward pass on the META device (no memory, no arithmetic: shapes only)
    and record the output shape of every submodule of layer 0, then the head.
    With the default this is Llama-3-8B's real forward pass, as shapes.
    """
    cfg = cfg or llama3_8b_config()
    with torch.device("meta"):
        m = Model(cfg)
    trace = []

    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            trace.append((name, tuple(inp[0].shape), tuple(o.shape)))
        return f

    b0 = m.blocks[0]
    for name, mod in [("embed", m.embed), ("blocks.0.n1", b0.n1),
                      ("blocks.0.attn.q_proj", b0.attn.q_proj), ("blocks.0.attn.k_proj", b0.attn.k_proj),
                      ("blocks.0.attn.v_proj", b0.attn.v_proj), ("blocks.0.attn.o_proj", b0.attn.o_proj),
                      ("blocks.0.n2", b0.n2), ("blocks.0.mlp.gate", b0.mlp.gate),
                      ("blocks.0.mlp.up", b0.mlp.up), ("blocks.0.mlp.down", b0.mlp.down),
                      ("norm", m.norm), ("head", m.head)]:
        mod.register_forward_hook(hook(name))
    ids = torch.zeros(B, T, dtype=torch.long, device="meta")
    with torch.no_grad():
        m(ids)
    return trace


# ==================================================================== Ch 2


def tile_utilization(n, tile):
    """Fraction of a tiled dimension doing useful work: n / (ceil(n/tile) * tile)."""
    return n / (math.ceil(n / tile) * tile)


def pad_to_multiple(n, m):
    return math.ceil(n / m) * m


def matmul_intensity(M, K, N, bytes_per=2):
    """(M,K) @ (K,N): FLOPs, bytes moved (read A, B, write C once), FLOP per byte."""
    flops = 2 * M * K * N
    bytes_ = bytes_per * (M * K + K * N + M * N)
    return flops, bytes_, flops / bytes_


def layer_flops_per_token(d, T, d_ff=None, causal=False):
    """
    Forward FLOPs per token for one standard (MHA, 4x GELU MLP) layer, from its matmuls.
    Projections Q,K,V,O: 4 d^2 weights; MLP: 2 d d_ff weights; each weight = 2 FLOPs/token.
    Attention scores QK^T and the AV mix: 2 T d each (half that on average if causal).
    """
    d_ff = d_ff or 4 * d
    proj = 2 * 4 * d * d
    mlp = 2 * 2 * d * d_ff
    attn = 2 * 2 * T * d * (0.5 if causal else 1.0)
    return {"proj": proj, "mlp": mlp, "attn": attn, "total": proj + mlp + attn}


# ==================================================================== Ch 3


class StreamingDetokenizer:
    """
    Emit text as tokens arrive without ever printing half a character.

    A byte-level token can end in the MIDDLE of a UTF-8 character (a Chinese
    character is 3 bytes, an emoji 4). Decoding each token on its own turns the
    fragments into U+FFFD. Keep the undecoded tail, emit only complete characters.
    """

    def __init__(self, tok):
        self.tok = tok
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id) -> str:
        return self.dec.decode(self.tok._vocab[token_id], final=False)

    def flush(self) -> str:
        return self.dec.decode(b"", final=True)


def naive_stream(tok, ids):
    """What a server that decodes token-by-token prints."""
    return [tok.decode([i]) for i in ids]


def careful_stream(tok, ids):
    s = StreamingDetokenizer(tok)
    return [s.push(i) for i in ids] + [s.flush()]


# GPT-4's cl100k_base pre-tokenizer, rewritten for Python's `re` (which has no \p{L}):
#   letters  [^\W\d_]      digits \d      "other" (not letter/digit/space)  (?:[^\s\w]|_)
CL100K_LIKE = re.compile(
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|(?:[^\r\n\w]|_)?[^\W\d_]+"
    r"|\d{1,3}"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)
# GPT-2's pattern (also rewritten for `re`): digits are NOT grouped in threes.
GPT2_LIKE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
)


def pretokenize(text, pattern=CL100K_LIKE):
    return pattern.findall(text)


def nfkc_examples():
    """NFKC folds compatibility characters together. It cannot be undone."""
    cases = ["ﬁle", "Ｈｅｌｌｏ", "x²", "①②", "½", "ℌ", "café"]
    return [(c, unicodedata.normalize("NFKC", c)) for c in cases]


def embedding_params(V, d, tied):
    return V * d * (1 if tied else 2)


def toy_vocab_sweep(text, sizes, d_model=4096):
    """
    Train the repo's byte-level BPE at several vocabulary sizes on the same text and
    measure bytes/token. Returns rows (V, bytes_per_token, tokens, embedding params at d_model).
    """
    from tfs.tokenizer import BPETokenizer
    rows = []
    for V in sizes:
        tok = BPETokenizer().train(text, V)
        n = len(tok.encode(text, allow_special=False))
        rows.append((tok.vocab_size, len(text.encode("utf-8")) / n, n, tok.vocab_size * d_model))
    return rows


def fertility(tok, text):
    """Tokens per whitespace-separated word, and UTF-8 bytes per character."""
    n_tok = len(tok.encode(text, allow_special=False))
    words = len(text.split()) or 1
    return n_tok / words, len(text.encode("utf-8")) / len(text)


def token_heal(tok, prompt):
    """
    Token healing (Microsoft guidance): drop the prompt's last token and require the
    first generated token to START WITH the text that was dropped. Returns
    (trimmed ids, the required prefix string, the vocabulary ids allowed first).
    """
    ids = tok.encode(prompt, allow_special=False)
    last = tok._vocab[ids[-1]]
    allowed = [i for i, b in tok._vocab.items() if b.startswith(last) and i not in tok.special.values()]
    return ids[:-1], last.decode("utf-8", errors="replace"), allowed


# ==================================================================== Ch 4


@torch.no_grad()
def residual_norms(cfg, T=64, seed=0, scaled_init=True):
    """RMS of the residual stream after the embedding and after each block, at init."""
    torch.manual_seed(seed)
    m = Model(cfg).eval()
    if not scaled_init:
        for name, p in m.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02)
    ids = torch.randint(0, cfg.vocab_size, (4, T))
    x = m.embed(ids)
    cos, sin = m.cos[:T], m.sin[:T]
    out = [x.pow(2).mean().sqrt().item()]
    for blk in m.blocks:
        x, _ = blk(x, cos, sin)
        out.append(x.pow(2).mean().sqrt().item())
    return out


def residual_variance_model(n_layers, var0=1.0, var_f=1.0):
    """If every block adds an independent update of variance var_f, the stream's RMS is sqrt(var0 + l var_f)."""
    return [math.sqrt(var0 + l * var_f) for l in range(n_layers + 1)]


def softcap(x, cap):
    """Gemma 2: logits <- cap * tanh(logits / cap). Smooth, bounded by +-cap."""
    return cap * torch.tanh(torch.as_tensor(x, dtype=torch.float64) / cap)


def z_loss(logits, coef=1e-4):
    """PaLM's auxiliary loss on the output softmax: coef * (log Z)^2, pulls log Z toward 0."""
    logz = torch.logsumexp(torch.as_tensor(logits, dtype=torch.float64), -1)
    return coef * logz.pow(2), logz


def superposition_toy(n_features=5, d_hidden=2, sparsity=0.9, steps=4000, seed=0):
    """
    Elhage et al. 2022, 'Toy Models of Superposition', the 5-features-in-2-dimensions case.
    x has n features, each zero with probability `sparsity`, else U(0,1); importance 1.
    Model: x_hat = ReLU(W^T W x + b), W: (d_hidden, n_features). Returns the column norms
    of W (a feature is "represented" when its norm is near 1) and the final loss.
    """
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(d_hidden, n_features, generator=g) * 0.1).requires_grad_()
    b = torch.zeros(n_features, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        x = torch.rand(1024, n_features, generator=g)
        x = x * (torch.rand(1024, n_features, generator=g) >= sparsity)
        xh = F.relu(x @ W.T @ W + b)                         # (1024, n)
        loss = (xh - x).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    norms = W.detach().norm(dim=0)
    return norms.tolist(), loss.item(), W.detach()


# ---- a tiny trained model, shared by Ch 4 (logit lens) and Ch 7 (length extrapolation)

TOY_CFG = Config(vocab_size=32, d_model=64, n_layer=2, n_head=4, n_kv_head=4, d_head=16,
                 d_ff=176, max_T=512)
_TOY = {}


def copy_batch(B, T, vocab, g):
    """First half random tokens 2..vocab-1, second half repeats it. Token 1 separates."""
    half = T // 2
    a = torch.randint(2, vocab, (B, half), generator=g)
    return torch.cat([a, a], dim=1)                          # (B, T)


def train_toy_copy_model(T=64, steps=300, seed=0, batch=16):
    """
    Train TOY_CFG (2 layers, RoPE) to copy: the second half of every sequence repeats
    the first, so after the midpoint the next token is predictable ONLY by attention
    to the right earlier position (an induction head). Cached per (T, steps, seed).
    """
    key = (T, steps, seed)
    if key in _TOY:
        return _TOY[key]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = Model(TOY_CFG)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(steps):
        ids = copy_batch(batch, T, TOY_CFG.vocab_size, g)
        _, loss, _ = m(ids, targets=ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    _TOY[key] = m
    return m


@torch.no_grad()
def copy_loss(m, T, n=64, seed=123):
    """Mean loss on the predictable second half (positions T/2 .. T-2 predicting T/2+1 .. T-1)."""
    g = torch.Generator().manual_seed(seed)
    ids = copy_batch(n, T, TOY_CFG.vocab_size, g)
    logits, _, _ = m(ids)
    half = T // 2
    lp = logits[:, half:-1].log_softmax(-1)                 # (n, half-1, V)
    tgt = ids[:, half + 1:]
    return -lp.gather(-1, tgt[..., None]).mean().item()


def periodic_batch(B, T, period, vocab, g):
    """A random block of `period` tokens, repeated until the sequence is T long."""
    a = torch.randint(2, vocab, (B, period), generator=g)
    return a.repeat(1, math.ceil(T / period))[:, :T]       # (B, T)


@torch.no_grad()
def periodic_loss(m, T, period=32, n=64, seed=321):
    """
    Loss on every predictable position (t >= period) when the copy distance stays at
    the TRAINED value but absolute positions run past the training length.
    Returns (loss on positions < 64, loss on positions >= 64).
    """
    g = torch.Generator().manual_seed(seed)
    ids = periodic_batch(n, T, period, TOY_CFG.vocab_size, g)
    logits, _, _ = m(ids)
    lp = logits[:, :-1].log_softmax(-1)
    nll = -lp.gather(-1, ids[:, 1:, None])[..., 0]          # (n, T-1): predicting position t+1
    pos = torch.arange(1, T)
    inside = (pos >= period) & (pos < 64)
    outside = pos >= 64
    out_l = nll[:, outside].mean().item() if outside.any() else float("nan")
    return nll[:, inside].mean().item(), out_l


@torch.no_grad()
def periodic_loss_interpolated(m, T, period=32, train_T=64, n=64, seed=321):
    """
    Position interpolation (Chen et al. 2023) without retraining: positions are scaled by
    train_T / T so no rotation angle exceeds the trained range. Returns the loss on
    predictable positions >= 64, as periodic_loss does.
    """
    g = torch.Generator().manual_seed(seed)
    ids = periodic_batch(n, T, period, TOY_CFG.vocab_size, g)
    d = TOY_CFG.d_head
    inv = 1.0 / (TOY_CFG.rope_base ** (torch.arange(0, d, 2).float() / d))
    pos = torch.arange(T).float() * (train_T / T)
    fr = torch.outer(pos, inv)
    emb = torch.cat([fr, fr], -1)
    cos, sin = emb.cos(), emb.sin()
    x = m.embed(ids)
    for blk in m.blocks:
        x, _ = blk(x, cos, sin)
    logits = m.head(m.norm(x))
    nll = -logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
    p = torch.arange(1, T)
    return nll[:, p >= 64].mean().item()


@torch.no_grad()
def logit_lens(m, T=64, n=64, seed=123):
    """
    Decode the residual stream after the embedding and after each block with the
    model's own final norm + unembedding. Returns, per layer, the accuracy of the
    top-1 guess and the mean probability of the correct token, on the copy half.
    """
    g = torch.Generator().manual_seed(seed)
    ids = copy_batch(n, T, TOY_CFG.vocab_size, g)
    x = m.embed(ids)
    cos, sin = m.cos[:T], m.sin[:T]
    streams = [x]
    for blk in m.blocks:
        x, _ = blk(x, cos, sin)
        streams.append(x)
    half = T // 2
    tgt = ids[:, half + 1:]
    rows = []
    for s in streams:
        p = m.head(m.norm(s))[:, half:-1].softmax(-1)       # (n, half-1, V)
        acc = (p.argmax(-1) == tgt).float().mean().item()
        pc = p.gather(-1, tgt[..., None]).mean().item()
        rows.append((acc, pc))
    return rows


# ==================================================================== Ch 5


def naive_softmax(x):
    e = torch.exp(x)
    return e / e.sum(-1, keepdim=True)


def stable_softmax(x):
    """Subtract the row max first: the largest exponent becomes exp(0) = 1, nothing overflows."""
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    return e / e.sum(-1, keepdim=True)


def softmax_with_sink(scores, sink_logit):
    """
    Softmax with one extra logit that has no value attached (gpt-oss's learned
    per-head sink; sink_logit = 0 is Miller's "softmax1" / off-by-one softmax).
    The weights on real tokens now sum to LESS than 1: the head can attend to nothing.
    """
    s = torch.as_tensor(scores, dtype=torch.float64)
    z = torch.cat([s, torch.tensor([float(sink_logit)], dtype=torch.float64)])
    return z.softmax(-1)[:-1]


def masked_attention(q, k, v, mask):
    """Plain attention with a boolean keep-mask (True = may attend). Shapes (T, d)."""
    s = (q @ k.T) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask, float("-inf"))
    return s.softmax(-1) @ v


def masked_attention_safe(q, k, v, mask):
    """Same, but rows with nothing to attend to output zeros instead of NaN."""
    s = (q @ k.T) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask, torch.finfo(s.dtype).min)       # finite: softmax stays defined
    w = s.softmax(-1) * mask.any(-1, keepdim=True)           # zero the empty rows
    return w @ v


def padding_masks(lengths, T, side):
    """Key-padding + causal masks for a batch padded on the left or the right. (B, T, T)."""
    B = len(lengths)
    real = torch.zeros(B, T, dtype=torch.bool)
    for b, n in enumerate(lengths):
        if side == "left":
            real[b, T - n:] = True
        else:
            real[b, :n] = True
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    return real, causal[None] & real[:, None, :]


def qk_norm_logit_bound(d_head, g_q=1.0, g_k=1.0):
    """
    With RMSNorm on q and k, |q_hat| = sqrt(d) * |g| (for a uniform gain g), so the
    scaled logit q_hat.k_hat / sqrt(d) is at most sqrt(d) * g_q * g_k, whatever the weights do.
    """
    return math.sqrt(d_head) * g_q * g_k


@torch.no_grad()
def all_position_loss_check(cfg=None, T=12, seed=0):
    """
    One forward pass yields T-1 predictions. Return (model loss, the mean of the T-1
    per-position losses computed one prefix at a time). They must be equal.
    """
    cfg = cfg or Config(vocab_size=50, d_model=32, n_layer=2, n_head=4, n_kv_head=2,
                        d_head=8, d_ff=64, max_T=32)
    torch.manual_seed(seed)
    m = Model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    _, loss, _ = m(ids, targets=ids)
    per = []
    for t in range(1, T):
        logits, _, _ = m(ids[:, :t])                         # only the prefix
        per.append(-logits[0, -1].log_softmax(-1)[ids[0, t]].item())
    return loss.item(), sum(per) / len(per)


# ==================================================================== Ch 6


def split_fused_qkv(W, n_head, d_head, layout):
    """
    A fused QKV weight (3*n_head*d_head, d_model), split into Wq, Wk, Wv.
      "blocked":   rows are [all Q heads | all K heads | all V heads]  (GPT-2 c_attn, many repos)
      "per_head":  rows are [q0 k0 v0 | q1 k1 v1 | ...]                (GPT-NeoX query_key_value)
    """
    C = W.shape[1]
    if layout == "blocked":
        return W.split(n_head * d_head, dim=0)
    w = W.view(n_head, 3, d_head, C)                         # (H, 3, dh, C)
    return tuple(w[:, i].reshape(n_head * d_head, C) for i in range(3))


def per_head_to_blocked(W, n_head, d_head):
    q, k, v = split_fused_qkv(W, n_head, d_head, "per_head")
    return torch.cat([q, k, v], dim=0)


def qk_ov_circuits(Wq, Wk, Wv, Wo, n_head, d_head):
    """
    Per-head bilinear forms (Elhage et al. 2021). With row-vector convention x @ W:
      W_QK^h = Wq^h Wk^h^T   (C, C): score(x_i, x_j) = x_i W_QK^h x_j^T / sqrt(d_head)
      W_OV^h = Wv^h Wo^h     (C, C): what head h writes, given what it reads
    Wq, Wk, Wv: (C, H*dh); Wo: (H*dh, C). Each circuit has rank <= d_head.
    """
    out = []
    for h in range(n_head):
        sl = slice(h * d_head, (h + 1) * d_head)
        out.append((Wq[:, sl] @ Wk[:, sl].T, Wv[:, sl] @ Wo[sl, :]))
    return out


# ==================================================================== Ch 7


def apply_partial_rope(x, pos, rot_dim, base=10000.0):
    """
    Rotate only the first rot_dim channels of each head (GPT-NeoX/Pythia rotary_pct,
    Phi-2 partial_rotary_factor); the rest pass through untouched. x: (T, d_head).
    """
    cos, sin = build_rope_cache(int(max(pos)) + 1, rot_dim, base)
    cos, sin = cos[pos], sin[pos]                            # (T, rot_dim)
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    xr = xr * cos + rotate_half(xr) * sin
    return torch.cat([xr, xp], dim=-1)


def mrope_angles(t, h, w, d_head=128, sections=(16, 24, 24), base=10000.0):
    """
    Qwen2-VL's multimodal RoPE: the d_head/2 frequencies are split into sections
    (temporal, height, width); each section rotates by its own position id.
    Returns the (d_head/2,) rotation angles for one token at position (t, h, w).
    """
    assert sum(sections) == d_head // 2
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float64) / d_head))
    pos = torch.cat([torch.full((n,), float(p), dtype=torch.float64)
                     for n, p in zip(sections, (t, h, w))])
    return pos * inv


# ==================================================================== Ch 8


def softmax_saturation(logit_scale, n=64, seed=0):
    """
    Attention logits z * scale for one query over n keys, z ~ N(0, 1).
    Returns (entropy in nats, max weight, norm of the softmax Jacobian applied to a
    unit upstream gradient, averaged over directions) as the scale grows.
    Weight growth multiplies q and k, so logits grow with the SQUARE of the weight scale.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, generator=g, dtype=torch.float64) * logit_scale
    p = z.softmax(-1)
    ent = -(p * p.clamp_min(1e-300).log()).sum().item()
    J = torch.diag(p) - torch.outer(p, p)                    # softmax Jacobian
    jac = torch.linalg.matrix_norm(J, ord="fro").item() / math.sqrt(n)
    return ent, p.max().item(), jac


def activation_bytes_per_layer(s, b, h, a, flash=False):
    """
    Korthikanti et al. 2022, eq. (1): activations a GPT-style layer saves for backward,
    16-bit, no recomputation, no parallelism:  s b h (34 + 5 a s / h) bytes.
    The 5 a s^2 b term is the attention score/softmax/dropout matrices; FlashAttention
    does not store them (it keeps only the output and the log-sum-exp), so drop it.
    """
    base = s * b * h * 34
    scores = 0 if flash else 5 * a * s * s * b
    return base + scores


def block_flops_per_token(d, d_ff, T, n_head, n_kv_head, d_head, swiglu=True, causal=True):
    """Forward FLOPs per token for one modern block (GQA + SwiGLU), by component."""
    q_o = 2 * 2 * d * n_head * d_head
    kv = 2 * 2 * d * n_kv_head * d_head
    mlp = 2 * (3 if swiglu else 2) * d * d_ff
    attn = 2 * 2 * T * n_head * d_head * (0.5 if causal else 1.0)
    return {"q,o proj": q_o, "k,v proj": kv, "mlp": mlp, "scores+mix": attn,
            "total": q_o + kv + mlp + attn}


class ParallelBlock(nn.Module):
    """GPT-J / PaLM: x + attn(LN(x)) + mlp(LN(x)) — both read the SAME input."""

    def __init__(self, cfg):
        super().__init__()
        from tfs.model import GroupedQueryAttention, SwiGLU
        self.n = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        h = self.n(x)
        a, _ = self.attn(h, cos, sin)
        return x + a + self.mlp(h)


# ==================================================================== Ch 9


def encdec_vs_decoder_flops(N_dec_only, N_enc, N_dec, S, T):
    """
    Forward FLOPs (2 per parameter per token, attention scores ignored) to read an
    S-token input and produce T output tokens.
      decoder-only, N params:  every one of S + T tokens passes through all N
      encoder-decoder:         S tokens through the encoder, T through the decoder
    """
    dec_only = 2 * N_dec_only * (S + T)
    enc_dec = 2 * N_enc * S + 2 * N_dec * T
    return dec_only, enc_dec


def info_nce(queries, docs, temperature=0.05):
    """
    Contrastive loss for embedding models with in-batch negatives: row i of `docs`
    is the positive for query i, every other row is a negative.
    Inputs (B, d); returns (loss, the (B, B) cosine-similarity matrix).
    """
    q = F.normalize(torch.as_tensor(queries, dtype=torch.float64), dim=-1)
    d = F.normalize(torch.as_tensor(docs, dtype=torch.float64), dim=-1)
    sim = q @ d.T                                            # (B, B)
    loss = F.cross_entropy(sim / temperature, torch.arange(sim.shape[0]))
    return loss.item(), sim


# The same sentence in eight languages (Ch 3.5's fertility table). Translations by the
# course author; what is measured is the TOKENIZER, so small wording choices barely matter.
PARALLEL = {
    "English": "The baker opens the shop before the sun is up and lights the oven first.",
    "French": "La boulangère ouvre la boutique avant le lever du soleil et allume d'abord le four.",
    "German": "Die Bäckerin öffnet den Laden vor Sonnenaufgang und heizt zuerst den Ofen an.",
    "Spanish": "La panadera abre la tienda antes de que salga el sol y enciende primero el horno.",
    "Russian": "Пекарь открывает лавку до восхода солнца и первым делом разжигает печь.",
    "Hindi": "बेकर सूरज उगने से पहले दुकान खोलती है और सबसे पहले ओवन जलाती है।",
    "Chinese": "面包师在太阳升起之前打开店门，先点燃烤炉。",
    "Japanese": "パン屋は日が昇る前に店を開け、まずオーブンに火を入れる。",
}


def corpus_text():
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tests", "fixtures", "ch03_corpus.txt")
    with open(p, encoding="utf-8") as f:
        return f.read()


def fertility_table(vocab_size=1024):
    """Train BPE on the English corpus fixture, then tokenize the same sentence in 8 languages."""
    from tfs.tokenizer import BPETokenizer
    tok = BPETokenizer().train(corpus_text(), vocab_size)
    base = len(tok.encode(PARALLEL["English"], allow_special=False))
    rows = []
    for lang, s in PARALLEL.items():
        n = len(tok.encode(s, allow_special=False))
        rows.append((lang, len(s), len(s.encode("utf-8")), n, n / base))
    return rows


def flops_per_character(n_body, V, d, chars_per_token):
    """
    Forward FLOPs to process one CHARACTER of text: (2 x body params + 2 V d for the
    output projection) per token, divided by how many characters a token covers.
    The input embedding is a lookup and costs no FLOPs (Ch 4). Attention's T^2 term
    is ignored here (it would favour the larger vocabulary further).
    """
    return (2 * n_body + 2 * V * d) / chars_per_token


def llama3_8b_body_params():
    cfg = llama3_8b_config()
    return cfg.n_params() - 2 * cfg.vocab_size * cfg.d_model
