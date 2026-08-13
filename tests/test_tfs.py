"""
The verification suite. Every test corresponds to a claim made in the course.

    python tests/test_tfs.py          # no pytest needed
    python -m pytest tests/ -v        # if you have pytest
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfs.model import (Config, Model, build_rope_cache, rotate_half, apply_rope,
                       RMSNorm, sample)
from tfs.tokenizer import BPETokenizer

CFG = Config(vocab_size=512, d_model=96, n_layer=3, n_head=4, n_kv_head=2,
             d_head=24, d_ff=256, max_T=128)


# ==================================================== Ch 3 — tokenizer

def test_tokenizer_roundtrip():
    corpus = ("the quick brown fox jumps over the lazy dog. " * 40 +
              "attention is all you need. transformers are models. " * 40)
    tok = BPETokenizer().train(corpus, vocab_size=400)
    for s in ["the quick brown fox", "attention", "", "x",
              "unseen WORDS with CAPS 123!", "🤖 émoji ñ 中文", "\n\t  "]:
        assert tok.decode(tok.encode(s)) == s, f"roundtrip failed for {s!r}"


def test_tokenizer_no_unknown_tokens():
    """Byte-level BPE can encode ANY string — Ch 3.2."""
    tok = BPETokenizer().train("hello world " * 50, vocab_size=300)
    weird = bytes(range(256)).decode("latin-1")
    assert tok.decode(tok.encode(weird)) == weird


def test_tokenizer_compresses():
    corpus = open(__file__, encoding="utf-8").read() * 3
    tok = BPETokenizer().train(corpus, vocab_size=800)
    assert tok.compression_ratio(corpus) > 2.0


def test_tokenizer_special_tokens():
    tok = BPETokenizer().train("hello world " * 50, vocab_size=300)
    tok.add_special_tokens(["<|bos|>", "<|eos|>"])
    ids = tok.encode("<|bos|>hello<|eos|>")
    assert ids[0] == tok.special["<|bos|>"] and ids[-1] == tok.special["<|eos|>"]
    assert tok.decode(ids) == "<|bos|>hello<|eos|>"


# ==================================================== Ch 7 — RoPE

def test_rope_is_relative():
    """<R_m q, R_n k> depends only on (n-m) — Ch 7.4, the whole point of RoPE."""
    d = 32
    cos, sin = build_rope_cache(200, d)
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)

    def score(m, n):
        qm = q * cos[m] + rotate_half(q) * sin[m]
        kn = k * cos[n] + rotate_half(k) * sin[n]
        return (qm * kn).sum().item()

    base = score(5, 8)
    for m, n in [(50, 53), (100, 103), (0, 3), (150, 153)]:
        assert abs(score(m, n) - base) < 1e-4, f"offset +3 differs at ({m},{n})"
    assert abs(score(5, 9) - base) > 1e-3, "different offset should differ"


def test_rope_preserves_norm():
    """Rotation is orthogonal — it must not change vector magnitude."""
    d = 64
    cos, sin = build_rope_cache(50, d)
    q = torch.randn(1, 1, 50, d)
    k = torch.randn(1, 1, 50, d)
    qr, _ = apply_rope(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4)


# ==================================================== Ch 8 — norms

def test_rmsnorm_unit_rms():
    n = RMSNorm(64)
    x = torch.randn(4, 10, 64) * 7.5 + 3.0
    y = n(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_does_not_center():
    """Unlike LayerNorm, RMSNorm leaves the mean alone — Ch 8.2."""
    n = RMSNorm(64)
    x = torch.randn(2, 5, 64) + 10.0        # large positive offset
    assert n(x).mean(-1).abs().min() > 0.1  # would be ~0 for LayerNorm


# ==================================================== Ch 9 — the model

def test_shapes():
    m = Model(CFG)
    ids = torch.randint(0, CFG.vocab_size, (2, 16))
    logits, loss, _ = m(ids, targets=ids)
    assert logits.shape == (2, 16, CFG.vocab_size)
    assert loss.ndim == 0


def test_initial_loss_is_ln_vocab():
    """An untrained model should be ~uniform: loss = ln(V) — Ch 9.3."""
    torch.manual_seed(0)
    m = Model(CFG)
    ids = torch.randint(0, CFG.vocab_size, (8, 32))
    _, loss, _ = m(ids, targets=ids)
    expected = math.log(CFG.vocab_size)
    assert abs(loss.item() - expected) < 0.3, f"{loss.item():.3f} vs ln(V)={expected:.3f}"


def test_causality():
    """Perturbing token t must not change outputs at positions < t — Ch 9.5."""
    torch.manual_seed(0)
    m = Model(CFG).eval()
    ids = torch.randint(0, CFG.vocab_size, (1, 12))
    with torch.no_grad():
        out1, _, _ = m(ids)
        ids2 = ids.clone()
        ids2[0, 7] = (ids2[0, 7] + 1) % CFG.vocab_size
        out2, _, _ = m(ids2)
    assert torch.allclose(out1[0, :7], out2[0, :7], atol=1e-5), "CAUSALITY VIOLATED"
    assert not torch.allclose(out1[0, 7], out2[0, 7], atol=1e-5)


def test_all_params_get_gradients():
    m = Model(CFG)
    ids = torch.randint(0, CFG.vocab_size, (2, 16))
    _, loss, _ = m(ids, targets=ids)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert p.grad.abs().sum() > 0, f"{name} has zero grad"


def test_param_count_matches_formula():
    """The analytic count from Ch 9.4 must match reality."""
    for cfg in [CFG,
                Config(vocab_size=1000, d_model=128, n_layer=4, n_head=8,
                       n_kv_head=8, d_head=16, d_ff=340, max_T=64),
                Config(vocab_size=256, d_model=64, n_layer=2, n_head=4,
                       n_kv_head=1, d_head=16, d_ff=170, max_T=32, tie_embed=False)]:
        m = Model(cfg)
        actual = sum(p.numel() for p in m.parameters())
        assert cfg.n_params() == actual, f"formula {cfg.n_params()} != actual {actual}"


# ==================================================== Ch 13 — GQA

def test_gqa_reduces_to_mha():
    """n_kv_head == n_head must be plain MHA."""
    cfg = Config(vocab_size=128, d_model=64, n_layer=1, n_head=4, n_kv_head=4,
                 d_head=16, d_ff=128, max_T=32)
    m = Model(cfg).eval()
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        logits, _, _ = m(ids)
    assert logits.shape == (1, 8, 128)


def test_gqa_cache_is_smaller():
    """The cache holds n_kv_head heads, not n_head — Ch 13.3."""
    m = Model(CFG).eval()
    ids = torch.randint(0, CFG.vocab_size, (1, 10))
    with torch.no_grad():
        _, _, caches = m(ids)
    k, v = caches[0]
    assert k.shape == (1, CFG.n_kv_head, 10, CFG.d_head), k.shape
    assert v.shape == (1, CFG.n_kv_head, 10, CFG.d_head)


def test_gqa_grouping_is_contiguous():
    """repeat_interleave, not repeat — Ch 13.3's gotcha box."""
    x = torch.arange(3).view(1, 3, 1, 1).float()          # 3 "kv heads"
    got = x.repeat_interleave(2, dim=1).flatten().tolist()
    assert got == [0, 0, 1, 1, 2, 2], got
    wrong = x.repeat(1, 2, 1, 1).flatten().tolist()
    assert wrong == [0, 1, 2, 0, 1, 2]                     # the bug, for contrast


# ==================================================== Ch 12 — KV cache

def test_kv_cache_matches_no_cache():
    """
    THE critical test. Cached and uncached greedy generation must be
    token-identical — Ch 12.2. Catches the RoPE-offset and is_causal bugs.
    """
    torch.manual_seed(0)
    m = Model(CFG).eval()
    prompt = torch.randint(0, CFG.vocab_size, (1, 5))

    a = m.generate(prompt.clone(), max_new_tokens=25, temperature=0, use_cache=False)
    b = m.generate(prompt.clone(), max_new_tokens=25, temperature=0, use_cache=True)
    assert torch.equal(a, b), (
        f"cache changes output!\n  no-cache {a.tolist()}\n  cache    {b.tolist()}"
    )


def test_incremental_forward_matches_full():
    """Feeding tokens one at a time with a cache == feeding them all at once."""
    torch.manual_seed(0)
    m = Model(CFG).eval()
    ids = torch.randint(0, CFG.vocab_size, (1, 9))
    with torch.no_grad():
        full, _, _ = m(ids)
        caches, outs = None, []
        for t in range(ids.size(1)):
            lg, _, caches = m(ids[:, t:t + 1], caches=caches, pos_offset=t)
            outs.append(lg[:, -1])
        inc = torch.stack(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max().item()


# ==================================================== Ch 11 — sampling

def test_temperature_zero_is_greedy():
    logits = torch.randn(4, 100)
    got = sample(logits, temperature=0)
    assert torch.equal(got, logits.argmax(-1, keepdim=True))


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.randn(1, 100)
    allowed = set(logits.topk(5, dim=-1).indices[0].tolist())
    for _ in range(200):
        assert sample(logits, temperature=1.0, top_k=5).item() in allowed


def test_top_p_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0, -5.0]])
    seen = {sample(logits, temperature=1.0, top_p=0.9).item() for _ in range(300)}
    assert seen <= {0, 1}, seen        # top two hold >90% of the mass


# ==================================================== Ch 14 — online softmax

def test_online_softmax_is_exact():
    """FlashAttention's core identity — Ch 14.3."""
    torch.manual_seed(0)
    for scale in [1.0, 50.0, 200.0]:
        xs = torch.randn(257) * scale
        m, l = float("-inf"), 0.0
        for i in range(0, len(xs), 16):
            blk = xs[i:i + 16]
            m_new = max(m, blk.max().item())
            l = l * math.exp(m - m_new) + torch.exp(blk - m_new).sum().item()
            m = m_new
        streaming = m + math.log(l)
        assert abs(streaming - torch.logsumexp(xs, 0).item()) < 1e-4


def test_tiled_attention_matches_naive():
    """A pure-PyTorch FlashAttention must be numerically identical — Ch 14 Ex 2."""
    torch.manual_seed(0)
    N, d = 48, 16
    Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)

    # naive
    S = Q @ K.T / math.sqrt(d)
    S = S.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool), 1), float("-inf"))
    naive = S.softmax(-1) @ V

    # tiled
    Br = Bc = 16
    O = torch.zeros(N, d)
    for i in range(0, N, Br):
        Qi = Q[i:i + Br]
        Oi = torch.zeros(Qi.size(0), d)
        mi = torch.full((Qi.size(0),), float("-inf"))
        li = torch.zeros(Qi.size(0))
        for j in range(0, N, Bc):
            if j > i + Qi.size(0) - 1:
                break                                    # fully masked block
            Kj, Vj = K[j:j + Bc], V[j:j + Bc]
            Sij = Qi @ Kj.T / math.sqrt(d)
            qi = torch.arange(i, i + Qi.size(0))[:, None]
            kj = torch.arange(j, j + Kj.size(0))[None, :]
            Sij = Sij.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(mi, Sij.max(dim=-1).values)
            corr = torch.exp(mi - m_new)
            Pij = torch.exp(Sij - m_new[:, None])
            li = corr * li + Pij.sum(-1)
            Oi = corr[:, None] * Oi + Pij @ Vj
            mi = m_new
        O[i:i + Br] = Oi / li[:, None]

    assert torch.allclose(O, naive, atol=1e-5), (O - naive).abs().max().item()


# ==================================================== Ch 5 — attention facts

def test_attention_rows_sum_to_one():
    torch.manual_seed(0)
    T, d = 12, 16
    q, k = torch.randn(T, d), torch.randn(T, d)
    s = q @ k.T / math.sqrt(d)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    w = s.softmax(-1)
    assert torch.allclose(w.sum(-1), torch.ones(T), atol=1e-6)
    assert (w.triu(1).abs() < 1e-9).all(), "masked positions must be exactly zero"


def test_sqrt_dk_scaling_claim():
    """std(q·k) == sqrt(d_k) for unit-variance inputs — Ch 5.4."""
    torch.manual_seed(0)
    for d in [16, 64, 256]:
        q, k = torch.randn(20000, d), torch.randn(20000, d)
        emp = (q * k).sum(-1).std().item()
        assert abs(emp - math.sqrt(d)) / math.sqrt(d) < 0.05, (d, emp, math.sqrt(d))


# ============================ Ch 9/10 — END-TO-END: the test that catches
#                                        label-shift bugs the loss curve hides

def test_generation_reproduces_memorised_text():
    """
    THE regression test for double-shifted labels.

    A model that has memorised a short text (train loss ~0) must, under greedy
    decoding, reproduce it. If the training loop shifts labels twice, the model
    learns to predict TWO tokens ahead: the loss curve still looks healthy,
    teacher-forced output is off by exactly one token, and free-running
    generation collapses into noise.

    Unit tests on forward() cannot catch this -- only closing the loop can.
    """
    from tfs.tokenizer import CharTokenizer
    from tfs.train import get_batch

    text = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
            "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather "
            "to die than to famish?\n\nAll:\nResolved. resolved.\n") * 3
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    cfg = Config(vocab_size=tok.vocab_size, d_model=128, n_layer=4, n_head=4,
                 n_kv_head=2, d_head=32, d_ff=256, max_T=192)
    torch.manual_seed(0)
    model = Model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95))

    T = 64
    for _ in range(400):
        x = get_batch(data, 16, T)
        _, loss, _ = model(x, targets=x)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()

    assert loss.item() < 0.35, f"did not memorise (loss {loss.item():.3f})"

    # Greedy-continue a real prefix; it must match the true text closely.
    P = 20
    prompt = data[:P][None]
    out = model.generate(prompt.clone(), 45, temperature=0, use_cache=True)
    got = out[0, P:P + 45].tolist()
    want = data[P:P + 45].tolist()
    match = sum(a == b for a, b in zip(got, want)) / len(want)

    assert match > 0.7, (
        f"memorised model cannot reproduce its own training text "
        f"(only {match:.0%} of characters match).\n"
        f"  got  {tok.decode(got)!r}\n"
        f"  want {tok.decode(want)!r}\n"
        f"  -> suspect DOUBLE-SHIFTED LABELS in the training loop."
    )


# ==================================================== Ch 10 — loss units

def test_bits_per_byte_normalises_across_tokenizers():
    """
    Ch 10.1: raw cross-entropy is per-TOKEN and so is NOT comparable across
    vocabularies; bits-per-byte is. This tests the PROPERTY, not any particular
    measured run (measurements go stale; the maths does not).
    """
    from tfs.train import bits_per_byte

    # ln(2) nats is exactly 1 bit; over 1 byte/token that is 1.0 bits/byte.
    assert abs(bits_per_byte(math.log(2), 1.0) - 1.0) < 1e-9

    # Two tokenizers that compress the SAME text equally well must report the
    # same bpb even though their per-token losses differ by the compression ratio.
    char_loss, char_bpt = 1.5115, 1.0             # measured char-level run
    ratio = 3.29                                   # a BPE tokenizer's bytes/token
    equiv_bpe_loss = char_loss * ratio             # equally good, per byte
    assert abs(bits_per_byte(char_loss, char_bpt)
               - bits_per_byte(equiv_bpe_loss, ratio)) < 1e-9

    # Sanity: the measured char-level run lands near the ~2.1 bits/byte that a
    # good small character model on English reaches.
    assert 2.0 < bits_per_byte(char_loss, char_bpt) < 2.4


def test_char_tokenizer_roundtrip():
    from tfs.tokenizer import CharTokenizer
    text = "Hello, world!\nSecond line.\t(tabs too)"
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == len(set(text))
    assert tok.compression_ratio(text) == 1.0


# ==================================================== Ch 20 — DPO

def test_dpo_loss_properties():
    from tfs.dpo import dpo_loss
    # Perfectly ranked pair with a big margin -> loss near 0
    loss, _, _, acc = dpo_loss(torch.tensor([0.0]), torch.tensor([-20.0]),
                               torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0)
    assert loss.item() < 1e-6 and acc.item() == 1.0
    # Reversed -> large loss, accuracy 0
    loss, _, _, acc = dpo_loss(torch.tensor([-20.0]), torch.tensor([0.0]),
                               torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0)
    assert loss.item() > 10 and acc.item() == 0.0
    # Identical to reference -> loss = -log(0.5) = ln 2
    loss, _, _, _ = dpo_loss(torch.tensor([-3.0]), torch.tensor([-5.0]),
                             torch.tensor([-3.0]), torch.tensor([-5.0]), beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6


def test_dpo_partition_function_cancels():
    """
    Adding any prompt-level constant to BOTH sequences' log-probs must not
    change the loss. That is the Z(x) cancellation of Ch 20.4, empirically.
    """
    from tfs.dpo import dpo_loss
    pc, pr = torch.tensor([-3.0]), torch.tensor([-7.0])
    rc, rr = torch.tensor([-4.0]), torch.tensor([-6.0])
    base, *_ = dpo_loss(pc, pr, rc, rr, beta=0.1)
    for c in [1.0, -5.0, 100.0]:
        shifted, *_ = dpo_loss(pc + c, pr + c, rc + c, rr + c, beta=0.1)
        assert abs(base.item() - shifted.item()) < 1e-5


# ==================================================== runner

def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}\n          {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
