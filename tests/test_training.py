"""
Tests for the training-and-scale chapters (Ch 10, 11, 12, 19, 21).
Every number those chapters quote from tfs/memory.py, tfs/data.py,
tfs/trainkit.py, tfs/fp8.py, tfs/parallel.py and tfs/scaling.py is asserted here.

    python tests/test_training.py      # no pytest needed
"""

import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfs.model import Config, Model, Block, build_rope_cache, apply_rope
from tfs import memory as M, data as D, trainkit as K, fp8 as F8, parallel as P, scaling as S

SMALL = Config(vocab_size=512, d_model=96, n_layer=2, n_head=4, n_kv_head=2,
               d_head=24, d_ff=256, max_T=64)
GB = 1e9

# ===================================================== Ch 10 — memory of backward

def test_saved_tensor_count_matches_autograd():
    """10.7: the analytic count of what a Block saves equals what autograd saves."""
    torch.manual_seed(0)
    cfg = Config(vocab_size=512, d_model=96, n_layer=1, n_head=4, n_kv_head=2,
                 d_head=24, d_ff=256, max_T=64)
    blk = Block(cfg)
    B, T = 2, 64
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    cos, sin = build_rope_cache(T, cfg.d_head)
    n, _ = M.measure_saved(lambda x: blk(x, cos, sin)[0], x,
                           exclude=list(blk.parameters()) + [cos, sin])
    per_tok = M.saved_elements_per_token(96, 4, 24, 256)
    assert per_tok == 1990
    assert n == per_tok * B * T * 4 == 1_018_880


def test_llama3_8b_activation_memory():
    """10.7 worked example: Llama-3-8B shape, B=1, T=8192, bf16."""
    e = M.saved_elements_per_token(4096, 32, 128, 14336)
    assert e == 98_338 and round(e / 4096, 1) == 24.0
    per_layer = e * 8192 * 2
    assert round(per_layer / GB, 2) == 1.61
    assert round(32 * per_layer / GB, 1) == 51.6
    e2 = M.saved_elements_per_token(4096, 32, 128, 14336, expanded_kv=False, n_kv_head=8)
    assert e2 == 92_194 and round(32 * e2 * 8192 * 2 / GB, 1) == 48.3
    # checkpointing: one boundary per layer is 67 MB
    assert round(4096 * 8192 * 2 / 1e6) == 67
    assert round(M.best_checkpoint_interval(32, e, 4096), 2) == 1.13
    mem = {k: M.checkpoint_memory(32, e, 4096, k) * 8192 * 2 / GB for k in (1, 2, 4, 8)}
    assert round(mem[1], 2) == 3.83 and round(mem[2], 2) == 4.43
    assert round(mem[4], 2) == 7.25 and round(mem[8], 2) == 13.69


def test_llama3_70b_exercise():
    """Exercise 10.5."""
    e = M.saved_elements_per_token(8192, 64, 128, 28672)
    assert e == 196_674
    assert round(e * 8192 * 2 * 80 / GB, 1) == 257.8
    assert round(80 * 8192 * 8192 * 2 / GB, 1) == 10.7
    assert round((e + 8192) * 8192 * 2 / GB, 1) == 3.4
    assert round(M.best_checkpoint_interval(80, e, 8192), 2) == 1.79


def test_logits_dominate_after_checkpointing():
    """10.8: the (B*T, V) tensors at the end, Llama-3 vocabulary."""
    V = 128_256
    assert round(8192 * V * 2 / GB, 2) == 2.10          # bf16 logits
    assert round(8192 * V * 4 / GB, 2) == 4.20          # one fp32 copy
    assert round(M.logits_bytes(1, 8192, V) / GB, 1) == 10.5
    assert round(1024 * V * 4 / GB, 3) == 0.525         # one 1024-token chunk


def test_cross_entropy_saves_a_vocab_sized_tensor():
    """10.8: F.cross_entropy saves exactly one fp32 (N, V); the chunked loss saves none."""
    torch.manual_seed(0)
    h = torch.randn(4096, 64, requires_grad=True)
    W = torch.randn(8000, 64, requires_grad=True)
    t = torch.randint(0, 8000, (4096,))
    full, _ = M.measure_saved(lambda h: F.cross_entropy(h @ W.T, t), h, exclude=[W, h, t])
    chunk, _ = M.measure_saved(lambda h: M.chunked_cross_entropy(h, W, t, 512), h, exclude=[W, h, t])
    assert full >= 4096 * 8000 * 4
    assert chunk == (4096 + 8000) * 64 * 4              # just dL/dh and dL/dW


def test_chunked_cross_entropy_is_exact():
    torch.manual_seed(0)
    h = torch.randn(300, 32, dtype=torch.float64, requires_grad=True)
    W = torch.randn(50, 32, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, 50, (300,))
    l1 = F.cross_entropy(h @ W.T, t)
    l1.backward()
    gh, gW = h.grad.clone(), W.grad.clone()
    h.grad = W.grad = None
    l2 = M.chunked_cross_entropy(h, W, t, chunk=64)
    l2.backward()
    assert abs(l1.item() - l2.item()) < 1e-12
    assert torch.allclose(h.grad, gh, atol=1e-12) and torch.allclose(W.grad, gW, atol=1e-12)


def test_embedding_backward_is_scatter_add():
    torch.manual_seed(0)
    E = torch.nn.Embedding(10, 4)
    ids = torch.tensor([3, 7, 3, 3, 1])
    g = torch.randn(5, 4)
    E(ids).backward(g)
    mine = M.embedding_backward(ids, g, 10)
    assert torch.allclose(E.weight.grad, mine)
    assert torch.allclose(mine[3], g[0] + g[2] + g[3])     # three lookups, summed
    assert mine[0].abs().sum() == 0                        # unseen token: zero


def test_swiglu_backward():
    torch.manual_seed(0)
    a = torch.randn(6, 9, dtype=torch.float64, requires_grad=True)
    b = torch.randn(6, 9, dtype=torch.float64, requires_grad=True)
    g = torch.randn(6, 9, dtype=torch.float64)
    (F.silu(a) * b).backward(g)
    da, db = M.swiglu_backward(a.detach(), b.detach(), g)
    assert torch.allclose(a.grad, da) and torch.allclose(b.grad, db)


def test_rope_backward_is_inverse_rotation():
    torch.manual_seed(0)
    cos, sin = build_rope_cache(16, 8)
    q = torch.randn(1, 2, 16, 8, dtype=torch.float64, requires_grad=True)
    g = torch.randn(1, 2, 16, 8, dtype=torch.float64)
    qr, _ = apply_rope(q, q.detach(), cos.double(), sin.double())
    qr.backward(g)
    assert torch.allclose(q.grad, M.rope_backward(g, cos.double(), sin.double()))
    # and the norm is preserved: rotation is orthogonal
    assert torch.allclose(q.grad.norm(), g.norm())


def test_gradient_accumulation_dtype():
    """10.9: summing 64 micro-batch gradients with the running total in each dtype."""
    torch.manual_seed(0)
    gs = [torch.randn(100_000) * 1e-3 + 2e-3 for _ in range(64)]
    ref = torch.stack([x.double() for x in gs]).sum(0)
    err = {dt: ((M.accumulate(gs, dt).double() - ref).norm() / ref.norm()).item()
           for dt in (torch.bfloat16, torch.float16, torch.float32)}
    assert 6e-3 < err[torch.bfloat16] < 9e-3
    assert 7e-4 < err[torch.float16] < 1.2e-3
    assert err[torch.float32] < 2e-7


def test_korthikanti_formula_gpt3():
    """19.6: GPT-3 175B, s=2048, b=1, h=12288, a=96, 96 layers."""
    s, b, h, a = 2048, 1, 12288, 96
    f = lambda **kw: M.korthikanti_bytes_per_layer(s, b, h, a, **kw)
    assert round(f() / GB, 2) == 2.87 and round(96 * f() / GB) == 275
    assert round(96 * f(t=8) / GB, 1) == 55.6
    assert round(96 * f(t=8, sp=True) / GB, 1) == 34.4
    assert round(96 * f(t=8, sp=True, recompute="selective") / GB, 1) == 10.3
    assert round(96 * f(recompute="full") / GB, 2) == 4.83


# ===================================================== Ch 11 — data

DOC_A = ("The city council voted on Tuesday to approve a new budget for public libraries, "
         "adding funds for longer opening hours, more staff and a mobile library that will "
         "visit rural schools twice a month. The mayor said the plan would cost about four "
         "million dollars over three years and called it an investment in the next generation "
         "of readers.")
DOC_B = DOC_A + " Subscribe to our newsletter for more local news."
DOC_C = DOC_A.replace("Tuesday", "Wednesday") + " Subscribe to our newsletter for more local news."
DOC_D = ("Gradient descent updates every parameter in the direction that most reduces the loss, "
         "scaled by a learning rate. With a learning rate that is too large the loss diverges; "
         "too small and training crawls. Adaptive optimizers such as Adam rescale each "
         "coordinate by a running estimate of its gradient magnitude.")


def test_minhash_worked_example():
    sa, sb, sc, sd = (D.shingles(x) for x in (DOC_A, DOC_B, DOC_C, DOC_D))
    assert (len(DOC_A.split()), len(sa), len(sb)) == (59, 55, 63)
    assert len(sa & sb) == 55 and round(D.jaccard(sa, sb), 3) == 0.873
    assert len(sa & sc) == 50 and len(sa | sc) == 68 and round(D.jaccard(sa, sc), 3) == 0.735
    assert D.jaccard(sa, sd) == 0.0
    mh = D.MinHasher(112, 0)
    ga, gb, gc, gd = (mh.signature(x) for x in (sa, sb, sc, sd))
    assert sum(x == y for x, y in zip(ga, gb)) == 101
    assert sum(x == y for x, y in zip(ga, gc)) == 85
    assert D.signature_agreement(ga, gd) == 0.0
    bands = lambda g: sum(ga[i * 8:i * 8 + 8] == g[i * 8:i * 8 + 8] for i in range(14))
    assert (bands(gb), bands(gc)) == (4, 0)       # B caught 4 times over; C missed
    # the estimator is unbiased: average over 200 independent hashers
    ests = [D.signature_agreement(m.signature(sa), m.signature(sc))
            for m in (D.MinHasher(112, s) for s in range(200))]
    assert abs(statistics.mean(ests) - 0.735) < 0.01
    assert abs(statistics.pstdev(ests) - math.sqrt(0.735 * 0.265 / 112)) < 0.005


def test_lsh_s_curve():
    p = lambda s: round(D.lsh_catch_probability(s, 14, 8), 3)
    assert (p(0.5), p(0.6), p(0.7), p(0.75), p(0.8), p(0.9)) == (0.053, 0.211, 0.565, 0.772, 0.924, 1.0)
    assert round(D.lsh_catch_probability(0.873, 14, 8), 3) == 0.997
    assert round(D.lsh_catch_probability(0.735, 14, 8), 3) == 0.712
    assert round(D.lsh_threshold(14, 8), 3) == 0.719


def test_dedup_pipeline():
    docs = [DOC_A, DOC_B, DOC_D, DOC_A.upper()]
    assert D.exact_dedup(docs) == [0, 1, 2]       # upper-case copy is an exact dup
    assert D.dedup(docs) == [0, 2]                # B is a near dup of A


def test_gopher_rules():
    assert D.gopher_rules(DOC_A) == []
    assert D.gopher_rules("BUY NOW!!! " * 60) == ["fewer than 2 stop words"]
    assert "symbol-to-word ratio > 0.1" in D.gopher_rules("# # # tag " * 30)
    assert "word count outside 50..100,000" in D.gopher_rules("the cat and the dog")


def test_quality_classifier_learns_the_judge():
    good = ["the derivative of a function measures how fast its value changes as the input changes",
            "photosynthesis converts light energy into chemical energy stored in glucose",
            "a prime number has exactly two divisors one and itself",
            "the french revolution began in 1789 and ended the absolute monarchy"]
    bad = ["click here buy now best price free shipping limited offer",
           "hot deals cheap watches buy now click here",
           "sign up now free gift card winner click",
           "best casino bonus click here free spins now"]
    clf = D.QualityClassifier().fit(good + bad, [1] * 4 + [0] * 4)
    assert clf.score("the derivative measures how a value changes") > 0.5
    assert clf.score("click here for free spins now") < 0.5


def test_mixture_weights_and_epochs():
    sizes = [12000, 1500, 300, 150, 50]               # B tokens, illustrative
    w1 = D.mixture_weights(sizes, 1)
    assert [round(x, 4) for x in w1] == [0.8571, 0.1071, 0.0214, 0.0107, 0.0036]
    w2 = D.mixture_weights(sizes, 2)
    assert [round(x, 3) for x in w2] == [0.592, 0.209, 0.094, 0.066, 0.038]
    ep = D.epochs_per_source(w2, sizes, 15000)
    assert [round(e, 1) for e in ep] == [0.7, 2.1, 4.7, 6.6, 11.5]


def test_wsd_schedule():
    lr = lambda s: D.wsd_lr(s, 3e-4, 100, 1000)
    assert lr(99) == 3e-4 and lr(500) == 3e-4 and lr(899) == 3e-4
    assert abs(lr(950) - 1.5e-4) < 1e-12 and lr(999) < 1e-5


# ===================================================== Ch 11 — batch, muP, MFU, stability

def _fallback_data():
    from tfs.train import _fallback_corpus
    from tfs.tokenizer import CharTokenizer
    text = _fallback_corpus()
    tok = CharTokenizer(text)
    return torch.tensor(tok.encode(text), dtype=torch.long), tok.vocab_size


def test_noise_scale_grows_during_training():
    """11.7: B_simple is small at init and grows as the loss falls."""
    data, V = _fallback_data()
    cfg = Config(vocab_size=V, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
                 d_head=16, d_ff=128, max_T=64)
    torch.manual_seed(0)
    model = Model(cfg)
    b0, G0, S0 = K.grad_noise_scale(model, data, 32, 2, 32, draws=6, seed=0)
    assert G0 > 0 and S0 > 0
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        ix = torch.randint(len(data) - 33, (16,), generator=g)
        x = torch.stack([data[i:i + 32] for i in ix])
        _, loss, _ = model(x, targets=x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    b1, _, _ = K.grad_noise_scale(model, data, 32, 2, 32, draws=6, seed=1)
    assert b1 > 2 * b0, (b0, b1)


def test_steps_examples_tradeoff():
    s, e = K.steps_and_examples(1000, 1000)
    assert (s, e) == (2.0, 2.0)
    s, e = K.steps_and_examples(100, 1000)
    assert (round(s, 1), round(e, 1)) == (11.0, 1.1)


def test_mup_makes_update_size_width_independent():
    sp = [K.adam_output_change(w, 1e-3) for w in (256, 1024, 4096)]
    mu = [K.adam_output_change(w, 1e-3, mup=True) for w in (256, 1024, 4096)]
    assert [round(x, 3) for x in sp] == [0.025, 0.087, 0.332]
    assert [round(x, 3) for x in mu] == [0.025, 0.022, 0.021]
    assert sp[2] / sp[0] > 12 and mu[2] / mu[0] < 1.0


def test_flops_per_token():
    cfg = Config(vocab_size=65, d_model=192, n_layer=6, n_head=6, n_kv_head=2,
                 d_head=32, d_ff=512, max_T=256)
    assert sum(p.numel() for p in Model(cfg).parameters()) == 2_374_272
    assert K.flops_per_token(cfg, 128) == 16_000_128
    assert 6 * 2_374_272 == 14_245_632


def test_mfu_measurement_is_sane():
    cfg = Config(vocab_size=65, d_model=96, n_layer=2, n_head=4, n_kv_head=2,
                 d_head=24, d_ff=256, max_T=64)
    peak = K.measure_peak_matmul(n=512, reps=3)
    tps = K.measure_train_throughput(cfg, 8, 64, steps=3, warm=1)
    u = K.mfu(tps, K.flops_per_token(cfg, 64), peak)
    assert 0 < u < 1, u


def test_z_loss_and_spike_detector():
    z = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    assert abs(K.z_loss(z, coef=1.0).item() - ((math.log(2)) ** 2 + (10 + math.log(2)) ** 2) / 2) < 1e-4
    det = K.SpikeDetector(window=50, k=6)
    flags = [det.update(2.0 + 0.01 * math.sin(i)) for i in range(60)]
    assert not any(flags)
    assert det.update(2.5) is True and det.update(2.0) is False


def test_exact_resumption_is_bit_identical():
    """11.9: stop at step 20, save, rebuild from scratch, load, finish: same weights."""
    data, V = _fallback_data()
    cfg = Config(vocab_size=V, d_model=48, n_layer=2, n_head=4, n_kv_head=2,
                 d_head=12, d_ff=96, max_T=32)
    a = K.ResumableRun(cfg, data, 8, 32, total=40)
    la = a.train(40)
    b = K.ResumableRun(cfg, data, 8, 32, total=40)
    lb = b.train(20)
    ckpt = {k: v for k, v in b.state_dict().items()}
    import copy
    ckpt = copy.deepcopy(ckpt)
    c = K.ResumableRun(cfg, data, 8, 32, total=40, seed=123)   # different init on purpose
    c.load_state_dict(ckpt)
    lc = c.train(20)
    assert la == lb + lc
    for (n, p), (_, q) in zip(a.model.state_dict().items(), c.model.state_dict().items()):
        assert torch.equal(p, q), n
    # forgetting the loader position is NOT bit-identical
    d = K.ResumableRun(cfg, data, 8, 32, total=40)
    d.load_state_dict(copy.deepcopy(ckpt))
    d.loader.pos = 0
    d.train(20)
    assert not torch.equal(a.model.head.weight, d.model.head.weight)


# ===================================================== Ch 12 — fp8 and bf16 updates

def test_fp8_scaling_granularity():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 1024, generator=g)
    res = {}
    for r in (100, 10_000):
        xo = x.clone()
        xo[:, 7] *= r
        mask = torch.ones_like(xo, dtype=torch.bool)
        mask[:, 7] = False
        qt, qb = F8.quantize(xo, "e4m3"), F8.quantize(xo, "e4m3", 128)
        res[r] = (((xo - qt)[mask].norm() / xo[mask].norm()).item(),
                  ((xo - qb)[mask].norm() / xo[mask].norm()).item(),
                  (qt[mask] == 0).float().mean().item(), (qb[mask] == 0).float().mean().item())
    assert round(res[100][0], 4) == 0.0264 and round(res[100][1], 4) == 0.0258
    assert round(res[10_000][0], 4) == 0.0363 and round(res[10_000][1], 4) == 0.0259
    assert round(res[10_000][2], 3) == 0.046 and round(res[10_000][3], 4) == 0.0017
    assert round(F8.rel_error(x, F8.quantize(x, "e4m3")), 4) == 0.0265
    assert round(F8.rel_error(x, F8.quantize(x, "e5m2")), 4) == 0.0525


def test_fp8_cast_saturates_and_delayed_scale():
    assert torch.tensor([659.0, 449.0]).to(torch.float8_e4m3fn).float().tolist() == [448.0, 448.0]
    assert round(F8.delayed_scale([3.1, 2.9, 3.4]), 2) == 131.76
    assert 5.0 * F8.delayed_scale([3.1, 2.9, 3.4]) > 448


def test_bf16_weight_updates():
    assert F8.bf16_accumulate(1.0, 1e-3, 1000, "nearest") == 1.0
    assert F8.bf16_accumulate(1.0, 1e-3, 1000, "kahan") == 2.0
    m = statistics.mean(F8.bf16_accumulate(1.0, 1e-3, 1000, "stochastic", s) for s in range(20))
    assert abs(m - 2.0) < 0.03


# ===================================================== Ch 19 — parallelism arithmetic

def test_context_parallel_bytes():
    T, H, KV, dh = 131_072, 32, 8, 128
    ring = {cp: P.ring_attention_bytes(T, cp, KV, dh) / 1e6 for cp in (2, 4, 8, 16)}
    uly = {cp: P.ulysses_bytes(T, cp, H, KV, dh) / 1e6 for cp in (2, 4, 8, 16)}
    assert {k: round(v) for k, v in ring.items()} == {2: 268, 4: 403, 8: 470, 16: 503}
    assert {k: round(v) for k, v in uly.items()} == {2: 671, 4: 503, 8: 294, 16: 157}
    for cp in (2, 4, 8, 16):
        assert abs(uly[cp] / ring[cp] - (2 * H + 2 * KV) / (2 * KV * cp)) < 1e-9
    assert P.allgather_cp_bytes(T, 16, KV, dh) == P.ring_attention_bytes(T, 16, KV, dh)


def test_ring_overlap_condition():
    assert round(P.ring_overlap_min_block(989e12, 50e9, 32, 8, causal=False)) == 4945
    assert round(P.ring_overlap_min_block(989e12, 50e9, 32, 8)) == 9890
    assert round(P.ring_overlap_min_block(989e12, 450e9, 32, 8)) == 1099
    assert round(P.ring_overlap_min_block(989e12, 50e9, 32, 32, causal=False)) == 19780  # Liu et al.: c >= F/B


def test_zigzag_balances_causal_work():
    z = P.zigzag_chunks(4)
    assert z == [(0, 7), (1, 6), (2, 5), (3, 4)]
    assert [P.causal_work(c, 8) for c in z] == [9, 9, 9, 9]
    assert [P.causal_work((2 * i, 2 * i + 1), 8) for i in range(4)] == [3, 7, 11, 15]


def test_fsdp_vs_hsdp_traffic():
    pb = 7e9 * 2
    assert round(P.fsdp_traffic(pb, 64) / GB, 1) == 41.3
    intra, inter = P.hsdp_traffic(pb, 8, 8)
    assert round(intra / GB, 2) == 36.75 and round(inter / GB, 2) == 3.06
    assert round(2 * 63 / 64 * pb / GB, 1) == 27.6
    assert P.hsdp_bytes_per_param(8) == 2.0


def test_fault_tolerance_arithmetic():
    mtbf = 54 * 24 * 3600 / 419                     # Llama 3: 419 unexpected in 54 days
    assert round(mtbf / 3600, 2) == 3.09
    tau = P.young_daly_interval(60, mtbf)
    assert round(tau / 60, 1) == 19.3
    assert round(P.expected_waste(tau, 60, mtbf), 3) == 0.104
    assert round(P.expected_waste(P.young_daly_interval(30, mtbf), 30, mtbf), 3) == 0.073
    assert round(P.expected_waste(tau, 60, mtbf, 600), 3) == 0.158
    assert round(405e9 * 16 / 1e12, 2) == 6.48
    assert round(390 / 1979, 3) == 0.197 and round(390 / 989, 3) == 0.394


# ===================================================== Ch 21 — scaling

def test_fitted_optimum_vs_rule_of_20():
    C = 3.14e23
    N20, D20 = S.rule_of_20(C)
    assert round(N20 / 1e9, 1) == 51.2 and round(D20 / 1e9) == 1023
    N, Dt, a, b = S.fitted_optimum(C)
    assert round(a, 3) == 0.452 and round(b, 3) == 0.548
    assert round(N / 1e9, 1) == 24.5 and round(Dt / 1e12, 2) == 2.14 and round(Dt / N) == 87
    assert round(S.chinchilla_loss(N, Dt), 4) == 1.9543
    assert round(S.chinchilla_loss(N20, D20), 4) == 1.9610
    ratios = [round(S.fitted_optimum(c)[1] / S.fitted_optimum(c)[0]) for c in (1e21, 1e23, 1e25, 1e27)]
    assert ratios == [50, 78, 122, 191]
    Nb, Db, ab, _ = S.fitted_optimum(C, **S.BESIROGLU)
    assert round(ab, 3) == 0.513 and round(Db / Nb, 1) == 18.7


def test_data_constrained_scaling():
    v = {e: round(S.effective_data(1.0, e), 2) for e in (1, 2, 4, 8, 16, 40, 100)}
    assert v == {1: 1.0, 2: 1.97, 4: 3.73, 8: 6.62, 16: 10.58, 40: 15.17, 100: 16.36}
    assert round(1 + S.R_D_STAR, 2) == 16.39


def test_test_time_compute():
    assert [round(S.coverage(0.159, k), 3) for k in (1, 5, 10)] == [0.159, 0.579, 0.823]
    assert round(S.coverage(0.05, 10), 3) == 0.401
    assert abs(S.pass_at_k(250, 40, 1) - 0.16) < 1e-12
    assert round(S.pass_at_k(250, 40, 10), 3) == 0.831


def test_attention_share_and_long_context_recipes():
    sh = lambda N, L, d, T: round(S.attention_share(N, L, d, T), 3)
    assert (sh(405e9, 126, 16384, 8192), sh(405e9, 126, 16384, 131072)) == (0.077, 0.572)
    assert (sh(8e9, 32, 4096, 8192), sh(8e9, 32, 4096, 131072)) == (0.212, 0.811)
    # DeepSeek-V3's two extension phases see the same number of tokens
    assert 1000 * 1920 * 32768 == 1000 * 480 * 131072 == 62_914_560_000
    # Llama 3 405B Table 4: TP x CP x PP x DP
    assert 8 * 1 * 16 * 64 == 8192 and 8 * 16 * 16 * 8 == 16384 and 8 * 1 * 16 * 128 == 16384


def test_exercise_and_prose_arithmetic():
    """Numbers quoted in the new exercises (10.x, 11.7, 12.6, 19.5, 19.7, 21.4-21.6)."""
    assert len(D.shingles(DOC_A) | D.shingles(DOC_D)) == 101
    # 19.6 / Ex 19.5: an 8B layer's activations at 128k, and per-rank memory with CP
    assert round(98_338 * 131_072 * 2 / GB, 1) == 25.8
    total = (32 * 131_072 * 4096 * 2 + 98_338 * 131_072 * 2 + 131_072 * 4096 * 2) / GB
    assert round(total) == 61 and round(total / 8, 1) == 7.7 and round(total / 16, 1) == 3.8
    # Ex 19.7
    mtbf = 54 * 24 * 3600 / 419 * 4
    w120 = P.expected_waste(P.young_daly_interval(120, mtbf), 120, mtbf, 600)
    w5 = P.expected_waste(P.young_daly_interval(5, mtbf), 5, mtbf, 600)
    assert round(P.young_daly_interval(120, mtbf) / 60) == 54 and round(w120, 3) == 0.087
    assert round(P.young_daly_interval(5, mtbf) / 60) == 11 and round(w5, 3) == 0.028
    assert round((w120 - w5) * 4096 * 24, -2) == 5700
    # Ex 11.7: 70B, 15T tokens, 8192 H100 at 40% MFU
    per_tok = 6 * 70e9 + 12 * 80 * 8192 * 8192
    days = per_tok * 15e12 / (8192 * 989e12 * 0.40) / 86400
    assert round(days) == 26 and round(12 * 80 * 8192 * 8192 / per_tok, 2) == 0.13
    # Ex 12.6: where per-tensor e4m3 scaling pushes |x| ~ 1 below the smallest normal
    assert round(448 * 64 / 3) == 9557
    # Ex 21.4 / 21.5 / 21.6
    N, Dt = S.rule_of_20(1e24)
    assert round(N / 1e9, 1) == 91.3 and round(Dt / 1e12, 2) == 1.83
    assert round(1e24 / (6 * 12e12) / 1e9, 1) == 13.9 and round(S.effective_data(3, 4), 1) == 11.2
    assert round(S.coverage(0.3, 10), 3) == 0.972
    f = lambda T: 6 * 8e9 + 12 * 32 * 4096 * T
    assert round(f(131_072) / f(8192), 1) == 4.2
    assert round((62.9e9 * f(32_768) + 62.9e9 * f(131_072)) / f(8192) / 1e12, 2) == 0.37
    assert S.attention_share(8e9, 32, 4096, 30_000) < 0.5 < S.attention_share(8e9, 32, 4096, 32_768)
    assert S.attention_share(405e9, 126, 16384, 98_000) < 0.5 < S.attention_share(405e9, 126, 16384, 100_000)
    # 21.3: DeepSeek-V3 tokens per active parameter
    assert round(14.8e12 / 37e9) == 400


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
