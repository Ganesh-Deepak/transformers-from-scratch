"""
Verification suite for Part IV (inference and systems, Chapters 13-18) and the
Ch 20 router widget's serving curve. Every test corresponds to numbers the
chapters print.

    python tests/test_inference.py          # no pytest needed
    python -m pytest tests/test_inference.py -v
"""

import math
import os
import sys
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfs import decoding as D
from tfs import flash as FL
from tfs import gpu as G
from tfs import kvcache as KV
from tfs import mla as MLA
from tfs import moe as M
from tfs import serving as S
from tfs.model import Config, Model

CFG = Config(vocab_size=512, d_model=96, n_layer=3, n_head=4, n_kv_head=2,
             d_head=24, d_ff=256, max_T=128)
r = round


# ==================================================== Ch 13 — decoding

def test_best_of_n_and_pass_at_k():
    """13.6: one sample right 30% of the time; n independent tries."""
    assert [r(D.pass_at_n(0.3, n), 3) for n in (1, 5, 9, 21)] == [0.3, 0.832, 0.96, 0.999]
    # unbiased estimator: 20 samples, 3 correct
    assert r(D.pass_at_k(20, 3, 1), 3) == 0.15 and r(D.pass_at_k(20, 3, 5), 3) == 0.601
    assert D.pass_at_k(20, 18, 5) == 1.0


def test_self_consistency():
    """13.6: majority vote helps when the right answer is the mode, hurts when a wrong one is."""
    good = [r(D.majority_vote_accuracy([0.4, 0.3, 0.2, 0.1], 0, n), 3) for n in (1, 5, 9, 21, 41)]
    assert good == [0.4, 0.49, 0.556, 0.668, 0.763]
    bad = [r(D.majority_vote_accuracy([0.35, 0.45, 0.1, 0.1], 0, n), 3) for n in (1, 5, 9, 21, 41)]
    assert bad == [0.35, 0.363, 0.353, 0.302, 0.236]
    # sanity: probabilities over all answers sum to 1
    tot = sum(D.majority_vote_accuracy([0.4, 0.3, 0.2, 0.1], i, 7) for i in range(4))
    assert abs(tot - 1) < 1e-12


def test_stop_sequences_streaming():
    """13.7: a stop string split across three tokens is caught and never shown."""
    emitted, tail, stopped = D.stream(["Sure", ".\n", "\nUs", "er", ": hi"], ["\n\nUser:"])
    assert emitted == ["Sure", ".", "", "", ""] and stopped and tail == ""
    # a held-back newline is released as soon as it cannot be a stop any more
    emitted, tail, stopped = D.stream(["a", "\n", "b"], ["\n\nUser:"])
    assert emitted == ["a", "", "\nb"] and not stopped
    # held-back text at the end of a stream is flushed, not lost
    emitted, tail, stopped = D.stream(["x", "\n"], ["\n\nUser:"])
    assert "".join(emitted) + tail == "x\n"


def test_fp32_reduction_order():
    """13.8: same numbers, different order, different sum."""
    a, b = D.f32(1e8), D.f32(-1e8)
    assert D.sum_fp32([a, 1.0, b]) == 0.0 and D.sum_fp32([a, b, 1.0]) == 1.0
    x = D.random_fp32(4096, 1)
    sums = [D.sum_fp32(x, c) for c in (1, 2, 4, 8, 16, 32)]
    assert len(set(sums)) == 6                                    # every split count differs
    assert r(sums[0], 4) == 27.7162 and r(math.fsum(x), 4) == 27.7162
    assert max(sums) - min(sums) < 2e-4


def test_argmax_flip():
    """13.8: a near-tie whose winner depends on the split count."""
    f = D.find_argmax_flip()
    assert f is not None and f["seed"] == 1
    assert f["winner"] == {1: 1, 8: 0}
    e0, e1 = f["exact"]
    assert e0 > e1 and 0 < e0 - e1 < 2e-6                         # token 0 truly wins, by ~1e-6


def test_grammar_masks():
    """13.4: the toy grammar {"n": <digits>}."""
    P = D.Pattern(D.TOY_PATTERN)
    V = D.TOY_VOCAB
    assert D.token_mask(P, P.start(), V) == ["{", '{"']
    st = P.walk(P.start(), '{"n": 1')
    assert D.token_mask(P, st, V) == ["1", "2", "7", "12", "42", "7}", "}"]
    assert D.token_mask_whole_terminals(P, st, V) == ["1", "2", "7", "12", "42"]
    assert P.accepts('{"n": 17}') and not P.accepts('{"n": }') and not P.accepts('{"n": 17')
    assert D.greedy_tokenize('{"n": 17}', V) == ['{"', 'n"', ":", " ", "1", "7}"]
    assert D.tokenizations('{"n": 17}', V) == 24


# ==================================================== Ch 14 — the GPU

def test_ridge_points():
    assert [r(G.ridge(g)) for g in ("A100", "H100", "H200", "B200")] == [153, 295, 206, 281]
    assert r(G.ridge("H100", "fp8"), 1) == 590.7 and r(G.ridge("H200", "fp8"), 1) == 412.3
    assert r(G.ridge("B200", "fp8"), 1) == 562.5 and G.ridge("B200", "fp4") == 1125.0
    # decode batch that saturates: bf16 weights (2 B) vs fp8 weights (1 B) on fp8 tensor cores
    assert r(G.decode_batch_to_saturate("H100")) == 295
    assert r(G.decode_batch_to_saturate("H100", "fp8", 1)) == 295
    assert r(G.decode_batch_to_saturate("B200", "fp4", 0.5)) == 281


def test_mfu():
    assert r(G.mfu(8.03e9, 9000, 989.4e12), 3) == 0.438
    assert r(G.tokens_per_s_at_mfu(8.03e9, 0.5, 989.4e12, train=False)) == 30803


def test_launch_overhead():
    assert G.kernels_per_decode_step(32) == 355 and G.kernels_per_decode_step(12) == 135
    big = G.decode_step_floor(16.06e9, 3.35e12, 355, 4e-6)
    assert big["bound"] == "memory" and r(big["memory"] * 1e3, 2) == 4.79 and r(big["launch"] * 1e3, 2) == 1.42
    small = G.decode_step_floor(0.25e9, 3.35e12, 135, 4e-6)
    assert small["bound"] == "overhead" and r(small["memory"] * 1e6) == 75 and r(small["launch"] * 1e6) == 540


def test_interconnect():
    assert r(G.ring_allreduce_seconds(1e9, 8, 450e9) * 1e3, 2) == 3.89
    assert r(G.ring_allreduce_seconds(1e9, 8, 50e9) * 1e3, 1) == 35.0
    assert r(G.LINKS["NVLink 4 (H100), per GPU"] / G.LINKS["PCIe 5.0 x16"], 1) == 7.0


# ==================================================== Ch 15 — KV cache

def test_kv_units():
    """15.4 in binary units throughout."""
    assert KV.kv_bytes(32, 8, 128) == 131_072 == 128 * KV.KIB
    assert KV.kv_bytes(32, 8, 128, 131072) / KV.GIB == 16.0
    assert r(8.03e9 * 2 / KV.GIB, 2) == 14.96                      # Llama-3-8B weights in GiB
    assert KV.kv_bytes(80, 8, 128) / KV.KIB == 320
    assert KV.kv_bytes(80, 8, 128, 8192) / KV.GIB == 2.5 and KV.kv_bytes(80, 8, 128, 131072) / KV.GIB == 40
    budget = 4 * 80e9 - 70.6e9 * 2
    assert r(budget / KV.GIB, 1) == 166.5
    assert KV.users_that_fit(budget, KV.kv_bytes(80, 8, 128, 131072)) == 4
    assert KV.users_that_fit(budget, KV.kv_bytes(80, 8, 128, 8192)) == 66
    assert KV.users_that_fit(40 * KV.GIB, KV.kv_bytes(32, 8, 128, 8192)) == 40


def test_chunked_prefill_exact():
    """15.6: chunked prefill must give the same logits and cache as one shot."""
    torch.manual_seed(0)
    m = Model(CFG).eval()
    ids = torch.randint(0, CFG.vocab_size, (2, 40))
    with torch.no_grad():
        full, _, full_cache = m(ids)
        for chunk in (1, 7, 16, 40):
            lc, c = KV.prefill_chunked(m, ids, chunk)
            assert torch.allclose(lc[:, -1], full[:, -1], atol=1e-5), chunk
            assert torch.allclose(c[-1][0], full_cache[-1][0], atol=1e-5)
        # and the trap: Model.forward with is_causal=True on a multi-token chunk
        _, _, cc = m(ids[:, :32])
        bad, _, _ = m(ids[:, 32:], caches=cc, pos_offset=32)
        assert r((bad[:, -1] - full[:, -1]).abs().max().item(), 2) == 0.34


def test_cache_timing_shape():
    """15.6: uncached cost per token grows with position; cached stays nearly flat."""
    torch.manual_seed(0)
    cfg = Config(vocab_size=512, d_model=256, n_layer=2, n_head=4, n_kv_head=2, d_head=64,
                 d_ff=688, max_T=256)
    m = Model(cfg)
    c = KV.time_generation(m, 16, 230, True, repeats=3)
    u = KV.time_generation(m, 16, 230, False, repeats=3)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    assert med(u[-20:]) > 1.5 * med(c[-20:])                       # loose: timings are noisy
    assert med(u[-20:]) > 1.3 * med(u[:20])


def test_eviction_policies():
    """15.7 on the synthetic sink + heavy-hitter + recency pattern."""
    acc, p = KV.synthetic_attention()
    T = 1024
    assert KV.streaming_keep(T, 4, 60)[:5] == [0, 1, 2, 3, 964]
    got = {}
    for B in (64, 128, 256):
        rec = list(range(T - B, T))
        st = KV.streaming_keep(T, 4, B - 4)
        h = KV.h2o_keep(acc, B, B // 2)
        assert len(st) == len(h) == B
        got[B] = (r(KV.mass_kept(p, rec), 2), r(KV.mass_kept(p, st), 2), r(KV.mass_kept(p, h), 2))
    assert got == {64: (0.14, 0.14, 0.44), 128: (0.26, 0.27, 0.54), 256: (0.46, 0.46, 0.69)}
    g = torch.Generator().manual_seed(1)
    k = torch.randn(1024, 64, generator=g)
    qw = torch.randn(32, 64, generator=g) + 0.5 * k[300]
    keep = KV.snapkv_keep(qw, k, 96)
    assert len(keep) == 96 + 32 and 300 in keep


# ==================================================== Ch 16 — cross-layer sharing, MLA

def test_cross_layer_kv():
    assert KV.cla_kv_bytes(32, 8, 128, 2) == 65_536                # CLA2 halves Llama-3-8B's 128 KiB
    assert KV.cla_kv_bytes(32, 8, 128, 3) == 45_056                # ceil(32/3) = 11 layers
    assert KV.yoco_kv_bytes(32, 8, 128, 131072, 4096) / KV.GIB == 0.75
    assert r(KV.kv_bytes(32, 8, 128, 131072) / KV.yoco_kv_bytes(32, 8, 128, 131072, 4096), 1) == 21.3


def test_mla_absorbed_equals_naive():
    torch.manual_seed(0)
    H, dn, dr, rank, dv, T = 8, 32, 16, 64, 32, 20
    args = [torch.randn(H, dn), torch.randn(H, dr), torch.randn(T, rank), torch.randn(T, dr),
            torch.randn(H, dn, rank) / 8, torch.randn(H, dv, rank) / 8]
    assert torch.allclose(MLA.mla_decode_naive(*args), MLA.mla_decode_absorbed(*args), atol=1e-5)


def test_decode_attention_intensity():
    """16.4: absorbed MLA is MQA over a 576-wide latent; ~30x the FLOPs per cached byte of GQA."""
    mla = MLA.decode_attention_intensity(128, 576, 512, 576)
    assert r(mla, 1) == 241.8
    assert MLA.decode_attention_intensity(64, 128, 128, 2 * 8 * 128) == 8.0     # Llama-3-70B GQA
    assert MLA.decode_attention_intensity(128, 128, 128, 2 * 128 * 128) == 1.0  # MHA
    assert r(mla / 8.0, 1) == 30.2


# ==================================================== Ch 17 — FlashAttention, decode

def test_flash_attention_in_tile_mask():
    torch.manual_seed(0)
    Q, K, V = torch.randn(3, 50, 32).unbind(0)
    for br, bc in ((16, 16), (8, 32), (32, 8), (50, 50)):
        assert torch.allclose(FL.flash_attention(Q, K, V, br, bc), FL.naive_attention(Q, K, V), atol=1e-5)
    assert torch.allclose(FL.flash_attention(Q, K, V, causal=False), FL.naive_attention(Q, K, V, False), atol=1e-5)


def test_split_kv_decode():
    torch.manual_seed(0)
    q, K, V = torch.randn(64), torch.randn(1000, 64), torch.randn(1000, 64)
    ref = (K @ q / 8).softmax(0) @ V
    for n in (1, 2, 7, 16, 1000):
        assert torch.allclose(FL.split_kv_decode(q, K, V, n), ref, atol=1e-5)
    assert r(FL.decode_occupancy(1, 32, 132), 3) == 0.242
    assert r(FL.decode_occupancy(1, 32, 132, 8), 2) == 1.94
    assert r(FL.decode_occupancy(16, 32, 132), 2) == 3.88


def test_split_kv_worked_example():
    """17.7: scores [1, 3, 2, 5], values [10, 20, 30, 40], two splits."""
    sc, v = torch.tensor([1., 3., 2., 5.]), torch.tensor([[10.], [20.], [30.], [40.]])
    oA, lA = FL.scores_partial(sc[:2], v[:2])
    oB, lB = FL.scores_partial(sc[2:], v[2:])
    assert (r(lA.item(), 4), r(oA.item(), 4), r(lB.item(), 4), r(oB.item(), 4)) == (3.1269, 18.808, 5.0486, 39.5257)
    tot = torch.logsumexp(torch.stack([lA, lB]), 0).item()
    assert r(tot, 5) == 5.18518 and r(math.exp(lA.item() - tot), 4) == 0.1277
    merged = FL.merge_partials(torch.stack([oA, oB]), torch.stack([lA, lB])).item()
    direct = (sc.softmax(0) @ v).item()
    assert r(merged, 3) == r(direct, 3) == 36.881


def test_mla_absorption_flops():
    """16.4: absorbed decode does 3.4x the attention FLOPs of the decompressed form, on 57x fewer bytes."""
    absorbed, decompressed = 128 * (2 * 576 + 2 * 512), 128 * (2 * 192 + 2 * 128)
    assert (absorbed, decompressed) == (278_528, 81_920) and r(absorbed / decompressed, 1) == 3.4
    assert r(2 * 128 * 128 / 576, 1) == 56.9


def test_chapter_exercise_numbers():
    """Exercise answers in Ch 14, 15 and 16."""
    assert r(G.mfu(70.6e9, 1.6e6 / 1024, 989.4e12), 3) == 0.669                    # 14.5
    assert r(12 * 80 * 8192 * 8192 / (6 * 70.6e9), 2) == 0.15
    cache = 20 * KV.kv_bytes(80, 8, 128, 131072) / 1e9                             # 15.2
    assert r(cache, 1) == 859.0 and math.ceil((cache + 141.2) / 80) == 13
    assert r(cache / 2 + 141.2, 1) == 570.7 and math.ceil((cache / 2 + 141.2) / 80) == 8
    assert KV.users_that_fit(50 * KV.GIB, KV.kv_bytes(32, 8, 128, 4096)) == 100    # 15.6
    assert KV.users_that_fit(50 * KV.GIB, KV.kv_bytes(32, 8, 128, 131072)) == 3
    mha = KV.kv_bytes(32, 32, 128, 32768) / KV.GIB                                 # 16.5
    assert (mha, mha / 4, mha / 8, mha / 16) == (16.0, 4.0, 2.0, 1.0)
    assert 8 * (2 * 80 + 2 * 64) == 2304 and 8 * (2 * 48 + 2 * 32) == 1280         # 16.4


def test_flex_style_masks():
    m = FL.dense_mask(FL.sliding_window(256), 4096)
    assert m.sum().item() == 1_015_936
    assert r(FL.block_sparsity(m, 128), 3) == 0.909
    assert r(FL.block_sparsity(FL.dense_mask(FL.causal, 4096), 128), 3) == 0.484
    doc = torch.tensor([0] * 1000 + [1] * 1500 + [2] * 1596)
    assert r(FL.block_sparsity(FL.dense_mask(FL.document(doc), 4096), 128), 3) == 0.789
    pl = FL.dense_mask(FL.prefix_lm(4), 8)
    assert pl[0, 3] and not pl[4, 5] and pl[5, 4]


def test_flex_attention_matches_sdpa():
    """The same predicate as a FlexAttention block mask gives SDPA's answer."""
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except ImportError:
        return
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 1, 2, 256, 32).unbind(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = create_block_mask(FL.sliding_window(64), None, None, 256, 256, device="cpu")
        o = flex_attention(q, k, v, block_mask=bm)
    o2 = F.scaled_dot_product_attention(q, k, v, attn_mask=FL.dense_mask(FL.sliding_window(64), 256))
    assert torch.allclose(o, o2, atol=1e-5)


# ==================================================== Ch 18 — serving economics

def test_decode_profile_and_cost():
    want = {1: (4.87, 205, 3.385), 8: (5.44, 1472, 0.472), 32: (7.36, 4349, 0.160),
            64: (9.92, 6450, 0.108), 128: (15.05, 8505, 0.082), 256: (25.31, 10116, 0.069)}
    for b, (tpot, tps, cost) in want.items():
        p = S.decode_profile(b, 2048)
        assert p["bound"] == "memory"
        assert (r(p["tpot"] * 1e3, 2), r(p["tok_per_s"]), r(S.cost_per_mtok(2.5, p["tok_per_s"]), 3)) == (tpot, tps, cost), b
    # at batch 256 the cache, not the weights, is most of the traffic
    assert 256 * 2048 * S.kv_bytes_per_token(32, 8, 128) > 4 * 8.03e9
    assert S.max_batch_for_tpot(0.030, 2048) == 314
    assert S.max_batch_for_memory(80e9, 2048, 4e9) == 223
    assert r(S.cost_per_mtok(2.5, 6450, 0.6), 3) == 0.179


def test_prefill_cost():
    tps = G.tokens_per_s_at_mfu(8.03e9, 0.5, 989.4e12, train=False)
    assert r(S.cost_per_mtok(2.5, tps), 4) == 0.0225
    assert r(S.cost_per_mtok(2.5, 6450) / S.cost_per_mtok(2.5, tps), 1) == 4.8


def test_chunked_prefill_trade():
    want = {256: (32, 9.92, 317.5), 512: (16, 9.92, 158.8), 1024: (8, 17.66, 141.3),
            2048: (4, 34.28, 137.1), 8192: (1, 134.01, 134.0)}
    for c, (steps, tpot, ttft) in want.items():
        t = S.chunked_prefill_trade(c, 8192, 64, 2048)
        assert (t["steps"], r(t["tpot"] * 1e3, 2), r(t["ttft"] * 1e3, 1)) == (steps, tpot, ttft), c
    assert S.balanced_chunk(64, 2048) == 547


def test_slo_metrics():
    reqs = [dict(ttft=t, tpot=p) for t, p in
            [(0.12, 0.020), (0.30, 0.025), (0.18, 0.045), (0.90, 0.030), (0.15, 0.022),
             (0.22, 0.028), (0.25, 0.031), (0.11, 0.019), (2.40, 0.024), (0.19, 0.026)]]
    assert S.percentile([x["ttft"] for x in reqs], 50) == 0.19
    assert S.percentile([x["ttft"] for x in reqs], 90) == 0.90
    assert S.percentile([x["ttft"] for x in reqs], 99) == 2.40
    assert S.slo_attainment(reqs, 0.5, 0.030) == 0.6
    assert S.goodput(reqs, 0.5, 0.030, 2.0) == 3.0           # 5 req/s throughput, 3 good


def test_moe_expert_parallel():
    n, B = S.batch_for_compute_bound_experts(256, 8)
    assert r(n) == 295 and r(B) == 9451
    assert S.tokens_per_expert(4096, 8, 256) == 128
    assert r(M.expected_distinct_experts(256, 8, 64) / 256, 2) == 0.87
    per_expert = 3 * 7168 * 2048
    assert per_expert == 44_040_192 and r(per_expert * 58 / 1e9, 2) == 2.55


def test_multi_lora():
    p = S.lora_params(S.llama_linear_shapes(), 16) * 32
    assert p == 41_943_040
    assert r(p * 2 / 1e6, 1) == 83.9 and r(p / 8.03e9 * 100, 2) == 0.52
    assert r(S.lora_batch_bytes(p * 2, 64) / 1e9, 2) == 5.37
    assert r(S.lora_batch_bytes(p * 2, 64) / (8.03e9 * 2), 2) == 0.33


def test_kv_offload_break_even():
    pcie = S.reload_vs_recompute(8192, 64e9)
    ssd = S.reload_vs_recompute(8192, 7e9)
    assert pcie["kv_bytes"] == KV.GIB
    assert r(pcie["reload"] * 1e3, 1) == 16.8 and r(ssd["reload"] * 1e3, 1) == 153.4
    assert r(pcie["recompute"] * 1e3, 1) == 301.5 and r(pcie["flops"] / 1e14, 2) == 1.49
    assert r(pcie["recompute"] / pcie["reload"]) == 18 and r(ssd["recompute"] / ssd["reload"], 1) == 2.0


def test_weight_formats():
    assert [S.bits_per_weight(f) for f in ("bf16", "mxfp4", "nvfp4")] == [16.0, 4.25, 4.5]
    assert r(S.bits_per_weight("fp8"), 3) == 8.002
    gb = {f: r(S.weight_bytes(70.6e9, f) / 1e9, 1) for f in ("bf16", "fp8", "mxfp4", "nvfp4")}
    assert gb == {"bf16": 141.2, "fp8": 70.6, "mxfp4": 37.5, "nvfp4": 39.7}
    h100 = {f: r(3.35e12 / S.weight_bytes(70.6e9, f)) for f in gb}
    assert h100 == {"bf16": 24, "fp8": 47, "mxfp4": 89, "nvfp4": 84}


# ==================================================== Ch 20 — router widget's curve

def test_widget_serving_curve():
    """The MoE widget plots E(1-(1-k/E)^b); its labelled points must match Ch 20 §20.7."""
    pts = {b: r(100 * M.expected_distinct_experts(256, 8, b) / 256) for b in (1, 8, 32, 64, 256)}
    assert pts == {1: 3, 8: 22, 32: 64, 64: 87, 256: 100}
    pts16 = {b: r(100 * M.expected_distinct_experts(16, 2, b) / 16) for b in (1, 4, 8, 16, 32)}
    assert pts16 == {1: 12, 4: 41, 8: 66, 16: 88, 32: 99}


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
