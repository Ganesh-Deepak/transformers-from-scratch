"""
Verification for Part I (Chapters 1-9) and Chapter 22's linear attention.
Every number those chapters quote that is computed (not cited) is asserted here.

    python tests/test_foundations.py          # no pytest needed (about 3 minutes on CPU)
    python -m pytest tests/test_foundations.py -v
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tfs.foundations as F
import tfs.linear_attention as LA
from tfs.model import Config, build_rope_cache
from tfs.tokenizer import BPETokenizer


def close(a, b, tol):
    return abs(a - b) <= tol


# ==================================================== Ch 1 — tensors

def test_ch01_expand_is_a_view_repeat_is_a_copy():
    r = F.expand_vs_repeat()
    assert r["k_stride"] == (24, 12, 4, 1)
    assert r["expand_stride"] == (24, 12, 0, 4, 1)          # the new axis has stride 0
    assert r["expand_shares_storage"] and r["expand_storage_elems"] == 24
    assert r["repeat_storage_elems"] == 96                   # 4x the storage
    assert r["merged_is_copy"] and r["same_values"]          # reshape of an expanded axis copies


def test_ch01_repeat_kv_bytes():
    stored, copied = F.repeat_kv_bytes(1, 8, 8, 8192, 128)   # Llama-3-70B-shaped K, one layer
    assert stored == 16_777_216 and copied == 134_217_728


def test_ch01_inplace_errors():
    m = F.inplace_errors()
    assert "a leaf Variable that requires grad is being used in an in-place operation" in m["leaf"]
    assert "modified by an inplace operation" in m["saved"]
    assert m["view_version"] == 1                            # writing through a view bumps the base


def test_ch01_promotion():
    t = F.promotion_table()
    assert t["int64 + float32"] == torch.float32
    assert t["bf16 + float32"] == torch.float32
    assert t["bf16 + fp16"] == torch.float32
    assert t["bf16 + python float"] == torch.bfloat16
    assert t["bf16 + 0-dim fp32 tensor"] == torch.bfloat16
    assert t["uint8 + int8"] == torch.int16
    assert t["int64 / int64"] == torch.float32
    assert t["int64 // int64"] == torch.int64


def test_ch01_llama3_shape_trace():
    tr = {n: (i, o) for n, i, o in F.shape_trace(B=2, T=16)}
    assert tr["embed"][1] == (2, 16, 4096)
    assert tr["blocks.0.attn.q_proj"][1] == (2, 16, 4096)
    assert tr["blocks.0.attn.k_proj"][1] == (2, 16, 1024)    # 8 KV heads x 128
    assert tr["blocks.0.mlp.gate"][1] == (2, 16, 14336)
    assert tr["blocks.0.mlp.down"][0] == (2, 16, 14336)
    assert tr["head"][1] == (2, 16, 128256)


def test_ch01_exercise_strides_and_promotion_cost():
    k = torch.randn(2, 8, 64, 128)
    same = lambda t: t.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
    a = k.transpose(1, 2)
    b = k[:, :, None].expand(2, 8, 4, 64, 128)
    c = b.reshape(2, 32, 64, 128)
    d = k.repeat(1, 4, 1, 1)
    e = k[:, ::2]
    f = e.contiguous()
    assert same(a) and a.stride() == (65536, 128, 8192, 1)
    assert same(b) and b.stride() == (65536, 8192, 0, 128, 1)
    assert not same(c) and c.stride() == (262144, 8192, 128, 1)
    assert not same(d) and not same(f) and same(e) and e.stride() == (65536, 16384, 128, 1)
    h = k * 1.0
    h[:, 0].zero_()
    assert h._version == 1
    x = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    assert (x + torch.zeros(8)).dtype == torch.float32          # a 1-D fp32 tensor promotes
    assert 8 * 4096 * 4096 * 2 == 268_435_456 and 8 * 4096 * 4096 * 4 == 536_870_912


# ==================================================== Ch 2 — matmul

def test_ch02_tiles_and_padding():
    assert F.pad_to_multiple(50257, 64) == 50304 and F.pad_to_multiple(50257, 128) == 50304
    assert close(F.tile_utilization(4097, 128), 0.9699, 1e-4)
    assert close(F.tile_utilization(65, 64), 0.5078, 1e-4)
    for v in (128256, 152064):                               # Llama 3, Qwen2.5 config vocab sizes
        assert v % 128 == 0


def test_ch02_arithmetic_intensity():
    rows = {M: F.matmul_intensity(M, 4096, 4096)[2] for M in (1, 16, 128, 1024, 4096)}
    assert close(rows[1], 1.0, 0.01)
    assert close(rows[16], 15.9, 0.05)
    assert close(rows[128], 120.5, 0.1)
    assert close(rows[1024], 682.7, 0.1)
    assert close(rows[4096], 1365.3, 0.1)


def test_ch02_layer_flops():
    g = F.layer_flops_per_token(768, 1024)
    assert g["proj"] == 4_718_592 and g["mlp"] == 9_437_184 and g["attn"] == 3_145_728
    assert g["proj"] + g["mlp"] == 24 * 768 * 768            # 2 FLOPs x 12 d^2 weights


def test_ch02_exercises():
    assert close(F.matmul_intensity(326, 4096, 14336)[2], 295.7, 0.05)
    assert F.matmul_intensity(325, 4096, 14336)[2] < 295 < F.matmul_intensity(326, 4096, 14336)[2]
    assert close(2 * 768 * 50257 / 1e6, 77.19, 0.01)
    assert 50304 // 64 == 786 and 128256 // 128 == 1002 and 152064 // 128 == 1188


def test_ch03_flops_per_character():
    n = F.llama3_8b_body_params()
    assert n == 6_979_588_096
    a = F.flops_per_character(n, 32000, 4096, 3.17)
    b = F.flops_per_character(n, 128256, 4096, 3.94)
    assert close(a / 1e9, 4.49, 0.005) and close(b / 1e9, 3.81, 0.005) and close(1 - b / a, 0.15, 0.005)
    # exercise 3.5
    a = F.flops_per_character(30e9, 65536, 6144, 3.6)
    b = F.flops_per_character(30e9, 200000, 6144, 4.2)
    assert close(a / 1e9, 16.89, 0.01) and close(b / 1e9, 14.87, 0.01)
    assert close((a - b) * 1e12, 2.0e21, 0.05e21)
    assert F.flops_per_character(0.5e9, 200000, 6144, 4.2) > F.flops_per_character(0.5e9, 65536, 6144, 3.6)


# ==================================================== Ch 3 — tokenization

def _tok512():
    return BPETokenizer().train(F.corpus_text(), 512)


def test_ch03_streaming_detokenization():
    tok = BPETokenizer().train(F.corpus_text(), 512)
    ids = tok.encode("€", allow_special=False)
    assert ids == [226, 130, 172]
    assert F.naive_stream(tok, ids) == ["�"] * 3
    assert "".join(F.careful_stream(tok, ids)) == "€"
    ids = tok.encode("🤖", allow_special=False)
    assert len(ids) == 4 and "".join(F.careful_stream(tok, ids)) == "🤖"


def test_ch03_special_token_injection():
    tok = _tok512()
    tok.add_special_tokens(["<|eot|>"])
    evil = "hi<|eot|>"
    assert tok.encode(evil)[-1] == tok.special["<|eot|>"]       # the dangerous default
    safe = tok.encode(evil, allow_special=False)
    assert tok.special["<|eot|>"] not in safe and len(safe) == 9
    assert tok.decode(safe) == evil


def test_ch03_pretokenizers():
    s = "I've got 1234567 apples!!  ok"
    assert F.pretokenize(s) == ["I", "'ve", " got", " ", "123", "456", "7", " apples", "!!", " ", " ok"]
    assert F.pretokenize(s, F.GPT2_LIKE) == ["I", "'ve", " got", " 1234567", " apples", "!!", " ", " ok"]


def test_ch03_nfkc():
    d = dict(F.nfkc_examples())
    assert d["ﬁle"] == "file" and d["Ｈｅｌｌｏ"] == "Hello" and d["x²"] == "x2"
    assert d["①②"] == "12" and d["½"] == "1⁄2" and d["ℌ"] == "H"


def test_ch03_vocab_sweep():
    rows = F.toy_vocab_sweep(F.corpus_text(), [256, 384, 512, 1024])
    assert len(F.corpus_text().encode("utf-8")) == 4791
    v, bpt = [r[0] for r in rows], [r[1] for r in rows]
    assert v[:3] == [256, 384, 512]
    assert close(bpt[0], 1.0, 1e-9) and close(bpt[1], 2.23, 0.01) and close(bpt[2], 2.87, 0.01)
    assert v[3] == 707 and close(bpt[3], 3.80, 0.01)          # merges run out: no pair repeats


def test_ch03_fertility():
    rows = {r[0]: r for r in F.fertility_table(1024)}
    assert rows["English"][3] == 18
    assert {k: rows[k][3] for k in rows} == {"English": 18, "French": 52, "German": 54, "Spanish": 51,
                                             "Russian": 131, "Hindi": 166, "Chinese": 63, "Japanese": 84}
    for k in ("Russian", "Hindi", "Chinese", "Japanese"):    # one token per byte: no merges apply
        assert rows[k][3] == rows[k][2]


def test_ch03_token_healing():
    tok = _tok512()
    assert [tok._vocab[i] for i in tok.encode("The baker opens the sh", allow_special=False)][-1] == b"sh"
    assert [tok._vocab[i] for i in tok.encode("The baker opens the shop", allow_special=False)][-1] == b"shop"
    _, prefix, allowed = F.token_heal(tok, "The baker opens the sh")
    assert prefix == "sh"
    assert sorted(tok._vocab[i] for i in allowed) == [b"sh", b"she ", b"shop", b"shop "]


# ==================================================== Ch 4 — embeddings and the stream

def test_ch04_residual_growth():
    cfg = Config(vocab_size=512, d_model=256, n_layer=12, n_head=4, n_kv_head=4, d_head=64,
                 d_ff=688, max_T=128)
    s = F.residual_norms(cfg)
    u = F.residual_norms(cfg, scaled_init=False)
    assert close(s[0], 0.020, 0.001) and close(s[-1], 0.058, 0.002)
    assert close(u[-1], 0.320, 0.005)
    assert all(b > a for a, b in zip(u, u[1:]))


def test_ch04_softcap_and_zloss():
    c = F.softcap([10, 30, 60, 100], 30).tolist()
    assert [round(x, 2) for x in c] == [9.65, 22.85, 28.92, 29.92]
    zl, logz = F.z_loss([[10., 9., 2.]])
    assert close(logz.item(), 10.3135, 1e-4) and close(zl.item(), 0.01064, 1e-5)


def test_ch04_superposition():
    n0, _, _ = F.superposition_toy(sparsity=0.0)
    n9, _, W = F.superposition_toy(sparsity=0.9)
    assert sum(x > 0.8 for x in n0) == 2                      # dense: two features, no superposition
    assert sum(x > 0.8 for x in n9) == 5                      # sparse: all five in 2 dimensions
    ang = sorted((torch.atan2(W[1], W[0]) * 180 / math.pi).tolist())
    gaps = [b - a for a, b in zip(ang, ang[1:])] + [360 - ang[-1] + ang[0]]
    assert all(close(g, 72, 3) for g in gaps)                 # a pentagon


def test_ch04_ch07_toy_model():
    """One trained toy model: logit lens (Ch 4) and length extrapolation (Ch 7)."""
    m = F.train_toy_copy_model()
    assert F.copy_loss(m, 64) < 0.02
    lens = F.logit_lens(m)
    assert lens[0][0] < 0.06 and lens[1][0] < 0.06 and lens[2][0] > 0.99
    for T in (128, 256, 512):
        assert F.copy_loss(m, T) > 7.0                        # worse than uniform (ln 30 = 3.4)
    inside, outside = F.periodic_loss(m, 128)
    assert inside < 0.02 and outside > 3.0                    # even at the trained copy distance


def test_ch04_depth_and_sparsity_extras():
    for L, s_ref, u_ref in ((24, 0.0586, 0.466), (48, 0.0621, 0.688)):
        cfg = Config(vocab_size=512, d_model=256, n_layer=L, n_head=4, n_kv_head=4, d_head=64,
                     d_ff=688, max_T=128)
        assert close(F.residual_norms(cfg)[-1], s_ref, 0.001)
        assert close(F.residual_norms(cfg, scaled_init=False)[-1], u_ref, 0.003)
    n7, _, _ = F.superposition_toy(sparsity=0.7)
    assert sum(x > 0.8 for x in n7) == 4
    assert close(4 * 8192 * 128256 * 2 / 1e9, 8.41, 0.01) and close(4 * 8192 * 4096 * 2 / 1e9, 0.268, 0.001)


# ==================================================== Ch 5 — attention

def test_ch05_stable_softmax():
    x = torch.tensor([1000., 999., 998.])
    assert torch.isnan(F.naive_softmax(x)).all()
    assert [round(v, 4) for v in F.stable_softmax(x).tolist()] == [0.6652, 0.2447, 0.0900]
    assert close(math.log(torch.finfo(torch.float32).max), 88.72, 0.01)
    assert close(math.log(torch.finfo(torch.float16).max), 11.09, 0.01)


def test_ch05_all_masked_rows():
    real, mask = F.padding_masks([3, 5], 5, "left")
    assert real.int().tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
    torch.manual_seed(0)
    q, k, v = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    bad = F.masked_attention(q, k, v, mask[0])
    good = F.masked_attention_safe(q, k, v, mask[0])
    assert torch.isnan(bad[:2]).all() and not torch.isnan(bad[2:]).any()
    assert (good[:2] == 0).all() and torch.allclose(good[2:], bad[2:])


def test_ch05_sink_softmax():
    w0 = F.softmax_with_sink([2.0, 1.0, 0.5], float("-inf"))
    w1 = F.softmax_with_sink([2.0, 1.0, 0.5], 0.0)
    w4 = F.softmax_with_sink([2.0, 1.0, 0.5], 4.0)
    assert close(w0.sum().item(), 1.0, 1e-12)
    assert [round(x, 4) for x in w1.tolist()] == [0.5793, 0.2131, 0.1293] and close(w1.sum().item(), 0.9216, 1e-4)
    assert close(w4.sum().item(), 0.1772, 1e-4)


def test_ch05_qk_norm_bound():
    assert close(F.qk_norm_logit_bound(128), 11.31, 0.01)


def test_ch05_one_pass_is_all_positions():
    a, b = F.all_position_loss_check()
    assert close(a, b, 1e-5)


def test_ch05_overflow_thresholds():
    def first_inf(d):
        lo, hi = 0.0, 200.0
        for _ in range(60):
            m = (lo + hi) / 2
            if torch.isinf(torch.exp(torch.tensor(m, dtype=d))):
                hi = m
            else:
                lo = m
        return hi
    assert close(first_inf(torch.float32), 88.72, 0.01)
    assert close(first_inf(torch.float16), 11.09, 0.01)
    assert close(first_inf(torch.bfloat16), 88.75, 0.01)
    x = torch.tensor([1000., 999.])
    assert close(torch.softmax(x, -1)[0].item(), 0.7311, 1e-4)
    assert torch.softmax(x.bfloat16(), -1)[0].item() == 0.5     # 999 rounds to 1000 in bf16
    a, b = F.all_position_loss_check()
    assert close(a, 3.96008, 1e-4)


# ==================================================== Ch 6 — multi-head attention

def test_ch06_fused_qkv_layouts():
    W = torch.arange(12.).view(12, 1)
    blocked = [t.flatten().tolist() for t in F.split_fused_qkv(W, 2, 2, "blocked")]
    per_head = [t.flatten().tolist() for t in F.split_fused_qkv(W, 2, 2, "per_head")]
    assert blocked[0] == [0, 1, 2, 3] and per_head[0] == [0, 1, 6, 7]
    torch.manual_seed(0)
    W = torch.randn(3 * 4 * 8, 32)
    conv = F.per_head_to_blocked(W, 4, 8)
    assert torch.equal(torch.cat(F.split_fused_qkv(W, 4, 8, "per_head")), conv)
    assert not torch.equal(conv, W)


def test_ch06_circuit_rank():
    torch.manual_seed(0)
    C, H, dh = 64, 4, 16
    Wq, Wk, Wv, Wo = torch.randn(C, H * dh), torch.randn(C, H * dh), torch.randn(C, H * dh), torch.randn(H * dh, C)
    for qk, ov in F.qk_ov_circuits(Wq, Wk, Wv, Wo, H, dh):
        assert qk.shape == (C, C) and torch.linalg.matrix_rank(qk) == dh
        assert torch.linalg.matrix_rank(ov) == dh


# ==================================================== Ch 7 — position

def test_ch07_partial_rope():
    torch.manual_seed(0)
    x = torch.randn(6, 64)
    y = F.apply_partial_rope(x, torch.arange(6), 16)
    assert torch.equal(y[:, 16:], x[:, 16:])                  # unrotated channels untouched
    q, k = torch.randn(64), torch.randn(64)

    def score(m, n):
        return (F.apply_partial_rope(q[None], torch.tensor([m]), 16)[0] @
                F.apply_partial_rope(k[None], torch.tensor([n]), 16)[0]).item()
    assert close(score(3, 5), score(100, 102), 1e-4)          # still relative


def test_ch07_mrope_reduces_to_rope_for_text():
    a = F.mrope_angles(7, 7, 7)
    cos, _ = build_rope_cache(8, 128)
    assert torch.allclose(a.cos().float(), cos[7, :64], atol=1e-6)
    b = F.mrope_angles(3, 5, 9)
    assert b[0].item() == 3.0 and b[16].item() == 5.0 * (1 / 10000 ** (32 / 128))


def test_ch07_interpolation_without_finetune():
    m = F.train_toy_copy_model()
    assert close(F.periodic_loss(m, 128)[1], 3.62, 0.01) and close(F.periodic_loss(m, 256)[1], 5.94, 0.01)
    assert close(F.periodic_loss_interpolated(m, 128), 5.45, 0.01)
    assert close(F.periodic_loss_interpolated(m, 256), 6.89, 0.01)
    assert close(F.mrope_angles(3, 5, 9)[40].item(), 0.0285, 1e-4)


# ==================================================== Ch 8 — the block

def test_ch08_softmax_saturation():
    rows = {s: F.softmax_saturation(s) for s in (1, 2, 4, 8, 16)}
    assert close(rows[1][0], 3.512, 0.001) and close(rows[1][1], 0.161, 0.001)
    assert close(rows[4][1], 0.916, 0.001) and close(rows[8][1], 0.998, 0.001)
    assert rows[16][2] < 1e-5 < rows[1][2]


def test_ch08_activation_memory():
    assert close(F.activation_bytes_per_layer(4096, 1, 4096, 32) / 1e9, 3.255, 0.001)
    assert close(F.activation_bytes_per_layer(4096, 1, 4096, 32, flash=True) / 1e9, 0.570, 0.001)


def test_ch08_block_flops():
    f = F.block_flops_per_token(4096, 11008, 4096, 32, 32, 128)   # Llama-2-7B at T = 4096
    assert f["mlp"] == 270_532_608 and f["q,o proj"] == 67_108_864 and f["scores+mix"] == 33_554_432
    assert close(f["mlp"] / f["total"], 0.617, 0.001)


def test_ch08_parallel_block_runs():
    cfg = Config(vocab_size=50, d_model=32, n_layer=1, n_head=4, n_kv_head=2, d_head=8, d_ff=64, max_T=16)
    blk = F.ParallelBlock(cfg)
    cos, sin = build_rope_cache(10, 8)
    assert blk(torch.randn(2, 10, 32), cos, sin).shape == (2, 10, 32)


def test_ch08_budget_totals():
    per = F.activation_bytes_per_layer(4096, 1, 4096, 32)
    assert close(per * 32 / 1e9, 104.2, 0.1)
    assert close(F.activation_bytes_per_layer(4096, 1, 4096, 32, flash=True) * 32 / 1e9, 18.25, 0.01)
    llama = 4096 * 5 + (4096 + 2 * 1024) + 32 + 3 * 14336        # exercise 8.5's element count
    assert llama == 69_664 and close(llama * 2 * 8192 * 32 / 1e9, 36.5, 0.05)


# ==================================================== Ch 9 — whole architectures

def test_ch09_encdec_flops():
    dec, encdec = F.encdec_vs_decoder_flops(220e6, 110e6, 110e6, 512, 128)
    assert dec == 2 * 220e6 * 640 and encdec == 2 * 110e6 * 640
    assert encdec * 2 == dec


def test_ch09_info_nce():
    q = [[1.0, 0.2, 0.0], [0.0, 1.0, 0.3], [0.2, 0.0, 1.0]]
    d = [[0.9, 0.5, 0.1], [0.1, 0.8, 0.5], [0.4, 0.1, 0.9]]
    l05, sim = F.info_nce(q, d, 0.05)
    l1, _ = F.info_nce(q, d, 1.0)
    assert close(sim[0, 0].item(), 0.9480, 1e-4) and close(sim[0, 1].item(), 0.2687, 1e-4)
    assert close(sim[0, 2].item(), 0.4160, 1e-4)
    assert l05 < 1e-4 and close(l1, 0.7593, 1e-4)


def test_ch09_worked_example_and_exercises():
    q = [[1.0, 0.2, 0.0], [0.0, 1.0, 0.3], [0.2, 0.0, 1.0]]
    d = [[0.9, 0.5, 0.1], [0.1, 0.8, 0.5], [0.4, 0.1, 0.9]]
    _, sim = F.info_nce(q, d, 1.0)
    assert [round(x, 3) for x in sim[0].softmax(-1).tolist()] == [0.477, 0.242, 0.280]
    hard = [[0.9, 0.5, 0.1], [0.95, 0.45, 0.1], [0.4, 0.1, 0.9]]
    l, s = F.info_nce(q, hard, 0.05)
    assert close(l, 0.777, 0.001) and s[0, 1] > s[0, 0]
    g = torch.Generator().manual_seed(0)
    res = {}
    for B in (8, 512):
        qq = torch.nn.functional.normalize(torch.randn(B, 256, generator=g), dim=-1)
        dd = qq + 0.05 * torch.randn(B, 256, generator=g)
        r1, r2 = torch.randn(B, 256, generator=g), torch.randn(B, 256, generator=g)
        res[B] = (F.info_nce(qq, dd)[0], F.info_nce(r1, r2)[0])
    assert close(res[8][0], 1.5e-6, 0.1e-6) and close(res[512][0], 1.9e-4, 0.1e-4)
    assert close(res[8][1], 3.23, 0.01) and close(res[512][1], 7.02, 0.01)
    assert close(math.log(512) + (1 / (0.05 * 16)) ** 2 / 2, 7.02, 0.01)
    assert close(1e7 * 1024 * 2 / 1e9, 20.48, 0.01)
    assert close(2 * 335e6 * 300 * 100, 2.0e13, 0.02e13)


# ==================================================== Ch 22 — linear attention, delta rule

def _qkvba(T=100, dk=16, dv=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, dk, generator=g, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(T, dk, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(T, dv, generator=g, dtype=torch.float64)
    b = torch.rand(T, generator=g, dtype=torch.float64)
    a = torch.rand(T, generator=g, dtype=torch.float64) * 0.5 + 0.5
    return q, k, v, b, a


def test_ch22_linear_attention_two_forms():
    q, k, v, _, _ = _qkvba()
    o, _ = LA.linear_attention_recurrent(q, k, v)
    assert torch.allclose(o, LA.linear_attention_parallel(q, k, v), atol=1e-10)


def test_ch22_chunked_equals_recurrent():
    q, k, v, b, a = _qkvba()
    for C in (1, 7, 16, 64, 100):
        o1, S1 = LA.gated_delta_rule_recurrent(q, k, v, b, a)
        o2, S2 = LA.chunked_delta_rule(q, k, v, b, a, chunk=C)
        assert torch.allclose(o1, o2, atol=1e-10) and torch.allclose(S1, S2, atol=1e-10)
        o1, _ = LA.delta_rule_recurrent(q, k, v, b)
        o2, _ = LA.chunked_delta_rule(q, k, v, b, chunk=C)
        assert torch.allclose(o1, o2, atol=1e-10)


def test_ch22_delta_is_a_gradient_step():
    torch.manual_seed(0)
    S = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn(3, dtype=torch.float64), dim=0)
    v, beta = torch.randn(4, dtype=torch.float64), 0.7
    (0.5 * (S @ k - v).pow(2).sum()).backward()
    sgd = S.detach() - beta * S.grad
    _, S_delta = LA.delta_rule_recurrent(k[None], k[None], v[None], torch.tensor([beta], dtype=torch.float64),
                                         S0=S.detach())
    assert torch.allclose(sgd, S_delta, atol=1e-12)


def test_ch22_overwrite():
    lin, delta, S_lin, S_delta = LA.overwrite_demo()
    assert lin == [2.0, 5.0] and delta == [-1.0, 4.0]
    assert S_lin == [[2.0, 0.0], [5.0, 2.0]] and S_delta == [[-1.0, 0.0], [4.0, 2.0]]


def test_ch22_recall_capacity():
    for n in (8, 32, 64):                                     # orthogonal keys: exact up to dk
        assert LA.recall_error(n, 64, "linear", "orthogonal") < 1e-10
        assert LA.recall_error(n, 64, "delta", "orthogonal") < 1e-10
    lin = {n: LA.recall_error(n, 64, "linear") for n in (8, 64, 256, 1024)}
    assert close(lin[8], 0.278, 0.002) and close(lin[64], 1.022, 0.002)
    assert close(lin[256], 1.983, 0.002) and close(lin[1024], 4.005, 0.002)
    for n in (64, 256, 1024):                                 # the sqrt((n-1)/dk) law
        assert abs(lin[n] / LA.interference_std(n, 64) - 1) < 0.04
    assert close(LA.recall_error(1024, 64, "delta"), 1.349, 0.002)
    assert LA.recall_error(1024, 64, "softmax") < 1e-6


def test_ch22_exercise_recall_by_age():
    for n, pred in ((32, 0.984), (128, 1.992), (512, 3.996)):
        assert close(LA.interference_std(n, 32), pred, 0.001)
    assert close(LA.recall_error(128, 32, "linear"), 2.130, 0.002)
    assert close(LA.recall_error(512, 32, "linear"), 3.989, 0.002)
    old, new = LA.recall_error_by_age(512, 32, "delta")
    assert close(old, 1.460, 0.002) and close(new, 0.970, 0.002)
    old, new = LA.recall_error_by_age(512, 32, "linear")
    assert close(old, 3.925, 0.002) and close(new, 4.014, 0.002)


def test_ch22_memory_arithmetic():
    assert LA.kv_cache_bytes_per_token(12, 2, 256) == 24_576
    assert LA.linear_state_bytes(36, 32, 128, 128) == 75_497_472
    hyb, dense = LA.qwen3_next_memory(262_144)
    assert close(hyb / 1e9, 6.518, 0.001) and close(dense / 1e9, 25.770, 0.001)
    assert LA.kv_cache_bytes_per_token(18, 8, 64) == 36_864        # gpt-oss-120b, dense layers
    assert LA.kv_cache_bytes_per_token(18, 8, 64) * 128 == 4_718_592
    assert close(LA.kv_cache_bytes_per_token(18, 8, 64) * 131072 / 1e9, 4.832, 0.001)
    # exercise 22.5
    assert LA.kv_cache_bytes_per_token(64, 8, 128) == 262_144 and LA.kv_cache_bytes_per_token(16, 8, 128) == 65_536
    assert LA.linear_state_bytes(48, 32, 128, 128) == 100_663_296
    assert LA.linear_state_bytes(48, 32, 128, 128) / (262_144 - 65_536) == 512


def test_ch22_flops():
    assert LA.chunked_delta_flops_per_token(64, 128, 128) == 155_648
    assert LA.softmax_attention_flops_per_token(4096, 128, 128) == 1_048_576
    assert LA.softmax_attention_flops_per_token(262144, 128, 128) == 67_108_864


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
