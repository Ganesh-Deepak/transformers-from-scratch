"""
Post-training and evaluation — the verification suite for Chapters 23-27.
Every test checks a number or a claim printed in those chapters.

    python tests/test_posttraining.py      # no pytest needed
    python -m pytest tests/test_posttraining.py -v
"""

import math
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfs import lora as L
from tfs import rl as RL
from tfs import evals as E
from tfs import dpo as D
from tfs.model import Config, Model

T = torch.tensor
close = lambda a, b, tol=1e-4: abs(float(a) - float(b)) < tol


# ==================================================== Ch 23 — LoRA mechanics

def test_lora_starts_as_the_pretrained_model():
    """23.6: B = 0, so the wrapped layer computes exactly the base layer at step 0."""
    torch.manual_seed(0)
    base = nn.Linear(64, 48, bias=False)
    x = torch.randn(5, 7, 64)                           # (B, T, in)
    ref = base(x)
    lo = L.LoRALinear(base, r=4, alpha=8)
    assert torch.equal(lo(x), ref)
    assert lo.A.shape == (4, 64) and lo.B.shape == (48, 4)


def test_lora_step0_gradient_reaches_B_only():
    """23.6: dL/dA = s B^T g x^T = 0 while B = 0; dL/dB = s g (Ax)^T != 0."""
    torch.manual_seed(0)
    lo = L.LoRALinear(nn.Linear(32, 16, bias=False), r=4, alpha=8)
    lo(torch.randn(10, 32)).pow(2).sum().backward()
    assert lo.A.grad.abs().max() == 0
    assert lo.B.grad.abs().max() > 0
    assert lo.base.weight.grad is None                  # frozen: no gradient at all


def test_lora_zero_init_both_is_a_saddle():
    """23.6: if A and B both start at 0, neither ever receives a gradient."""
    torch.manual_seed(0)
    lo = L.LoRALinear(nn.Linear(32, 16, bias=False), r=4)
    with torch.no_grad():
        lo.A.zero_()
    lo(torch.randn(10, 32)).pow(2).sum().backward()
    assert lo.A.grad.abs().max() == 0 and lo.B.grad.abs().max() == 0


def test_lora_only_adapters_train_on_the_course_model():
    """23.6: apply_lora on tfs.model freezes everything except A and B."""
    torch.manual_seed(0)
    cfg = Config(vocab_size=256, d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_head=16,
                 d_ff=128, max_T=32)
    m = Model(cfg)
    wrapped = L.apply_lora(m, ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down"], r=4)
    assert len(wrapped) == 7 * cfg.n_layer
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all(n.endswith(".A") or n.endswith(".B") for n in names)
    assert L.trainable_params(m) == L.lora_adapter_params(64, 2, 4, 2, 16, 128, 4)
    ids = torch.randint(0, 256, (2, 16))
    _, loss, _ = m(ids, ids)
    loss.backward()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n


def test_lora_merge_equals_unmerged_and_unmerge_restores():
    """23.6: merged and unmerged outputs agree; unmerge gives W back."""
    torch.manual_seed(0)
    base = nn.Linear(64, 32, bias=False)
    W0 = base.weight.detach().clone()
    lo = L.LoRALinear(base, r=8, alpha=16)
    with torch.no_grad():
        lo.B.normal_(std=0.1)                           # pretend we trained
    x = torch.randn(4, 64)
    y_unmerged = lo(x)
    lo.merge()
    assert torch.allclose(lo(x), y_unmerged, atol=1e-5)
    lo.unmerge()
    assert torch.allclose(lo.base.weight, W0, atol=1e-6)
    assert torch.allclose(lo(x), y_unmerged, atol=1e-5)


def test_lora_merge_on_whole_model():
    torch.manual_seed(0)
    cfg = Config(vocab_size=128, d_model=32, n_layer=2, n_head=2, n_kv_head=1, d_head=16,
                 d_ff=64, max_T=16)
    m = Model(cfg)
    L.apply_lora(m, ["q_proj", "v_proj"], r=2)
    for mod in L.lora_modules(m):
        with torch.no_grad():
            mod.B.normal_(std=0.05)
    ids = torch.randint(0, 128, (2, 12))
    a = m(ids)[0]
    L.merge_all(m)
    assert torch.allclose(m(ids)[0], a, atol=1e-5)


def test_rslora_scaling_keeps_update_size_constant_in_rank():
    """23.6: with random adapters, s=alpha/r shrinks the update ~1/sqrt(r); alpha/sqrt(r) does not."""
    torch.manual_seed(0)
    d, alpha = 512, 16
    x = torch.randn(256, d)
    std = {}
    for r in (4, 64):
        A = torch.randn(r, d) / math.sqrt(d)
        B = torch.randn(d, r)
        low = x @ A.T @ B.T                             # entries have variance ~ r
        std[r] = (float((alpha / r * low).std()), float((alpha / math.sqrt(r) * low).std()))
    ratio_lora = std[4][0] / std[64][0]
    ratio_rs = std[4][1] / std[64][1]
    assert 3.5 < ratio_lora < 4.5                       # sqrt(64/4) = 4
    assert 0.9 < ratio_rs < 1.1
    assert close(L.LoRALinear(nn.Linear(8, 8), r=16, alpha=16, rslora=True).scaling, 4.0)
    assert close(L.LoRALinear(nn.Linear(8, 8), r=16, alpha=16).scaling, 1.0)


def test_dora_starts_at_the_pretrained_weight():
    torch.manual_seed(0)
    W0 = torch.randn(16, 32)
    A, B = torch.randn(4, 32), torch.zeros(16, 4)
    m = W0.norm(dim=1)
    assert torch.allclose(L.dora_weight(W0, B, A, m, 2.0), W0, atol=1e-6)


def test_multi_lora_batch_matches_per_request_merged_weights():
    """23.6 / Ch 18: one base matmul, a different adapter per row."""
    torch.manual_seed(0)
    n_ad, r, d_in, d_out = 3, 4, 32, 24
    W = torch.randn(d_out, d_in)
    As, Bs = torch.randn(n_ad, r, d_in), torch.randn(n_ad, d_out, r)
    x = torch.randn(7, d_in)
    ids = torch.tensor([0, 2, 1, 1, 0, 2, 2])
    y = L.multi_lora_linear(x, W, As, Bs, ids, 0.5)
    for i in range(7):
        Wi = W + 0.5 * Bs[ids[i]] @ As[ids[i]]
        assert torch.allclose(y[i], Wi @ x[i], atol=1e-4)


# ==================================================== Ch 23 — LoRA arithmetic

def test_llama3_8b_parameter_count():
    c = L.llama_param_counts(128256, 4096, 32, 32, 8, 128, 14336)
    assert c["total"] == 8_030_261_248
    assert c["linear"] == 6_979_321_856 and c["embed"] == 1_050_673_152
    c70 = L.llama_param_counts(128256, 8192, 80, 64, 8, 128, 28672)
    assert round(c70["total"] / 1e9, 2) == 70.55


def test_lora_adapter_counts():
    assert L.lora_params_per_matrix(4096, 4096, 16) == 131_072        # vs 16,777,216 for full
    all7 = L.lora_adapter_params(4096, 32, 32, 8, 128, 14336, 16)
    assert all7 == 41_943_040
    assert round(100 * all7 / 8_030_261_248, 2) == 0.52
    assert L.lora_adapter_params(4096, 32, 32, 8, 128, 14336, 16, targets=("q_proj", "v_proj")) == 6_815_744
    # the capstone model (Ch 27)
    assert Config().n_params() == 12_587_904
    assert L.lora_adapter_params(384, 6, 6, 2, 64, 1024, 8) == 325_632
    assert L.lora_adapter_params(384, 6, 6, 2, 64, 1024, 8, targets=("q_proj", "v_proj")) == 61_440


def test_finetune_memory_full_lora_qlora():
    c = L.llama_param_counts(128256, 4096, 32, 32, 8, 128, 14336)
    n = L.lora_adapter_params(4096, 32, 32, 8, 128, 14336, 16)
    assert round(L.finetune_memory_gb(c, n, "full"), 1) == 128.5
    assert round(L.finetune_memory_gb(c, n, "lora"), 1) == 16.7
    assert round(L.finetune_memory_gb(c, n, "qlora"), 1) == 6.4
    assert close(L.NF4_BITS, 4.127, 1e-3)
    assert round(c["linear"] * L.NF4_BITS / 8 / 1e9, 2) == 3.60
    c70 = L.llama_param_counts(128256, 8192, 80, 64, 8, 128, 28672)
    n70 = L.lora_adapter_params(8192, 80, 64, 8, 128, 28672, 16)
    assert [round(L.finetune_memory_gb(c70, n70, m)) for m in ("full", "lora", "qlora")] == [1129, 144, 43]


# ==================================================== Ch 23 — GAE and KL shaping

R4 = T([-1.2, -0.5, -2.0, -0.3]), T([-1.0, -0.7, -2.5, -0.3])
V4 = T([0.5, 0.6, 0.7, 0.9, 0.0])


def test_kl_shaped_rewards_sum_to_sequence_objective():
    logp, ref = R4
    r = RL.kl_shaped_rewards(logp, ref, score=1.0, beta=0.1)
    assert torch.allclose(r, T([0.02, -0.02, -0.05, 1.0]), atol=1e-6)
    assert close(r.sum(), 1.0 - 0.1 * float((logp - ref).sum()))           # 0.95
    assert close(r.sum(), 0.95)


def test_gae_recursion_equals_definition_and_limits():
    logp, ref = R4
    r = RL.kl_shaped_rewards(logp, ref, 1.0, 0.1)
    a95, ret = RL.gae(r, V4, 1.0, 0.95)
    assert torch.allclose(a95, RL.gae_explicit(r, V4, 1.0, 0.95), atol=1e-6)
    assert torch.allclose(a95, T([0.4171, 0.3127, 0.2450, 0.1000]), atol=1e-4)
    a0, _ = RL.gae(r, V4, 1.0, 0.0)                                          # TD(0): the deltas
    assert torch.allclose(a0, T([0.12, 0.08, 0.15, 0.10]), atol=1e-6)
    a1, _ = RL.gae(r, V4, 1.0, 1.0)                                          # Monte Carlo
    assert torch.allclose(a1, RL.discounted_returns(r) - V4[:-1], atol=1e-6)
    assert torch.allclose(a1, T([0.45, 0.33, 0.25, 0.10]), atol=1e-6)
    assert torch.allclose(ret, a95 + V4[:-1])


def test_forward_vs_reverse_kl_prefer_different_students():
    """23.3: forward KL prefers the student that covers both modes; reverse KL the one that picks one."""
    p = T([0.45, 0.05, 0.45, 0.05])
    cover, mode = T([0.25] * 4), T([0.85, 0.05, 0.05, 0.05])
    assert round(RL.forward_kl(p, cover), 4) == 0.3681 and round(RL.forward_kl(p, mode), 4) == 0.7026
    assert round(RL.reverse_kl(p, cover), 4) == 0.5108 and round(RL.reverse_kl(p, mode), 4) == 0.4307


def test_rejection_sampling_yield():
    assert [round(RL.rejection_yield(0.2, k), 3) for k in (1, 4, 8, 16, 32)] == [0.2, 0.59, 0.832, 0.972, 0.999]


# ==================================================== Ch 24 — online DPO

REF8 = T([0.30, 0.30, 0.20, 0.14, 0.02, 0.02, 0.01, 0.01]).log()
REW8 = torch.arange(8.) / 7


def test_bandit_dpo_gradient_matches_autograd_and_ignores_unseen():
    pairs = [(3, 0), (2, 1), (3, 1)]
    logits = (REF8 + torch.randn(8, generator=torch.Generator().manual_seed(1)) * 0.3).requires_grad_(True)
    lp = D.bandit_logps(logits)
    w, l = T([p[0] for p in pairs]), T([p[1] for p in pairs])
    loss, *_ = D.dpo_loss(lp[w], lp[l], REF8[w], REF8[l], 0.1)
    loss.backward()
    g = D.bandit_dpo_grad(logits.detach(), REF8, pairs, 0.1)
    assert torch.allclose(g, logits.grad, atol=1e-6)
    assert g[4:].abs().max() == 0                      # responses 4..7 are in no pair
    assert close(g.sum(), 0.0, 1e-7)                   # equal and opposite pushes


def test_online_dpo_beats_offline_dpo_at_equal_budget():
    assert round(D.bandit_expected_reward(REF8, REW8), 3) == 0.204
    off, on = [], []
    for seed in range(40):
        lo, _ = D.offline_dpo(REF8, REW8, seed=seed)
        ln, _, _ = D.online_dpo(REF8, REW8, seed=seed)
        off.append(D.bandit_expected_reward(lo, REW8))
        on.append(D.bandit_expected_reward(ln, REW8))
    assert round(sum(off) / 40, 3) == 0.395 and round(sum(on) / 40, 3) == 0.483
    assert sum(b > a for a, b in zip(off, on)) == 40


# ==================================================== Ch 25 — Dr. GRPO, GSPO, process rewards, agents

def test_grpo_length_normalisation_bias():
    A = RL.grpo_advantages([1, 1, 0, 0])
    lens = [100, 300, 200, 1000]
    g = RL.token_weights(A, lens, "grpo")
    assert [round(x * 1e4, 2) for x in g] == [25.0, 8.33, -12.5, -2.5]
    assert close(g[2] / g[3], 5.0, 1e-3)               # the long wrong answer is penalised 5x less per token
    dr = RL.token_weights(RL.grpo_advantages([1, 1, 0, 0], False), lens, "dr_grpo", max_len=1000)
    assert [round(x * 1e4, 2) for x in dr] == [1.25, 1.25, -1.25, -1.25]
    dapo = RL.token_weights(A, lens, "dapo")
    assert len({round(abs(x), 8) for x in dapo}) == 1  # one weight per token, sign by advantage


def test_grpo_std_normalisation_bias():
    one = [1, 0, 0, 0, 0, 0, 0, 0]
    half = [1, 1, 1, 1, 0, 0, 0, 0]
    assert round(float(RL.grpo_advantages(one)[0]), 3) == 2.645
    assert round(float(RL.grpo_advantages(one)[1]), 3) == -0.378
    assert RL.grpo_advantages(one, False).tolist() == [0.875] + [-0.125] * 7
    assert RL.grpo_advantages(half, False).tolist() == [0.5] * 4 + [-0.5] * 4
    s1 = float(T(one, dtype=torch.float64).std(unbiased=False))
    s2 = float(T(half, dtype=torch.float64).std(unbiased=False))
    assert round(s1, 4) == 0.3307 and round((1 / s1) / (1 / s2), 2) == 1.51


def test_gspo_sequence_ratio_is_geometric_mean():
    lr = T([[0.10, -0.05, 0.40, -0.25]])
    tok = RL.token_ratios(lr, torch.zeros_like(lr))[0]
    assert [round(float(x), 3) for x in tok] == [1.105, 0.951, 1.492, 0.779]
    assert int(((tok < 0.8) | (tok > 1.2)).sum()) == 2                    # PPO/GRPO would clip 2 of 4
    s = RL.gspo_ratio(lr, torch.zeros_like(lr), torch.ones_like(lr))
    assert round(float(s), 4) == 1.0513
    assert close(s, torch.exp(lr.mean()))
    # padding does not count toward the length
    lr2 = T([[0.10, -0.05, 0.40, -0.25, 9.0]])
    m2 = T([[1., 1., 1., 1., 0.]])
    assert close(RL.gspo_ratio(lr2, torch.zeros_like(lr2), m2), s)


def test_process_reward_finds_the_bad_step():
    vals = RL.mc_step_values([[1] * 6 + [0] * 2, [1] * 5 + [0] * 3, [1] + [0] * 7, [0] * 8])
    assert vals == [0.75, 0.625, 0.125, 0.0]
    adv = RL.step_advantages(vals, 0.7)
    assert [round(a, 3) for a in adv] == [0.05, -0.125, -0.5, -0.125]
    assert min(range(4), key=lambda i: adv[i]) == 2                       # step 3 is the mistake


def test_agent_loss_mask_counts_only_policy_tokens():
    m = RL.agent_loss_mask([("system", 20), ("user", 30), ("assistant", 40), ("tool", 200),
                            ("assistant", 25), ("tool", 150), ("assistant", 35)])
    assert m.shape == (500,) and int(m.sum()) == 100
    assert m[50:90].all() and not m[90:290].any()


# ==================================================== Ch 26 — evaluation

def test_benchmark_resolution():
    assert round(E.standard_error(0.70, 198), 4) == 0.0326
    assert round(100 * 1.959964 * E.standard_error(0.70, 198), 1) == 6.4
    assert round(100 / 30, 2) == 3.33                                       # one AIME problem
    assert round(100 * 1.959964 * E.standard_error(0.5, 30), 1) == 17.9
    assert round(100 * 1.959964 * E.standard_error(0.85, 14042), 2) == 0.59
    assert round(100 * E.mde_unpaired(0.70, 198), 1) == 12.9
    assert round(100 * E.mde_unpaired(0.85, 14042), 1) == 1.2
    lo, hi = E.ci95(0.70, 198)
    assert round(lo, 3) == 0.636 and round(hi, 3) == 0.764


def test_items_needed_for_a_two_point_gain():
    assert E.n_items_unpaired(0.70, 0.72) == 8077
    assert E.n_items_paired(0.10, 0.02) == 1955


def test_pass_at_k_unbiased_estimator():
    assert close(E.pass_at_k(10, 3, 1), 0.3)
    assert round(E.pass_at_k(10, 3, 5), 4) == 0.9167                       # 1 - 21/252
    assert close(E.pass_at_k(10, 3, 5), 1 - math.comb(7, 5) / math.comb(10, 5), 1e-12)
    assert round(E.naive_pass_at_k(10, 3, 5), 4) == 0.8319
    # unbiasedness, exactly: average the estimator over c ~ Binomial(n, p)
    for n, p, k in [(10, 0.3, 5), (20, 0.1, 10), (16, 0.6, 4)]:
        assert close(E.expected_estimator(E.pass_at_k, n, p, k), 1 - (1 - p) ** k, 1e-9)
        assert E.expected_estimator(E.naive_pass_at_k, n, p, k) < 1 - (1 - p) ** k
    assert round(E.expected_estimator(E.naive_pass_at_k, 10, 0.3, 5), 3) == 0.760
    assert E.pass_at_k(5, 0, 3) == 0.0 and E.pass_at_k(5, 3, 3) == 1.0


def test_pass_hat_k_for_agents():
    assert close(E.pass_hat_k(10, 8, 4), 1 / 3)
    for n, p, k in [(10, 0.8, 4), (12, 0.6, 8)]:
        assert close(E.expected_estimator(E.pass_hat_k, n, p, k), p ** k, 1e-9)
    assert round(0.6 ** 8, 4) == 0.0168


def test_exact_match_and_f1():
    assert E.exact_match("The Eiffel Tower.", "Eiffel Tower") == 1
    assert E.exact_match("The Eiffel Tower.", "Eiffel Tower", normalize=False) == 0
    assert close(E.token_f1("the tower in Paris", "Eiffel Tower"), 0.4)
    assert E.majority_vote(["42", "41", "42.", "forty-two", "42"]) == "42"


def test_paired_bootstrap_is_tighter_than_unpaired():
    A = [1] * 350 + [0] * 150
    B = [1] * 340 + [0] * 10 + [1] * 25 + [0] * 125
    assert sum(A) == 350 and sum(B) == 365
    assert E.bootstrap_ci(A) == (0.66, 0.74)
    lo, hi = E.paired_bootstrap_diff(A, B)
    assert (round(lo, 3), round(hi, 3)) == (0.008, 0.054) and lo > 0
    ulo, uhi = E.unpaired_bootstrap_diff(A, B)
    assert (round(ulo, 3), round(uhi, 3)) == (-0.024, 0.086) and ulo < 0


def test_contamination_signals():
    corpus = ("the quick brown fox jumps over the lazy dog and then the fox runs into the "
              "forest to find food").split()
    cg = E.ngrams(corpus, 5)
    assert round(E.ngram_overlap("a quick brown fox jumps over the lazy dog today", cg, 5), 3) == 0.667
    assert round(E.ngram_overlap("the fox jumps over the lazy cat and runs away", cg, 5), 3) == 0.167
    seen = [-0.2, -0.5, -0.1, -1.1, -0.3, -0.4, -0.9, -0.2, -0.6, -0.3]
    unseen = [-0.4, -3.9, -0.2, -2.7, -0.6, -5.1, -0.8, -0.3, -1.9, -0.5]
    assert close(E.min_k_prob(seen, 0.2), -1.0) and close(E.min_k_prob(unseen, 0.2), -4.5)


def test_position_swap_debiasing():
    assert E.debias_pairwise("first", "second") == "win"
    assert E.debias_pairwise("first", "first") == "tie"       # it just likes slot one
    assert E.debias_pairwise("second", "first") == "loss"


def test_length_controlled_winrate_removes_a_pure_length_effect():
    rng = random.Random(0)
    lm, lb, wins = [], [], []
    for _ in range(805):
        b = rng.randint(150, 450)
        lb.append(b)
        lm.append(max(20, int(b + rng.gauss(120, 150))))
    sd = (sum((a - b) ** 2 for a, b in zip(lm, lb)) / 805) ** 0.5
    for a, b in zip(lm, lb):                                   # the judge rewards ONLY length
        wins.append(int(rng.random() < 1 / (1 + math.exp(-1.5 * math.tanh((a - b) / sd)))))
    res = E.length_controlled_winrate(wins, lm, lb)
    assert round(res["raw"], 3) == 0.620
    assert round(res["lc"], 3) == 0.474
    assert round(res["phi"], 2) == 1.54


def test_elo_and_kappa():
    assert round(E.elo_win_prob(100), 3) == 0.640 and round(E.elo_win_prob(400), 3) == 0.909
    assert round(E.elo_win_prob(30), 3) == 0.543
    a, b = list("YYYYYYYYYYYYYYNNNNNN"), list("YYYYYYYYYYYYNNYYNNNN")
    assert close(sum(x == y for x, y in zip(a, b)) / 20, 0.8)
    assert round(E.cohens_kappa(a, b), 3) == 0.524


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
