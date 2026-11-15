# Content audit — Transformers From Scratch (November 2026)

Read as a book for a working AI engineer: does each chapter explain, in order, with
every claim checkable, and is it complete and current? Each entry gives a verdict,
what was **fixed in this pass**, and what is **missing**, ordered by value.

Verdicts: **A** worth the read as is · **B** solid, needs the listed additions ·
**C** too thin for its importance, needs a rewrite to the Ch 0 / Ch 20 standard.

## Site-wide findings (fixed and now enforced by tests)

| Finding | Fix | Now enforced by |
|---|---|---|
| 13 chapters promised a companion notebook that did not exist (03, 04, 07, 08, 10, 11, 13, 14, 18, 19, 21, 23, 25) | Marked as planned (`<code class="planned">`) until written | `check_ui.js` CONTENT INTEGRITY: every referenced repo file must exist or be marked planned |
| Ch 10 "built on" Ch 11 | Moved to "Leads to" | `check_ui.js`: "Builds on" may only point backwards |
| Ch 20 (MoE) quoted a nonexistent notebook and had a wrong aux-loss-free code sample | Chapter rewritten; `tfs/moe.py`, 9 tests, `notebooks/20_moe.ipynb` | `tests/test_tfs.py`, `tests/run_notebooks.py` |

**The standard to hold every chapter to** (what Ch 0, Ch 12 and the new Ch 20 do):
every number in the text is computed by code that is in the repo and asserted in a test or
notebook; each chapter has a "whole chapter in N sentences" box, worked examples with every
number shown, 3+ quiz questions per major section, and exercises that include at least one
systems/cost estimate, one implementation and one "predict, then measure".

## Chapter by chapter

### Ch 00 · Numbers in a Machine — **A**
Written to the standard in October 2026 (8.1k words, 12 exercises, numfmt-tested). No action.

### Ch 01 · Tensors, Shapes — **A** (raised November 2026)
3.5k → 5.4k words, 8 → 11 quiz questions, 4 → 6 exercises. Added: stride-0 `expand` vs `repeat`, why
`expand`+`reshape` (HF `repeat_kv`) copies anyway and what that costs for a 70B-shaped layer (16.8 → 134 MB);
in-place ops and autograd (both error messages reproduced, version counters, writes through views); a type-promotion
table from `torch.result_type` incl. the 0-dim and Python-scalar exceptions and the bf16 + fp32 silent upcast;
Checkpoint 3; exercise 1.5 reads Llama-3-8B's real forward trace (built on the meta device from its config);
exercise 1.6 predict-then-measure on strides and versions. Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.
### Ch 02 · Matrix Multiplication — **A** (raised November 2026)
Earlier fix kept (MFU in the H100 estimate). 3.1k → 4.7k words, 5 → 8 quiz questions, 4 → 6 exercises. Added: one
layer's FLOPs derived matmul by matmul (24d² weights + 4Td attention; GPT-2 small numbers; the T = 6d crossover);
new §2.6 on arithmetic intensity (M = 1 … 4096 against the H100 ridge; decode is M = 1) and tile alignment
(tile utilisation, 50,257 → 50,304, Llama 3 / Qwen2.5 vocab sizes as multiples of 128); Checkpoint 3; exercises
on the batch size that makes decode compute-bound (M ≈ 326) and a 4096-vs-4097 predict-then-measure.
Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.
### Ch 20 · Mixture of Experts — rewritten, **A**
2.1k → ~7.5k words. New: three router types and their equivalence, the top-1 zero-gradient
trap, sorted dispatch / grouped GEMM, capacity and measured drop rates, aux loss derived,
z-loss, global-batch balancing, DeepSeek-V3 bias rule (code corrected), expert choice and why
it breaks causality, fine-grained and shared experts with combinatorics, expert parallelism and
the all-to-all bill, node-limited routing, the serving fact that batched decode reads most
experts, training stability, upcycling, a 7-model spec table with arithmetic cross-checks,
8 quiz questions, 7 exercises. Every number from `tfs/moe.py` / `notebooks/20_moe.ipynb`.
Interactive router widget added in §20.4 (`data-viz="moe"`): tokens × experts with top-k picks, live load
bars, a DeepSeek-V3 bias-balancing toggle that takes the load CV from ~2.1 to under half over 40 steps, and the
experts-read-per-decode-step curve vs batch; interaction-tested in `check_interaction.js`.

### Ch 03 · Tokenization — **A** (raised November 2026)
Earlier fixes kept. 2.7k → 5.7k words, 4 → 8 quiz questions, 3 → 6 exercises. All six items addressed:
(1) §3.6 streaming detokenization with a worked byte-level example and a `StreamingDetokenizer`; (2) special-token
injection shown with the repo tokenizer (`allow_special`), tiktoken vs HF defaults; (3) new §3.3: cl100k
pre-tokenizer vs GPT-2 on the same string, NFKC (lossy, computed), Unigram vs BPE; (4) vocabulary as a scaling
decision computed as FLOPs per character for Llama-3-8B's body at 32k vs 128k (15% cheaper, from the Llama 3
paper's 3.17 → 3.94 chars/token), Tao et al. 2024's 216K, a measured vocab sweep on a fixed corpus fixture
(`tests/fixtures/ch03_corpus.txt`), BLT; (5) token healing as a mechanism with a measured example; (6) a measured
8-language fertility table. Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.
### Ch 04 · Embeddings and the Residual Stream — **A** (raised November 2026)
Earlier AdamW fix kept. 2.3k → 4.8k words, 3 → 9 quiz questions, 3 → 6 exercises. Added: a whole-chapter box; §4.5
residual-stream growth measured on `tfs.model` at init (2.9× with GPT-2's scaled init vs 16× without, and at 24/48
layers), pre-norm dilution, why tied models scale embeddings by √d; §4.6 Gemma 2 soft-capping and PaLM z-loss
(computed), vocab padding; a reproducible logit-lens run on a 2-layer model trained in the test suite
(chance → chance → 100%: induction composition) plus the tuned-lens caveat (the promise of a notebook that does not
exist removed); §4.8 the 5-features-in-2-dimensions superposition toy (2 / 4 / 5 features, a pentagon at 72°); two
checkpoints; exercises on depth scaling, logits memory (8.4 GB bf16) and importance-weighted superposition.
Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.
### Ch 05 · Attention From Scratch — **A** (raised November 2026)
Earlier complexity-table fix kept. 3.3k → 5.9k words, 3 → 7 quiz questions, 4 → 6 exercises. All four items:
(1) §5.9 left vs right padding, position ids under left padding, the all-masked-row NaN reproduced and fixed;
(2) §5.10 the QK-norm bound (√d·g² = 11.3 at d = 128), soft-capping, attention sinks and the off-by-one / learned-sink
softmax with computed weights; (3) the stable softmax, with overflow thresholds per dtype; (4) the all-positions
cross-entropy derived and checked against prefix-by-prefix evaluation on `tfs.model` (3.96008 both). Exercise 5.6's
measurement showed bf16 rounds 999 to 1000 (softmax [0.5, 0.5]); the answer says so. Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.
### Ch 06 · Multi-Head Attention — **A**
2.6k → 4.1k words, 3 → 6 quiz questions, 4 → 6 exercises. Added the two small items: §6.5 QK and OV circuits
(rank ≤ d_head, checked) and §6.6 fused-QKV layouts (blocked vs NeoX per-head vs separate, a 12-row worked example,
conversion with a test), plus GPT-2's Conv1D transposition and the Llama RoPE row permutation; a whole-chapter box,
Checkpoint 2, a conversion exercise and a previous-token-head exploration. Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.
### Ch 07 · Position — **A** (raised November 2026)
Earlier exercise 7.3 fixes kept. 2.7k → 4.6k words, 3 → 6 quiz questions, 3 → 5 exercises. Added: §7.6 partial and
decoupled RoPE (code, tested relative property, GPT-J / NeoX / Phi-2 / DeepSeek configs); §7.7 M-RoPE (Qwen2-VL
sections; reduces exactly to 1-D RoPE for text, tested); §7.8 a measured length-extrapolation table from the
repo's own model (it fails past the trained length even at a trained copy distance, and why); exercise 7.4 measures
position interpolation without fine-tuning (it makes this model worse: 5.45 vs 3.62), and 7.5 costs partial RoPE.
A table rather than a plot (there is no plotting component). Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.
### Ch 08 · The Transformer Block — **A** (raised November 2026)
2.6k → 5.0k words, 3 → 7 quiz questions, 4 → 7 exercises. All three items: (1) §8.6 logit growth → saturation →
vanishing softmax Jacobian → spikes, with a measured saturation table, QK-norm and sandwich/output norms
(Dehghani 2023, Wortsman 2023, OLMo 2, Gemma 2); (2) §8.7 parallel blocks (GPT-J, PaLM's ~15% and 8B vs 62B), with a
tested `ParallelBlock`; (3) §8.8 the per-block FLOP budget (Llama-2-7B, MLP 61.7%) and Korthikanti activation memory
(3.25 GB/layer, 0.57 GB with FlashAttention), linked to Ch 10, 17 and 19. The "measured plot" is a table.
Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.
### Ch 09 · Whole Architectures + Shape Trace — **A**
Earlier parameter-count fixes kept. 1.9k → 3.8k words, 3 → 6 quiz questions, 4 → 6 exercises. Added both items:
§9.5 decoder-only vs encoder-decoder FLOPs (220M vs 110M + 110M, the T5 "2P at equal cost" result) and what the count
omits; §9.6 bi-encoders vs cross-encoder rerankers, InfoNCE with in-batch negatives, a worked example showing
temperature and a hard negative (10⁻⁴ → 0.777); a whole-chapter box, Checkpoint 2, a retrieval sizing exercise
and an InfoNCE chance-level exercise (measured: chance is ln B + σ²/2, not ln B). Every new number is computed in `tfs/foundations.py` and asserted in `tests/test_foundations.py`.

### Ch 10 · The Backward Pass — **A** (raised to the standard, November 2026)
Every derivation checks numerically (softmax Jacobian at [1, 2, 0.5]; p − y at [2, 1, 0.1]; the RMSNorm
correction term and its orthogonality to x; attention backward). This pass added (2.6k → 6.1k words, 3 → 10 quiz
questions, 4 → 8 exercises), all numbers from `tfs/memory.py`, asserted in `tests/test_training.py`:
1. §10.7 memory of backward: what a Block saves, as a formula (6C + 2 + 4·H·dh + H + 4·d_ff per token) that matches
   autograd's saved tensors byte for byte (1,018,880); the GQA-expansion and shared-storage details; Llama-3-8B at 8k
   = 51.6 GB; checkpointing derived (k* = √(L·b/(a+b)) = 1.13, so per-layer, 3.83 GB, ~33% compute).
2. §10.8 the logits: 10.5 GB at the end for V = 128k; `chunked_cross_entropy` (Liger-style, exact to 1e-12, saves no
   N·V tensor; measured).
3. §10.9 embedding backward (scatter-add), SwiGLU backward, RoPE backward (inverse rotation), each autograd-checked.
4. §10.10 low-precision gradients: bf16 running total of 64 micro-batches 7.4e-3 vs fp32 1.1e-7; where frameworks
   keep sums in fp32.
Remaining nice-to-have: a measured GPU peak-memory plot for exercise 10.7.

### Ch 11 · Training a Transformer — **A** (raised to the standard, November 2026)
Real measured runs, the bits-per-byte lesson and the double-shift case study kept. Fixed earlier: GPT-3 175B's LR.
This pass (3.4k → 9.8k words, 3 → 13 quiz questions, 3 → 7 exercises), code in `tfs/data.py` and `tfs/trainkit.py`,
asserted in `tests/test_training.py`:
1. §11.6 dedup: exact hashing; MinHash with a worked example (J = 0.873 / 0.735, 101/112 and 85/112 slots, unbiased over
   200 seeds); LSH S-curve with FineWeb's 14 × 8 and a caught vs a missed copy; FineWeb's per-crawl-vs-global finding;
   Llama 3's URL / MinHash / line-level pipeline.
2. §11.7 quality: Gopher rules implemented; FineWeb-Edu's distil-a-judge recipe and Llama 3's classifiers; mixture
   weights by temperature with epochs per source; Llama 3's final mix; annealing in WSD (Llama 3, OLMo 2 souping,
   DeepSeek-V3's schedule).
3. §11.8 batch size: noise scale (estimator implemented, measured on the Ex 11.1 model), the S/E trade-off and B_crit,
   batch ramps of GPT-3, Llama 3 405B, DeepSeek-V3.
4. §11.9 μP: measured one-Adam-step output change 0.025 → 0.332 (standard) vs flat (μP) from width 256 to 4096.
5. §11.10 MFU: exact FLOPs/token for the Ex 11.1 model, measured throughput and MFU on CPU, profiler recipe.
6. §11.12 spikes: z-loss, QK-norm, OLMo 2's stability list, PaLM's roll-back-and-skip, a spike detector.
7. §11.13 exact resumption: what state, and a test that resumes bit-identically (and fails if one integer is lost).
Also fixed: the checkpointing row said "~√L" (it is per-layer for transformers; Ch 10 §10.7); an axis colour used for
"warmup" in the schedule figure.

### Ch 12 · Precision in Training — audited November 2026: **A−** → **A**
Audited this pass: the underflow trajectory, loss-scale table, matmul errors and accumulation table reproduce from
`notebooks/12_precision.ipynb`; bit layouts and finfo values check. Missing before this pass: (1) fp8 scaling done by
hand, with the granularity trade measured; (2) delayed scaling and its saturation failure; (3) pure-bf16 weights and the
two fixes (stochastic rounding, Kahan); (4) links to where precision meets gradient sums (Ch 10), cluster-scale FP8
(Ch 19) and precision scaling laws (Ch 21); (5) only 3 quiz questions for 6 sections and no whole-chapter summary.
All added (4.1k → 5.9k words, 3 → 8 quiz questions, 5 → 7 exercises), code in `tfs/fp8.py`, asserted in
`tests/test_training.py`: §12.7 per-tensor vs per-128 e4m3 (2.64% vs 2.58% at a 100× outlier; 3.63% / 4.6% zeros vs
2.59% / 0.17% at 10⁴×: floating fp8 tolerates outliers that would break int8), delayed scaling clipping 5.0 to 3.4;
§12.8 1.0 + 1000 × 0.001 in bf16 = 1.0 (nearest), 2.005 (stochastic, mean of 20), 2.0 (Kahan).

### Ch 13 · Decoding — **A** (raised November 2026)
All five missing items done, with reference code in `tfs/decoding.py` and every number asserted in
`tests/test_inference.py`. §13.4 rewritten: how a grammar engine turns a character automaton into a token mask,
a worked 22-token example (2 / 7 legal tokens, the boundary-crossing `7}` that a per-terminal engine bans, 24
token paths for one 9-character string), token healing, XGrammar's context-independent masks. New §13.5
best-of-n / self-consistency (exact plurality-vote accuracy, including the case where voting makes it worse:
0.35 → 0.236 at n = 41), §13.6 stop sequences while streaming (hold-back algorithm, traced), §13.7
batch-dependent nondeterminism (bit-exact fp32 split-K sums; a real argmax flip between 1 and 8 splits).
Whole-chapter box, two new checkpoints (3 + 4 questions), exercises 13.4–13.7 (cost estimate, implementation,
predict-then-measure, hard).

### Ch 14 · The GPU — **A** (raised November 2026)
§14.4 was fixed earlier (see the October note below). Added: §14.7 MFU (with the Llama 3 38–43% reference)
and the overhead-bound regime with a computed launch floor (355 kernels, 1.42 ms vs 4.79 ms for an 8B model;
7× over the memory floor for a 125M one) and CUDA graphs; §14.8 fp8/fp4 tensor-core rates (dense) and ridges,
and why the saturating decode batch does not move with precision; §14.9 the interconnect ladder (NVLink 4/5,
PCIe 5, IB NDR, a ring all-reduce 3.89 vs 35.0 ms); §14.10 a Triton fused-softmax introduction. Checkpoint 2
(4 questions), exercises 14.5–14.7. Code: `tfs/gpu.py`.

### Not yet audited in this pass
Ch 15–19, 21–26 (Ch 0 was rewritten in October; Ch 12 is audited above). Add a check_ui rule rejecting raw LaTeX outside code.

### Ch 14 · §14.4 — FIXED (October pass)
Rewritten: tensor cores require multiples of 16 (32 for fp8); 64/128 dominate because attention kernels ship
tuned paths for them, not because of cache lines. Real counterexamples cited (Phi-2 80, Phi-3-mini 96,
DeepSeek-V3 192, Gemma 256). `check_ui.js` fails on raw LaTeX in page text.

### Ch 15 · The KV Cache — **A** (raised November 2026)
Units are now consistent: KiB/GiB for cache sizes, GB for vendor figures, with a margin note; this corrected
three answers (70B at 8k: 66 users, not ~70; exercise 15.2: 13 and 8 H100s, not 12 and 7; quiz: 40 users from
40 GiB). New §15.6 measured cached-vs-uncached timing from the repo's own model, drawn by a new `series` widget
(11–14 ms flat vs 13 → 156 ms per token; 5.5 s vs 30.7 s for 480 tokens), with the overhead-bound reading.
§15.7 chunked prefill, exact, and the silent bug in `tfs.model.Model` (top-left `is_causal` on a multi-token
chunk: 0.34 logit error), fixed path in `tfs/kvcache.py`. §15.8 eviction: StreamingLLM sinks, H2O, SnapKV, with a
computed mass-kept table. Map table extended. Checkpoint 2, exercises 15.4–15.6.

### Ch 16 · MQA, GQA, MLA — **A** (completed November 2026)
Added: absorbed MLA decode shown to be MQA over a 576-wide latent, with code that checks absorbed = naive
(`tfs/mla.py`), the arithmetic intensity table (241.8 vs 8 vs 1 FLOP/byte), FlashMLA's reported figures, and why
prefill decompresses (3.4× the FLOPs for 57× fewer bytes). New §16.5 cross-layer sharing: CLA (2× / 2.9×) and
YOCO (21.3× at 128k), Character.AI's production use. Whole-chapter box, checkpoint 3, exercises 16.4–16.5.

### Ch 17 · FlashAttention — **A** (raised November 2026)
§17.4 pseudo-code now has the in-tile causal mask on diagonal blocks and skips future blocks before loading
them, with a box on the two granularities; `tfs/flash.py` runs it, tested at four tile shapes. Exercise 17.2's
stray closing tags removed. FA4 row and box re-checked against the paper (arXiv 2603.05451, MLSys 2026): up to
1,613 TFLOP/s bf16 on B200 (71%), 1.3× cuDNN 9.13, 2.7× Triton; the unverified "1.2–3.2× FlexAttention" and
"requires Hopper or Blackwell" claims removed. New §17.7 Flash-Decoding / split-KV with occupancy arithmetic and a
worked merge (36.881 both ways), §17.8 FlexAttention examples (mask_mod / score_mod, block sparsity 48% / 79% /
91%, tested against SDPA). Checkpoint 2, exercises 17.4–17.6.

### Ch 18 · Serving — **A** (raised November 2026)
All four missing items done, numbers from `tfs/serving.py`, asserted in `tests/test_inference.py`. New §18.3 SLOs,
percentiles and goodput (worked ten-request example); §18.4 the scheduler's trade computed on the roofline
(batch vs TPOT vs throughput; the cache is 81% of traffic at batch 256; memory caps batch at 223 before the TPOT
limit of 314; chunked-prefill TTFT/TPOT table, balanced chunk 547); §18.5 $/M-token model (assumed
$2.50/GPU-hour, stated as an assumption; output tokens 4.8× input). §18.7 now covers fp8 / MXFP4 / NVFP4 weight
serving with bits-per-weight including scales and Hopper-vs-Blackwell execution, linked to Ch 0. New §18.9 MoE
expert parallelism (tokens per expert, batch 9,451 for compute-bound fp8 experts, DeepSeek-V3's EP32 / EP320
deployment, cross-linked to Ch 20 §20.7), §18.10 multi-LoRA (83.9 MB adapters; 64 distinct adapters = 5.37 GB
per step), §18.11 KV offload tiers (reload 16.8 ms vs recompute 301.5 ms), §18.12 structured output in the engine
(XGrammar, overlap, jump-forward). Three checkpoints, exercises 18.4–18.7.

### Ch 19 · Distributed Training — **A** (raised to the standard, November 2026)
ZeRO floors, TP traffic and the bubble table check. This pass (2.8k → 7.0k words, 3 → 10 quiz questions, 4 → 7
exercises), code in `tfs/parallel.py` / `tfs/memory.py`, asserted in `tests/test_training.py`:
1. §19.6 context parallelism: ring attention (bytes per rank, overlap condition c ≥ F/BW · n_kv/H, 4,945 / 9,890 tokens
   on IB), zigzag load balance (3/7/11/15 → 9/9/9/9 blocks), Llama 3's all-gather CP, DeepSpeed-Ulysses (bytes ∝ 1/cp,
   cp ≤ n_kv), the ring/Ulysses crossover at cp = 5 for GQA, comparison table, composition with DP/TP.
2. §19.7 Korthikanti's activation formula with GPT-3 numbers (275 → 55.6 → 34.4 → 10.3 → 4.83 GB), SP vs CP.
3. §19.8 FP8 at scale: what stays high-precision, which-peak MFU (Llama 4 Behemoth 390 TFLOP/s = 19.7% / 39.4%).
4. §19.9 fault tolerance: Llama 3's 466/419 interruptions, MTBF 3.09 h, Young–Daly intervals and waste table.
5. §19.10 the 5-D mesh with Llama 3's Table 4 checked row by row; HSDP vs FSDP traffic (41.3 GB vs 3.06 GB inter-node).
Also fixed: raw LaTeX (`W_1^{(a)}`, `\gtrsim`, `\in \{`) that `check_ui.js`'s pattern list missed.

### Ch 21 · Scaling Laws and Long Context — **A** (raised to the standard, November 2026)
Chinchilla arithmetic checks. This pass (2.4k → 5.8k words, 4 → 10 quiz questions, 3 → 6 exercises), code in
`tfs/scaling.py`, asserted in `tests/test_training.py`:
1. Corrected "both √C": the fitted law's own optimum is N ∝ C^0.45, D ∝ C^0.55 (24.5B on 2.14T, 87 tokens/param at
   GPT-3's budget, ratio drifting 50 → 191), stated next to the 20-token rule, with Besiroglu et al.'s refit
   (C^0.51, ~19 tokens/param). The worked example now labels which "optimum" each number is.
2. §21.2 data-constrained scaling (Muennighoff): effective data table, 4 / 16 epochs, the 16.4× ceiling.
3. §21.3 MoE and precision scaling laws, cross-linked to Ch 20 and Ch 12.
4. §21.4 test-time compute: coverage, unbiased pass@k, Brown et al. (15.9% → 56%), Snell et al. (4×, 14×).
5. §21.5 the staged long-context recipe: attention share of FLOPs (405B: 7.7% at 8k → 57.2% at 128k), Llama 3's six
   stages / ~800B tokens, DeepSeek-V3's 4K → 32K → 128K (62.9B tokens each, 4.3% of GPU-hours).

### Ch 22 · The 2026 Frontier — **A** (raised November 2026)
1.8k → 9.0k words, 2 → 13 quiz questions, 3 → 6 exercises. New: whole-chapter box; §22.2 gpt-oss alternating
128-token windows / full layers (KV arithmetic) and learned sinks (worked example); §22.3 linear attention as a
fast-weight memory (associativity, a 2-D overwrite example, measured interference √((n−1)/d_k)); §22.4 the delta rule
as one SGD step per token (tested against autograd), gating, and the update-rule table (linear / Mamba-2 / DeltaNet /
Gated DeltaNet / KDA); §22.5 the chunked UT-transform algorithm with code, tested equal to the recurrence, and its
FLOPs vs softmax attention; §22.6 the associative-recall argument (MQAR, Jelassi 2024), Qwen3-Next's config
(12 × [3 Gated DeltaNet, 1 gated attention]; memory 6.5 vs 25.8 GB at 256k), Kimi Linear (3:1 KDA:MLA, NoPE, the
ratio-ablation table); diffusion LMs as a watch item; §22.8 agents (tool protocol, injection, prefix caching,
long-horizon RL with masked tool tokens). Nemotron 3 row corrected to the Nano model card (23 Mamba-2, 23 MoE,
6 attention). Every computed number is in `tfs/linear_attention.py`, asserted in `tests/test_foundations.py`.
Not stated because it could not be verified: why Qwen chose 3:1 (the Qwen3-Next blog could not be fetched).

### Ch 23 · SFT, LoRA, RLHF, PPO — raised to **A**
Was **B**. Reward-model arithmetic still checks (0.3133 / 0.0181 / ln 2 / 1.3133). All four missing items addressed:
1. **LoRA / PEFT (new §23.3).** ΔW = BA with the gradient argument for B = 0 / A random and the both-zero saddle;
   rank, α and rsLoRA's α/√r (measured: α/r shrinks the update 4× from r = 4 to 64, α/√r does not); which matrices
   to adapt; memory for Llama-3-8B computed from the config (8,030,261,248 params; r = 16 adapters 41,943,040 =
   0.52%; full 128.5 GB / LoRA 16.7 GB / QLoRA 6.4 GB with NF4 at 4.127 bits) and for 70B (1,129 / 144 / 43 GB);
   what LoRA does not save (activations; ~4N vs 6N FLOPs); merging/unmerging; DoRA in one paragraph; multi-LoRA
   serving (83.9 MB adapter, 191× smaller than the base, +0.60% FLOPs) cross-linked to Ch 18.
   Code: `tfs/lora.py`; tests prove merged == unmerged, unmerge restores W, only A/B get gradients, step-0 identity.
2. **The modern recipe (new §23.4):** synthetic SFT data, rejection-sampling FT with the 1 − (1 − p)^k yield,
   distillation including reasoning traces (R1, s1), on-policy distillation with a computed forward-vs-reverse KL
   example (0.3681 / 0.7026 vs 0.5108 / 0.4307).
3. **Per-token KL as reward shaping and GAE derived (§23.6):** the telescoping n-step identity, the λ-weighted
   average, both limits, the backward recursion, and a four-token worked example at λ ∈ {0, 0.95, 1}
   (A = [0.4171, 0.3127, 0.2450, 0.1000] at 0.95). Code: `tfs/rl.py`.
4. **Evaluation of post-trained models (§23.8)**, pointing to the new Ch 26.
Now 12 quiz questions in 3 checkpoints, 7 exercises (sizing, implementation, predict-then-measure, GAE).
Companion notebook still planned (notebooks session).

### Ch 24 · DPO — **A** (kept)
Missing items addressed: **online / iterative DPO in code (new §24.8)** with a tabular policy whose DPO gradient has a
closed form (a response in no pair gets exactly zero gradient; checked against autograd) and a measured comparison at
equal budget: reference 0.204, offline 0.395, online 0.483, online better on 40/40 seeds (`tfs/dpo.py`
`offline_dpo`/`online_dpo`); **DPO on reasoning traces (new §24.9)**: Iterative RPO, the NLL term, length effects.
Also fixed: malformed `<big(` tags, undefined `var(--q/--k/--o)` borders, a dead `href="#"`. Second checkpoint (3
questions) and exercise 24.4 added.

### Ch 25 · DPO Variants, GRPO, RLVR — raised to **A**
Was **A−**. GRPO advantage arithmetic still checks (+2.645 / −0.378). Added: **Dr. GRPO (§25.4)**, both biases with
computed per-token weights (GRPO 25.0 / 8.33 / −12.5 / −2.5 ×10⁻⁴; a 1,000-token wrong answer penalised 5× less per
token; std bias 1.51×), DAPO and Dr. GRPO reductions side by side; **GSPO (§25.5)**, sequence-level ratio as a
geometric mean (1.0513 vs token ratios 1.105 / 0.951 / 1.492 / 0.779), Routing Replay and why MoE needed it
(cross-linked to Ch 20); **process vs outcome rewards (§25.6)** with a Monte Carlo step-label example; **agentic
multi-turn RL (§25.7)**: environment rewards, the action-token mask (100 of 500 tokens), trajectory-level advantages,
asynchronous rollouts, environment hacking, SWE-RL and RAGEN. A checkpoint for the DPO family (it had none), a
third checkpoint, and exercises 25.4–25.6. Removed an unverifiable blog link. Companion notebook still planned.

### Ch 26 · Evaluating LLMs — new, **A**
Written to the Ch 20 standard (~7.7k words, 11 quiz questions in 3 checkpoints, 6 exercises). What a score is (the
five-part pipeline, log-likelihood vs generative scoring); benchmarks with item counts and saturation; contamination
(n-gram overlap, Min-K% Prob worked example, GSM1k, time splits, canaries); scoring (normalised EM, F1, extraction
bugs); error bars from first principles with the resolution of five real benchmarks (GPQA Diamond ±6.4 pts, AIME
±17.9); paired vs unpaired bootstrap on the same data ([+0.8, +5.4] vs [−2.4, +8.6]); items for a 2-point gain
(8,077 unpaired, 1,955 paired); pass@k's unbiased estimator derived and proved unbiased by exact expectation (naive
plug-in 0.760 vs true 0.832); LLM-as-judge biases with the swap test and a fitted length-controlled win rate (62.0%
raw → 47.4% LC on a length-only judge); Arena as Bradley-Terry; reasoning and agent evals with pass^k (0.6⁸ = 0.0168);
building a private eval with Cohen's κ (80% agreement = κ 0.524). Code `tfs/evals.py`, tests
`tests/test_posttraining.py`, notebook `notebooks/26_evaluation.ipynb`.

### Ch 27 · Capstone — raised to **A** (was Ch 26)
Renumbered from Ch 26. The stretch-goal table's chapter links are still checked against their targets. Fixed the
remaining "15M parameters" to 12.6M. Added **M6 — LoRA** (325,632 trainable parameters at r = 8 = 2.59%, asserted;
step-0 identity, B-only gradient, merge tests; DPO with two adapters on one base) and **M7 — Evaluation** (held-out
perplexity and bits per byte with document-level bootstrap; a programmatic task benchmark with a contamination test,
pass@k, paired bootstrap and a power calculation). Final check grown to 8 questions.

## Priorities, across the whole course

1. **LoRA / PEFT** — absent, and the most-used fine-tuning tool (Ch 23 + capstone).
2. **Data for pretraining** — dedup, filtering, mixtures, annealing (Ch 11).
3. **Context parallelism** — ring attention / Ulysses (Ch 19), and the staged long-context recipe (Ch 21).
4. **Linear-attention hybrids** — delta-rule attention explained, not just listed (Ch 22).
5. ~~**Serving economics** — SLOs, goodput, $/token, MoE expert-parallel serving (Ch 18).~~ Done, November 2026.
6. **The 13 planned notebooks**, so every chapter's numbers are executable like Ch 0, 12 and 20.
7. **Evaluation** — nowhere taught as a subject (benchmarks, contamination, LLM-as-judge, length control).
