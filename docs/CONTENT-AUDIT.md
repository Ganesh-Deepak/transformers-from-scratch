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

### Ch 01 · Tensors, Shapes — **A−**
Excellent pacing; strides, view vs reshape, and the memory arithmetic are exactly right for
the audience. Fact-checked: Llama-3-8B untied embeddings, memory sums.
Missing, by value:
1. `expand` vs `repeat` (stride-0 views): needed for GQA's `repeat_kv` in Ch 16 and a classic
   silent memory blow-up.
2. In-place ops and autograd ("a leaf Variable that requires grad is being used in an in-place
   operation"), and why views make in-place dangerous.
3. dtype promotion rules (`int + float`, `bf16 + fp32` → fp32) with the silent-upcast cost.
4. A "read this real shape trace" exercise from an actual open model's forward pass.

### Ch 02 · Matrix Multiplication — **A−**
Fixed: the H100 estimate divided by peak FLOP/s while calling it "realistic utilisation"
(said 10 days on 100 H100s; at ~45% MFU it is ~3 weeks). Clarified that `12·n_layer·d·T²`
is a training-step figure.
Missing:
1. Tensor-core shape alignment (dimensions as multiples of 8/64/128 and why vocab sizes are
   padded) — the first place a reader meets "shapes affect speed".
2. Arithmetic intensity of a matmul in one paragraph, forward-linked to Ch 14.
3. The FLOP count of a full transformer layer derived from its matmuls (2·12·d² per token
   for attention + MLP projections), connecting 2.5 to Ch 9's parameter count.

### Ch 20 · Mixture of Experts — rewritten, **A**
2.1k → ~7.5k words. New: three router types and their equivalence, the top-1 zero-gradient
trap, sorted dispatch / grouped GEMM, capacity and measured drop rates, aux loss derived,
z-loss, global-batch balancing, DeepSeek-V3 bias rule (code corrected), expert choice and why
it breaks causality, fine-grained and shared experts with combinatorics, expert parallelism and
the all-to-all bill, node-limited routing, the serving fact that batched decode reads most
experts, training stability, upcycling, a 7-model spec table with arithmetic cross-checks,
8 quiz questions, 7 exercises. Every number from `tfs/moe.py` / `notebooks/20_moe.ipynb`.
Next: an interactive router widget (tokens × experts, top-k, load bars, bias balancing).

### Ch 03 · Tokenization — **B+**
Clear BPE build-up; the hand-traced merges are verified (24 → 19 → 14 → 10 → 9 tokens).
Fixed: Gemma 3's vocabulary is 262,144, not 256,000; the GPT-2 digit splits are now labelled
illustrative (they were stated without a reproducible source).
Missing, by value:
1. **Streaming detokenization**: a token can end mid-way through a UTF-8 character, so a server
   must buffer bytes before emitting text. Every inference engineer hits this.
2. **Special-token injection**: user text containing `<|eot_id|>` must not be encoded as the
   special id (`allowed_special`/`disallowed_special`); this is a security boundary.
3. Modern pre-tokenization (GPT-4's `cl100k` pattern, digit groups `\p{N}{1,3}`), Unigram LM
   vs BPE (SentencePiece), normalisation (NFKC) and why it is lossy.
4. Vocabulary size as a scaling decision (Tao et al. 2024), with the parameter/sequence-length
   trade computed rather than asserted; byte-level and tokenizer-free models (BLT, 2024).
5. Token healing / prompt-boundary effects as a mechanism, not only the trailing-space anecdote.
6. A measured multilingual fertility table produced by code in the repo.

### Ch 04 · Embeddings and the Residual Stream — **B+**
The residual-stream section is the best explanation in Part I. Fixed: "a token seen once gets one
gradient update, ever" was wrong under AdamW (momentum keeps moving the row; weight decay shrinks
every row every step), which also changes the explanation of glitch tokens.
Missing:
1. Embedding scale and initialisation (Gemma multiplies by √d_model; std choices; why the
   residual stream's norm grows with depth and what pre-norm does about it).
2. A real logit-lens run with output shown (the chapter promises one in a notebook that does
   not exist yet), plus the tuned-lens caveat.
3. Output-side details that matter in practice: final-logit soft-capping (Gemma 2), z-loss on
   the output softmax, vocab padding to a multiple of 64/128 for tensor cores.
4. Superposition deserves one worked toy example (Elhage et al. 2022), not one sentence.

### Ch 05 · Attention From Scratch — **A−**
The best chapter in Part II: the √dₖ derivation, a hand-worked causal pass whose every number
checks, and four "bugs that don't crash". Fixed a self-contradiction: the complexity table used
d = 64 for both attention and projections, which made attention 16× *larger* at 1k tokens while
the text said projections dominate there. Rebuilt per layer for a 7B-class model (crossover at
T = 6·d_model ≈ 25k). Exercise 5.4 compared training attention FLOPs (12·L·d·T²) with
forward-only parameter FLOPs (2N·T) and got ~9k; like with like it is ~27k.
Missing:
1. Padding and batching: left- vs right-padding for batched generation, and the all-masked-row
   NaN with a worked fix (the chapter mentions it in one clause).
2. Modern attention details: QK-norm, logit soft-capping, attention sinks as a mechanism
   (softmax must sum to 1), with the "off-by-one" softmax and sink tokens.
3. Numerically stable softmax (subtract the row max) shown here, since Ch 17 depends on it.
4. Multi-query/causal efficiency of training: one forward pass yields T predictions — derive
   the cross-entropy over all positions here, not only state it.

### Ch 06 · Multi-Head Attention — **A**
The reshape-dance treatment (memory positions, the transpose-before-merge bug, the loop-vs-vectorised
proof test) is excellent. Fixed a quiz explanation that wrote T×64 for T²×64. Small additions:
per-head QK/OV circuit view (one paragraph), and the fused-QKV weight layout differences between
codebases (a frequent checkpoint-conversion bug).

### Ch 07 · Position — **A−**
Clean derivation; the worked RoPE dot product (0.47775 at every offset 2) checks. Fixed exercise
7.3: the longest wavelengths are 5.4×10⁴ / 2.6×10⁶ tokens (said 5.9×10⁴ / 3.0×10⁶) and 18 / 32
pairs exceed 4096 tokens (said 21 / 30); added the closed form. Added Llama 4's interleaved NoPE
global layers. Missing: partial RoPE (rotating only part of the head, used by DeepSeek's decoupled
RoPE and others), multimodal RoPE (2D/3D, Qwen2-VL M-RoPE), and a measured length-extrapolation
plot from the repo's own model.

### Ch 08 · The Transformer Block — **A−**
Checked: 11008 / 22016 / 28672 d_ff arithmetic, GPT-2 12·L·d² sanity check, Llama-2-7B MLP share
(65%). Strong on norm placement and SwiGLU. Missing:
1. QK-norm and sandwich norm get one paragraph each; they deserve the same derivation treatment
   as pre- vs post-LN (logit growth → softmax saturation → loss spikes), with a measured plot.
2. Parallel attention+MLP blocks (GPT-J, PaLM) and why they fell out of favour.
3. A per-block FLOP and memory budget (activations saved for backward), linking to Ch 19's
   activation checkpointing.

### Ch 09 · Whole Architectures + Shape Trace — **A**
The 20-step shape trace is the course's centrepiece. Fixed: `count_params` was claimed to return
GPT-2's 124,439,808 but returns 123,551,232; the difference is GPT-2's learned position table
(786,432), linear biases (82,944) and LayerNorm biases (19,200), now spelled out to the parameter.
The trace model is ~52M parameters (said ~53M); the 128k-token KV cache of Llama-3-8B equals
its bf16 weights (said "more than"). Missing: a decoder-only vs encoder-decoder FLOP comparison,
and modern embedding/reranker models (they are still BERT-shaped; one paragraph on how they are
trained, contrastively, would help engineers who build retrieval).

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

### Ch 13 · Decoding — **B+** (checked; sampler arithmetic correct)
Missing: best-of-n / self-consistency, stop sequences, batch-dependent nondeterminism, grammar engines (XGrammar) and token-boundary issues in constrained decoding.

### Ch 14 · The GPU — **A−**, one section wrong (NOT YET FIXED)
Ridge points, decode ceilings and prefill/decode numbers check. **§14.4 is wrong**: it claims d_head is 64/128 because of 128-byte cache lines and that "you never see 48, 80, 96". Counterexamples: Phi-2 (80), Phi-3-mini and GPT-NeoX-20B (96), Gemma (256), DeepSeek-V3 (192 q/k). The real constraints are tensor-core multiples and which head sizes kernels ship tuned versions of. Rewrite §14.4 and its quiz. Fixed this pass: raw LaTeX (`d_{head}`, `\sqrt`) shown literally to readers in Ch 7, 14, 16, 19.

### Not yet audited in this pass
Ch 15–19, 21–26 (Ch 0 was rewritten in October; Ch 12 is audited above). Add a check_ui rule rejecting raw LaTeX outside code.

### Ch 14 · §14.4 — now FIXED
Rewritten: tensor cores require multiples of 16 (32 for fp8); 64/128 dominate because attention kernels ship
tuned paths for them, not because of cache lines. Real counterexamples cited (Phi-2 80, Phi-3-mini 96,
DeepSeek-V3 192, Gemma 256). Quiz replaced with a Phi-2-shaped question. `check_ui.js` now fails on raw
LaTeX in page text (verified to fail on the old Ch 14).

### Ch 15 · The KV Cache — **A−**
All arithmetic checks (128 KB/token Llama-3-8B, 320 KB/token 70B, user counts, fleet sizing). Missing:
cache eviction and compression beyond quantization (StreamingLLM sinks, H2O / SnapKV), chunked prefill,
and a measured cached-vs-uncached timing plot from the repo's model. Units mix GB and GiB in places.

### Ch 16 · MQA, GQA, MLA — **A**
MLA absorption and the decoupled-RoPE argument are the best treatment in the course; all numbers check.
Fixed: Gemma 2 27B uses 16 KV heads (chapter said all Gemma landed on ≤8); removed an unverified
"Kimi K2.5" row. Missing: cross-layer KV sharing (CLA, YOCO), and why absorbed MLA decode behaves like
MQA over a 576-wide latent (compute-dense, which is why it suits decode).

### Ch 17 · FlashAttention — **A−**
Online softmax worked example and FA1–3 numbers check. Fixed: "d²/M ≈ 1/25 → 25× fewer HBM accesses"
depended on counting M in bytes vs elements; now "≈1/12, an order of magnitude". Note: the §17.4 pseudo-code
skips fully-masked blocks but omits the in-tile causal mask on the diagonal block (exercise 17.2's code has it);
say so in the pseudo-code. The FA4 (2026) row is stated from announcements and should be re-checked against
the paper. Missing: FlashDecoding / split-K for decode explained with the same rigour; FlexAttention examples.

### Ch 18 · Serving — **B+**
Speculative-decoding arithmetic (2.77, 6.13), PagedAttention waste figures check. Missing, by value:
1. SLOs and goodput: TTFT, TPOT, p99, and how a scheduler trades them; a $/million-token cost model.
2. Serving MoE with expert parallelism (now in Ch 20 §20.7 — cross-link both ways).
3. Multi-LoRA serving, KV offload tiers (CPU/SSD, LMCache), structured-output engines (XGrammar).
4. FP8 / FP4 (NVFP4, MXFP4) weight serving, linking Ch 0's formats to deployment.

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

### Ch 22 · The 2026 Frontier — **B**
Most needs refreshing, by design. Fixed: added the 2025 major-lab linear-attention hybrids (Qwen3-Next:
3 Gated DeltaNet : 1 gated attention; Kimi Linear: 3 KDA : 1 MLA; MiniMax-M1: lightning attention 7 : 1) and a
sentence on the delta-rule direction; exercise 22.3 pointed to §22.3 for a box that is in §22.4. To check:
the Nemotron 3 row. Missing: delta-rule / gated linear attention explained (it is now more relevant than
Mamba), gpt-oss-style alternating sliding/full attention with learned sinks, diffusion language models
(as a watch item), and an "agentic" section (tool use, long-horizon RL) since that is where 2026 effort went.

### Ch 23 · SFT, RLHF, PPO — **B**
Reward-model arithmetic checks (0.3133 / 0.0181 / ln 2 / 1.3133). Missing, by value:
1. **LoRA / QLoRA / parameter-efficient fine-tuning — not taught anywhere in the course.** For a working AI
   engineer this is the most common fine-tuning tool. Needs its own section: the low-rank update, rank and
   alpha, which matrices to adapt, memory arithmetic, merging, multi-LoRA serving (Ch 18).
2. The modern post-training recipe: synthetic SFT data, rejection-sampling fine-tuning, distillation from a
   stronger model (including reasoning traces), on-policy distillation.
3. GAE derived (the chapter names it but does not explain it), and the per-token KL penalty as reward shaping.
4. Evaluation of post-trained models (win rates with length control, LLM-as-judge pitfalls).

### Ch 24 · DPO — **A**
The best derivation in the course; every step follows. Fixed a typo (§24.2–20.4). Missing: online / iterative
DPO in code, and a short note on DPO with reasoning traces.

### Ch 25 · DPO Variants, GRPO, RLVR — **A−**
GRPO advantage arithmetic checks (+2.645 / −0.378). Missing: Dr. GRPO (the length and std normalisation
biases in GRPO), GSPO (sequence-level ratios, introduced for stable RL on MoE models — cross-link Ch 20),
process vs outcome rewards, and the agentic/tool-use RL setting (multi-turn rollouts, environment rewards).

### Ch 26 · Capstone — **A−**
Fixed: the stretch-goal table's chapter numbers all predated Ch 0 (MLA → 16, MoE → 20, FlashAttention → 17,
speculative decoding → 18, YaRN → 21, GRPO → 25); they are now links, so a label can be checked against its
target. The model is 12,587,904 parameters (said ≈15 M). Missing: a LoRA milestone (see Ch 23), and an
evaluation milestone (held-out perplexity plus a tiny task benchmark).

## Priorities, across the whole course

1. **LoRA / PEFT** — absent, and the most-used fine-tuning tool (Ch 23 + capstone).
2. **Data for pretraining** — dedup, filtering, mixtures, annealing (Ch 11).
3. **Context parallelism** — ring attention / Ulysses (Ch 19), and the staged long-context recipe (Ch 21).
4. **Linear-attention hybrids** — delta-rule attention explained, not just listed (Ch 22).
5. **Serving economics** — SLOs, goodput, $/token, MoE expert-parallel serving (Ch 18).
6. **The 13 planned notebooks**, so every chapter's numbers are executable like Ch 0, 12 and 20.
7. **Evaluation** — nowhere taught as a subject (benchmarks, contamination, LLM-as-judge, length control).
