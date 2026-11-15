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
Interactive router widget added in §20.4 (`data-viz="moe"`): tokens × experts with top-k picks, live load
bars, a DeepSeek-V3 bias-balancing toggle that takes the load CV from ~2.1 to under half over 40 steps, and the
experts-read-per-decode-step curve vs batch; interaction-tested in `check_interaction.js`.

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

### Ch 10 · The Backward Pass — **A**
Every derivation checks numerically (softmax Jacobian at [1, 2, 0.5]; p − y at [2, 1, 0.1]; the
RMSNorm correction term and its orthogonality to x; attention backward). Missing, by value:
1. **Memory of the backward pass**: what is saved for backward per layer, activation memory in
   bytes for a real config, and activation checkpointing derived here (Ch 11 only tabulates it).
2. The vocabulary-sized logits: (B·T·V) in fp32 is often the largest tensor in training;
   chunked / fused cross-entropy (Liger-style) as the fix, with numbers.
3. Embedding backward as scatter-add, SwiGLU backward, and RoPE backward (the inverse rotation).
4. Mixed-precision gradients: where bf16 gradients lose information and why reductions run in fp32.

### Ch 11 · Training a Transformer — **A−**
Real measured runs, the bits-per-byte lesson, and the double-shift case study are excellent.
Fixed: GPT-3 175B's learning rate was 0.6e-4, not 1.5e-4. Missing, by value (this chapter is
where a pretraining engineer's job actually lives):
1. **Data**: deduplication (exact and MinHash), quality filtering (FineWeb-Edu-style classifiers),
   mixture weights, and annealing on high-quality data at the end of a WSD run. Today the chapter's
   only data section is packing.
2. Batch size: critical batch size, why it grows during training, and batch-size warmup.
3. Hyperparameter transfer (μP) and LR scaling with width, so the reader can pick an LR for a
   model they cannot afford to sweep.
4. Throughput: tokens/s and MFU measured on the repo's own model, and how to find the bottleneck.
5. Loss spikes at scale: z-loss on the output, QK-norm, skipping batches, and restart policy.
6. Checkpointing and exact resumption (data-loader state, RNG), which every long run needs.

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
Ch 15–19, 21–26 (Ch 12 and Ch 0 were rewritten in October). Add a check_ui rule rejecting raw LaTeX outside code.

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

### Ch 19 · Distributed Training — **A−**
ZeRO floors, TP traffic (0.54 / 0.81 / 0.94 GB per layer, 75 GB over 80 layers) and the bubble table check.
Missing, by value: **context / sequence parallelism** (ring attention, DeepSpeed-Ulysses) — how every
long-context model is trained, absent today; HSDP; FP8 training at scale; fault tolerance and checkpoint
cadence at thousands of GPUs; the activation-memory formula (Korthikanti et al.) with real numbers.

### Ch 21 · Scaling Laws and Long Context — **B+**
Chinchilla arithmetic checks (51.2B / 1,023B for GPT-3's budget; token ratios). Fixed: the NTK snippet reused
`scale` with the inverse meaning of the PI snippet (now `s = target/trained`, with the 10000 → ~82,700 value);
Gemma 2 alternates local/global 1:1, Gemma 3 is 5:1. Note: the fitted law's own optimum is N ∝ C^0.45,
D ∝ C^0.55 (α ≠ β); the chapter's "both √C" is the 20-tokens-per-parameter rule, and should say so. Missing:
data-constrained scaling (repeating data, Muennighoff et al. 2023), test-time compute scaling, MoE and
precision scaling laws (link Ch 20, Ch 12), and the staged long-context training recipe (e.g. Llama 3's
8k → 128k extension).

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
5. ~~**Serving economics** — SLOs, goodput, $/token, MoE expert-parallel serving (Ch 18).~~ Done, November 2026.
6. **The 13 planned notebooks**, so every chapter's numbers are executable like Ch 0, 12 and 20.
7. **Evaluation** — nowhere taught as a subject (benchmarks, contamination, LLM-as-judge, length control).
