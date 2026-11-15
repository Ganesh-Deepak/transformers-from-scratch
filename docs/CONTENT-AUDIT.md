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
Next: an interactive router widget (tokens × experts, top-k, load bars, bias balancing).

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

### Ch 13 · Decoding — **B+** (checked; sampler arithmetic correct)
Missing: best-of-n / self-consistency, stop sequences, batch-dependent nondeterminism, grammar engines (XGrammar) and token-boundary issues in constrained decoding.

### Ch 14 · The GPU — **A−**, one section wrong (NOT YET FIXED)
Ridge points, decode ceilings and prefill/decode numbers check. **§14.4 is wrong**: it claims d_head is 64/128 because of 128-byte cache lines and that "you never see 48, 80, 96". Counterexamples: Phi-2 (80), Phi-3-mini and GPT-NeoX-20B (96), Gemma (256), DeepSeek-V3 (192 q/k). The real constraints are tensor-core multiples and which head sizes kernels ship tuned versions of. Rewrite §14.4 and its quiz. Fixed this pass: raw LaTeX (`d_{head}`, `\sqrt`) shown literally to readers in Ch 7, 14, 16, 19.

### Not yet audited in this pass
Ch 15–19, 21–26 (Ch 12 and Ch 0 were rewritten in October). Add a check_ui rule rejecting raw LaTeX outside code.

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
