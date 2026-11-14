# After Transformers: the rest of the AI engineering curriculum

Ten courses, one library. Each course is authored with `NEW-COURSE-PROMPT.md`
and plugs into the shared platform described in `PLATFORM.md`: one look, one
theme setting, one record of progress across every course, and links that go
from "you need this" straight to the chapter that teaches it.

Each entry below is a **paste-ready brief**: drop the title, slug and brief into
the `<<< >>>` placeholders at the top of the prompt.

## Status

| # | Course | Slug | Repo (Pages path) | Ink | Status |
|---|---|---|---|---|---|
| 0 | Transformers From Scratch | `transformers` | `/transformers-from-scratch/` | `--part-2` | **live**, 28 chapters, being raised to grade A throughout |
| 1 | Evaluation & Experimentation for LLM Systems | `evals` | `/llm-evaluation/` | `--part-4` | planned, build first |
| 2 | Post-Training in Practice | `posttraining` | `/post-training/` | `--part-1` | planned |
| 3 | GPU Programming: CUDA & Triton | `gpu` | `/gpu-programming/` | `--part-3` | planned |
| 4 | Inference Engineering at Scale | `inference` | `/inference-engineering/` | `--part-5` | planned |
| 5 | Retrieval & Context Engineering | `retrieval` | `/retrieval/` | `--part-6` | planned |
| 6 | Agents & Tool Use | `agents` | `/agents/` | `--part-0` | planned |
| 7 | Production AI Systems | `production` | `/production-ai/` | `--part-7` | planned |
| 8 | AI Security & Red Teaming | `security` | `/ai-security/` | `--part-1` | planned |
| 9 | Mathematics for ML, from First Principles | `maths` | `/ml-maths/` | `--part-6` | planned |
| 10 | Multimodal & Generative Models Beyond Text | `multimodal` | `/multimodal/` | `--part-3` | planned |

Inks repeat after eight; no two courses in the same track share one.

## What Transformers From Scratch now covers, so the others do not repeat it

The new courses **build on** these chapters with cross-course links (`xref`, see
`PLATFORM.md` §5) instead of re-teaching them:

| Topic | Where it is taught | Courses that link to it |
|---|---|---|
| Number formats, fp8/fp4, quantization arithmetic | Ch 0, Ch 12 | 3, 4 |
| Tensors, shapes, matmul, FLOP counting | Ch 1, Ch 2 | 3, 9 |
| Attention, MHA, RoPE, the block, full architectures | Ch 5–9 | 4, 5, 10 |
| Backprop, training loop, data, μP, MFU | Ch 10, Ch 11 | 2, 3, 9 |
| GPU roofline, tensor cores, kernels, FlashAttention | Ch 14, Ch 17 | 3, 4 |
| KV cache, GQA/MLA, serving, SLOs and $/token | Ch 15, Ch 16, Ch 18 | 4, 7 |
| Distributed training, context parallelism | Ch 19 | 2, 3 |
| Mixture of Experts (training, EP, serving) | Ch 20 | 3, 4 |
| Scaling laws, long context | Ch 21 | 1, 4, 5 |
| SFT, LoRA/QLoRA, RLHF, DPO, GRPO | Ch 23–25 | 2, 6 |
| Evaluating LLMs (benchmarks, judges, pass@k, significance) | Ch 26 | 1, 2, 6, 7 |

Courses 1, 2 and 4 are therefore the *practice* courses for material whose
theory already exists: they start where those chapters stop.

## How these were chosen

Pay tracks **scarcity × leverage**, not enthusiasm. Sorting the field that way:

| | abundance | pay ceiling |
|---|---|---|
| Calling an LLM API, prompt tweaking | very high | low |
| Standard RAG chatbot | high and rising | medium |
| Agents that survive contact with production | medium | high |
| Knowing whether any of it actually works (evals) | **low** | high, and universal |
| Post-training a model that beats the base on your task | **low** | high |
| Making models fast on real hardware (kernels, serving) | **lowest** | highest |

Transformers From Scratch already put you in the scarce column on *model
internals*, which is the part most people skip. The gap is **shipping and
measuring**, plus the **systems layer** underneath.

Honest advice on sequencing: **four of these done deeply beats ten done
shallowly.** Course 1 is the multiplier — do it first regardless of track,
because every other skill is unverifiable without it. Then commit to one track.

    Everyone        1 (Evals)  →  7 (Production)
    Systems track   3 (GPU)    →  4 (Inference)
    Applied track   5 (Retrieval) → 6 (Agents)
    Model track     2 (Post-training)
    Frontier labs   9 (Maths) — interview gate, and it unblocks reading papers
    Breadth         8 (Security), 10 (Multimodal)

Caveat: this reflects how the skills are distributed, not a market forecast.
Verify against actual job postings you want before committing months to a track.

---

## 1. Evaluation & Experimentation for LLM Systems
*~18 chapters. Do this first. Builds on Transformers Ch 26 (the theory of evaluating models) and goes on to evaluating whole systems.*

> A course on how to know whether an AI system actually works. For engineers who
> can build LLM features but cannot yet prove one version is better than another.
> By the end you can design a golden set, build an LLM judge you have validated
> against humans, compute whether a 3% win is real or noise, and run a regression
> suite in CI. Covers: defining "good" operationally before writing any code;
> task metrics vs proxy metrics; building and maintaining a golden set;
> inter-annotator agreement; LLM-as-judge and its measured failure modes
> (position bias, verbosity bias, self-preference); pairwise vs pointwise
> scoring; statistical power and how tiny most eval sets really are; confidence
> intervals and why per-example variance matters; regression suites and CI gates;
> online evaluation, A/B and sequential testing; guardrail metrics; cost and
> latency as first-class outcomes; contamination and leakage; eval-driven
> development as a workflow. Deliberately not covered: model training, and any
> benchmark you cannot reproduce.

## 2. Post-Training in Practice
*~20 chapters. The applied counterpart to Transformers Ch 23–25 (incl. LoRA) and Ch 26 (evaluation).*

> How to take a base model and make it good at your task. For engineers who
> understand SFT, DPO and GRPO in theory and need to actually run them. By the
> end you can decide whether to fine-tune at all, build and decontaminate a
> dataset, train a LoRA that beats prompting, and evaluate it honestly. Covers:
> the fine-tune vs prompt vs retrieve decision, with the cost model; dataset
> construction, curation and dedup; decontamination against your eval set; SFT
> mechanics and the hyperparameters that actually matter; LoRA and QLoRA derived
> from the low-rank hypothesis; PEFT variants and when each wins; catastrophic
> forgetting, measured; preference data collection and its biases; reward
> modelling; DPO, IPO, KTO compared on the same data; GRPO and verifiable-reward
> setups; synthetic data generation and mode collapse; distillation; the honest
> evaluation of a fine-tune. Deliberately not covered: pre-training from scratch.

## 3. GPU Programming: CUDA & Triton from First Principles
*~22 chapters. Scarcest skill, highest ceiling.*

> Why GPU code is fast or slow, and how to write the fast kind. For engineers who
> can train models but have never written a kernel. By the end you can write a
> tiled matmul that approaches cuBLAS, implement FlashAttention in Triton, and
> read a profile to find the real bottleneck. Covers: the memory hierarchy as the
> only thing that matters; arithmetic intensity and the roofline model; the
> execution model — warps, blocks, occupancy; coalescing and bank conflicts,
> measured; your first kernel; tiled matrix multiplication built up in stages;
> reductions, scans and their tree structure; the Triton programming model and
> what it buys you; fusing elementwise chains; a numerically stable softmax
> kernel; FlashAttention derived and implemented; autotuning; profiling with
> Nsight; and the discipline of knowing when the compiler already wins.
> Every claim is measured on real hardware, never asserted.

## 4. Inference Engineering at Scale
*~20 chapters. Extends Transformers Ch 14–18 and Ch 20 §20.7 (serving MoE).*

> How to serve a language model to real traffic without going bankrupt. For
> engineers who understand KV caching and want to build or operate a serving
> stack. By the end you can reason about a latency budget, choose a quantization
> scheme knowing what it costs in quality, and explain what vLLM is doing and
> why. Covers: latency budgets — TTFT and inter-token latency as separate
> problems; prefill compute-bound vs decode memory-bound; the arithmetic of the
> decode bottleneck; static vs continuous batching; PagedAttention and KV memory
> fragmentation; quantization (INT8, INT4, FP8, GPTQ, AWQ) and exactly what
> degrades; speculative decoding and its variants; prefix and prompt caching;
> tensor and pipeline parallelism for serving; multi-LoRA serving; scheduling,
> fairness and starvation; autoscaling on bursty traffic; disaggregated
> prefill/decode; cost per million tokens as an engineering target.

## 5. Retrieval & Context Engineering
*~18 chapters.*

> Getting the right information into the context window. For engineers who have
> built a RAG demo that worked and a RAG product that did not. By the end you can
> build an ANN index from first principles, evaluate retrieval separately from
> generation, and diagnose why a pipeline returns plausible-but-wrong answers.
> Covers: what an embedding actually encodes, and what it does not; training and
> fine-tuning a retriever; ANN indexes — HNSW and IVF-PQ built up from brute
> force; the recall/latency/memory trilemma; chunking as a design decision with
> measurable consequences; sparse retrieval and BM25, still undefeated on some
> tasks; hybrid fusion; cross-encoder rerankers; query rewriting, expansion and
> HyDE; **retrieval evaluation as its own discipline** — the step almost everyone
> skips; long context vs retrieval, with the cost model; lost-in-the-middle and
> context rot; structured and hybrid retrieval over SQL and graphs; freshness,
> invalidation and incremental indexing.

## 6. Agents & Tool Use
*~18 chapters.*

> Building systems that take actions, and keeping them from taking bad ones. For
> engineers who can call a model and want to build something that plans, uses
> tools and recovers. By the end you can design a tool interface a model actually
> uses correctly, bound cost and blast radius, and evaluate a trajectory rather
> than just a final answer. Covers: the agent loop from first principles; tool
> schema design and the specific ways models misuse tools; function calling
> internals; planning strategies — ReAct, plan-and-execute, tree search — and
> when each is worth its tokens; memory architectures; context management and
> compaction; the Model Context Protocol; sandboxing and least privilege;
> multi-agent patterns and the cases where they measurably hurt; a taxonomy of
> agent failures with worked diagnoses; step, cost and time budgets;
> human-in-the-loop checkpoints; evaluating agents on trajectory *and* outcome.

## 7. Production AI Systems
*~16 chapters.*

> Operating nondeterministic systems in production. For engineers whose feature
> works on their laptop and is about to meet real users. By the end you can trace
> a request through an LLM pipeline, cache it safely, degrade gracefully and run
> an incident when the model is the thing that broke. Covers: tracing and
> observability when the same input yields different outputs; structured logging
> of prompts and responses without leaking PII; caching strategies — exact,
> prefix, semantic — and when semantic caching is actively dangerous; fallbacks
> and graceful degradation; rate limits, backpressure and queueing; timeouts,
> retries and idempotency; cost engineering and per-feature unit economics; model
> version pinning and migration when a provider deprecates; canary releases and
> rollback for prompts; incident response for AI systems; data retention and
> privacy; defining SLOs for a probabilistic system.

## 8. AI Security & Red Teaming
*~14 chapters.*

> Attacking and defending LLM systems. For engineers shipping AI features that
> touch untrusted content or real permissions. By the end you can threat-model an
> agent, run a structured red team and explain to a security reviewer why your
> design is safe. Covers: prompt injection, direct and indirect; the lethal
> trifecta — private data, untrusted content and an exfiltration channel — and
> why removing any one leg fixes it; tool-use and confused-deputy attacks;
> a jailbreak taxonomy and what actually generalises; exfiltration channels
> including markdown images and DNS; guardrail design and how guardrails are
> evaded; PII detection and handling; supply chain risk in models, datasets and
> packages; red team methodology and reporting; threat modelling for agentic
> systems. Defensive framing throughout: every attack is taught to build the
> mitigation.

## 9. Mathematics for Machine Learning, from First Principles
*~20 chapters. The frontier-lab interview gate.*

> The maths you actually need, derived rather than recited. For engineers who can
> implement a transformer but stall when a paper opens with an expectation over a
> distribution. By the end you can read a methods section without skipping the
> equations. Covers: linear algebra as geometry — span, projection, change of
> basis; eigendecomposition and what it means; **SVD, and why it silently
> explains PCA, low-rank adaptation, conditioning and half the field**;
> probability and the handful of axioms you use daily; expectation, variance and
> concentration; maximum likelihood and MAP as the same idea twice; the
> exponential family; entropy, KL divergence and cross-entropy derived from
> information theory, so the training loss stops being arbitrary; convexity and
> why non-convexity is survivable; gradient descent, momentum and Adam derived
> rather than quoted; second-order methods and why nobody uses them; matrix
> calculus sufficient for backprop by hand; floating point and numerical
> stability.

## 10. Multimodal & Generative Models Beyond Text
*~18 chapters.*

> Everything that is not a language model, built on the same foundations. For
> engineers with solid transformer knowledge who need vision, audio or image
> generation. By the end you can explain why diffusion and autoregression are
> different answers to the same question, and implement a small one of each.
> Covers: the Vision Transformer and why patches work; contrastive learning and
> CLIP; vision-language architectures and connector design; the tokenizer problem
> for continuous modalities; diffusion from first principles — the forward
> process, the reverse process, and what the network actually predicts; DDPM to
> DDIM; flow matching as the simpler modern framing; classifier-free guidance;
> latent diffusion; audio as tokens vs as continuous features; speech recognition
> and synthesis; video and the temporal consistency problem; unified multimodal
> models and where the field is heading.

---

## Sequencing this against a job search

The portfolio argument matters more than the certificate argument: each of these
courses produces a **tested reference implementation**, which is a far stronger
artefact than a course completion. Build them in the order that matches the jobs
you are actually applying for, and finish the reference implementation for each:
a half-built course teaches you less than a small finished one.

---

## Building them: the plan

**The quality bar is the one Transformers now meets**: every chapter graded A in
the course's own `docs/CONTENT-AUDIT.md` (see `NEW-COURSE-PROMPT.md` §5), every
number backed by code and a test, every chapter with a companion notebook, every
page clean in `shoot.js` across 8 viewports and both themes.

**Order of work**

1. **Platform first** (`PLATFORM.md` §8): add the Library control and registry to
   Transformers' `course.js`, then create the hub repository
   (`ganesh-deepak.github.io`) with the library page, tracks, registry, canonical
   assets and `check_platform.js`. Every course after that is born connected.
2. **Course 1, Evaluation.** It is the multiplier: every later course uses its
   methods to prove its own claims.
3. Then by track. Everyone: 7. Systems: 3 → 4. Applied: 5 → 6. Model: 2.
   Frontier labs: 9. Breadth: 8, 10.

**How each course is produced**

1. One session writes the outline (`chapters.js` with Parts, every chapter `wip`),
   the course's `CONTENT-AUDIT.md` with the target per chapter, its semantic hue
   mapping, and its registry entry. Review the outline before any prose.
2. Parallel sessions author one Part each, on their own branches, with disjoint
   files (`NEW-COURSE-PROMPT.md` §10). A separate session writes the companion
   notebooks.
3. One session merges, runs every suite to zero problems, reads the contact
   sheets, grades every chapter, bumps the cache version and deploys.
4. Flip the course's `status` to `live` in the hub registry. The library, the
   course switcher and the prerequisite map update across every course.

**Connecting them** (`PLATFORM.md`)

- One origin, so one theme, one sidebar preference and shared progress: the
  library page shows every course's progress and a "continue" link for each.
- A **Library** control in every course's topbar to jump between courses.
- **Tracks** on the hub: the sequences above, drawn as paths through the courses.
- A **prerequisite map** built from every `xref` in every course's "Builds on"
  notes: a dependency graph of chapters across the whole library.
- **Search** across all courses' headings and glossary terms.
