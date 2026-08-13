# Transformers From Scratch

A complete, visual, hands-on course: from *"what is a tensor"* to *"here is why DeepSeek's
latent attention shrinks the KV cache ~57×"*.

Built for someone starting with LLMs who wants **real depth**, and who finds tensor shapes
and matrix operations hard to picture. Every dimension is drawn. Every claim is derived or
measured. Nothing is asserted and left there.

---

## Start here

```powershell
# 1. Open the course (this is the main thing)
python serve.py
#    -> opens http://127.0.0.1:8777/index.html
#    Light/dark toggle is in the top bar; it follows your OS by default.

# 2. Set up Python for the notebooks
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Confirm everything works
python tests/test_tfs.py        # 27 tests, ~1 minute
```

Work through the chapters **in order**. Parts 0 and I look elementary — do not skip them.
Almost everyone who finds transformers confusing is actually confused about tensor shapes.

> Run `serve.py` rather than double-clicking `index.html`: browsers block `localStorage`
> on `file://` URLs, so your chapter-progress marks wouldn't save.

---

## What's here

```
course/            22 interactive chapters. Open with serve.py. This is the course.
  index.html         roadmap, progress tracking, the dimension cheat sheet
  ch01..ch22         the chapters
  assets/            style.css, course.js (quiz engine), viz.js (12 widgets)

notebooks/         Runnable Jupyter notebooks, one per key chapter
tfs/               Reference implementation — working, tested code
  tokenizer.py       byte-level BPE (pure Python, no deps)
  model.py           RMSNorm + RoPE + GQA + SwiGLU + KV cache
  train.py           full training loop; `python -m tfs.train` trains a real model
  dpo.py             DPO, SimPO, sequence log-probs
tests/
  test_tfs.py        27 tests — every one checks a claim the course makes
  run_notebooks.py   executes all notebook code cells as a smoke test
data/              downloaded corpus + trained tokenizers (created on first run)
```

**Use `tfs/` to check your own implementation, not to skip writing one.** The gap between
"I understand attention" and "my attention passes the causality test" is where the learning is.

---

## The curriculum

| Part | Chapters | What you get |
|---|---|---|
| **0 · Seeing tensors** | 1–2 | Shapes, strides, broadcasting, matmul, einsum. The foundation everything else stands on. |
| **I · Text → vectors** | 3–4 | BPE from scratch; embeddings; **the residual stream** |
| **II · Attention** | 5–7 | Attention derived (incl. why √d_k); the multi-head reshape dance; RoPE |
| **III · The model** | 8–10 | The block, whole architectures, end-to-end shape trace, training |
| **IV · Making it fast** | 11–15 | Sampling, **KV cache & the memory wall**, MQA/GQA/**MLA**, **FlashAttention**, serving |
| **V · Making it big** | 16–18 | MoE, scaling laws, long context, the 2026 frontier |
| **VI · Making it useful** | 19–21 | SFT/RLHF/PPO, **DPO derived line by line**, GRPO & RLVR |
| **VII · Capstone** | 22 | Build the whole thing: RMSNorm+RoPE+GQA+SwiGLU+cache, trained, DPO-aligned |

~25 hours of work. 78 quiz questions, 77 exercises, 12 interactive widgets.

---

## The interactive widgets

These are the answer to "I can't picture tensor dimensions." Drag the sliders.

| Widget | Chapter | What it makes concrete |
|---|---|---|
| **matmul** | 2 | Which row and column feed each output cell; the shared dim vanishing |
| **tensor** | 1, 4 | A rank-3 tensor drawn as labelled slabs |
| **broadcast** | 2 | Shapes aligning from the right, with the failure case |
| **heads** | 6 | The `view`/`transpose` dance, drawn — same numbers, re-partitioned |
| **attention** | 5 | Scores → mask → softmax, with √d_k and causal toggles |
| **rope** | 7 | Position as rotation; why the dot product depends only on distance |
| **trace** | 9 | 19-step walk through a full forward pass, shape by shape |
| **kv** | 12 | KV cache calculator: MHA vs GQA vs MQA vs MLA |
| **gqa** | 13 | Which query heads share which KV head |
| **online** | 14 | Streaming vs one-shot softmax agreeing exactly — FlashAttention's theorem |
| **softmax** | 11 | Temperature, top-k, top-p reshaping a distribution |
| **chinchilla** | 17 | Compute-optimal model/data split |

---

## Verification

Everything the course claims is checked by a test. Notable ones:

| Test | Claim it verifies |
|---|---|
| `test_causality` | Perturbing token *t* leaves outputs at positions < *t* bit-identical |
| `test_kv_cache_matches_no_cache` | Cached and uncached greedy generation are **token-identical** |
| `test_initial_loss_is_ln_vocab` | Untrained loss = ln(V) |
| `test_rope_is_relative` | ⟨R_m q, R_n k⟩ depends only on (n−m) |
| `test_online_softmax_is_exact` | Streaming softmax == one-shot, to 1e-4, at any scale |
| `test_tiled_attention_matches_naive` | Tiled (FlashAttention-style) == naive attention |
| `test_sqrt_dk_scaling_claim` | std(q·k) == √d_k empirically, for d ∈ {16, 64, 256} |
| `test_dpo_partition_function_cancels` | Z(x) cancels — shifting both log-probs changes nothing |
| `test_param_count_matches_formula` | The analytic parameter formula matches the real model |

```powershell
python tests/test_tfs.py         # 30 tests
python tests/run_notebooks.py    # executes every notebook's code

cd tests && npm install jsdom && node check_ui.js
#   49 UI checks: WCAG contrast for every colour pair in BOTH themes, plus
#   confirmation that each page builds its layout, topbar, theme toggle,
#   copy buttons, chapter nav and quiz markers.
```

---

## The design

A technical monograph rather than a web app: a measured column with a **live margin** carrying
cross-references, so you can see how a chapter connects without losing your place.

- **Type** — `Sitka Text` / `Sitka Heading` (Matthew Carter's family, built from legibility
  research for long on-screen reading, with real optical sizes), `Bahnschrift` for technical
  labels, `Cascadia Mono` for every shape annotation. All resolve natively on Windows — no
  webfont, no silent fallback.
- **Colour** — the boldness is spent on the **dimension palette**, because in this course colour
  encodes *which tensor axis you are looking at*. Ochre, indigo, clay, moss — one per role,
  consistent in every diagram. The verdigris accent is deliberately excluded from that set so it
  can never be mistaken for a dimension.
- **Both themes** are designed, not inverted. Every pair passes WCAG AA in both.

---

## Train a real model right now

```powershell
# character-level -- best samples at this scale, matches Ch 10 Exercise 10.1
python -m tfs.train --char --steps 2500 --d_model 192 --n_layer 6 --batch 32 --lr 1e-3

# or with the BPE tokenizer (the default)
python -m tfs.train --steps 3000
```

Downloads Tiny Shakespeare, trains a tokenizer, trains a ~2.4M-parameter model with
RoPE + GQA + SwiGLU, and generates a sample. CPU only, no GPU needed.

**Two things to watch, both of which the script prints for you:**

- **Initial loss should be ≈ `ln(vocab_size)`.** If it isn't, something is broken.
  Cheapest sanity check in deep learning.
- **`bpb` (bits per byte), not raw loss.** Cross-entropy is measured *per token*, so it is
  **not comparable across tokenizers**. Two real runs of the same model on the same data:

  | tokenizer | best val loss | bytes/token | **bits/byte** | best step |
  |---|---|---|---|---|
  | character (V=65) | **1.512** | 1.00 | 2.18 | 1250 |
  | BPE (V=2048) | **4.362** | 3.29 | **1.91** | 500 |

  The BPE run's loss is **2.9× larger** and it is the **better model** — 12% fewer bits per byte,
  and visibly better samples. Judge by `bpb`, never by the loss column. (Note also that the larger
  vocabulary overfits *sooner*: 2048 embedding rows trained on ~324k tokens see each rare token only
  a handful of times.)

The `--char` run reaches val 1.51 in **~14 minutes on CPU** and produces recognisable verse with
correct speaker tags. Both runs then overfit — Tiny Shakespeare has ~1M characters against 2.4M
parameters (0.42 tokens/param, where Chinchilla wants ~20). The script flags `tokens/param`, marks
the best checkpoint, warns when train ≪ val, and restores the best weights rather than the last.

> **A bug this repo shipped, and what caught it.** `get_batch` originally returned a pre-shifted
> `y` which `Model.forward` then shifted *again* — training the model to predict **two tokens
> ahead**. The buggy version had a *lower* training loss (0.0457 vs 0.0482) and produced
> character soup. Every unit test still passed, because `forward()` was correct; the caller
> wasn't. What caught it: `test_generation_reproduces_memorised_text` — train a model to
> memorise a short text, then check greedy decoding reproduces it. Case study in Chapter 10.6.

---

## Accuracy

Architectural numbers (DeepSeek-V3 MLA ranks, GQA head counts, FlashAttention versions,
Chinchilla coefficients, MoE configs) were taken from primary papers and public model
configs and checked in **August 2026**. Where a claim is contested or an open research
question, the course flags it rather than stating it flatly. Every chapter ends with sources.

**Chapter 18 is deliberately dated** — it covers the moving frontier (DeepSeek Sparse
Attention, Mamba-2 hybrids, FlashAttention-4) and will age. Chapters 1–17 cover mechanisms
that have been stable for years.

---

## Notes

- **Windows console + Unicode:** plain `python foo.py` on a cp1252 console crashes on emoji.
  The scripts here call `sys.stdout.reconfigure(encoding="utf-8")`. Notebooks are unaffected.
- **No GPU required** anywhere. Everything is sized for a laptop CPU.
- **`einops` is optional** — used in two exercises, with fallbacks if it's missing.
