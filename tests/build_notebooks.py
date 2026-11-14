"""
Build the companion notebooks that recompute every number a chapter states.

    python tests/build_notebooks.py            # write all notebooks this script owns
    python tests/build_notebooks.py 07 13      # only matching ones

Each notebook is a list of (kind, source) cells, kind in {"md", "code"}. Every code
cell ASSERTS the claim it checks, so `python tests/run_notebooks.py NN` fails the
moment a chapter and the code disagree. Cells marked DISCREPANCY assert what the
code measures and say, in the markdown above them, how the chapter text differs.

The notebooks are written without outputs: run them (Jupyter, or
tests/run_notebooks.py) to see the numbers.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB_DIR = os.path.join(ROOT, "notebooks")

# Same path setup as notebooks/20_moe.ipynb: works from the repo root or from notebooks/.
SETUP = """\
import math, sys, os
sys.path.insert(0, os.path.abspath(".."))      # run from notebooks/
sys.path.insert(0, os.path.abspath("."))       # or from the repo root
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
"""

# Course text as an offline corpus (chapters 1, 2, 5, 6): real English prose, no download.
CORPUS_HELPER = '''\
import re, html as _html
def course_text(*names):
    """Visible prose of course chapters: tags, code, scripts and SVG stripped."""
    out = []
    for name in names:
        path = next(p for p in (os.path.join("course", name), os.path.join("..", "course", name))
                    if os.path.exists(p))
        s = open(path, encoding="utf-8").read()
        s = re.search(r"<main.*?</main>", s, re.S).group(0)
        s = re.sub(r"<(script|style|svg|pre)[^>]*>.*?</\\1>", " ", s, flags=re.S)
        s = re.sub(r"<[^>]+>", " ", s)
        out.append(re.sub(r"\\s+", " ", _html.unescape(s)))
    return " ".join(out)
'''

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {}


def notebook(name: str):
    def deco(fn):
        NOTEBOOKS[name] = fn()
        return fn
    return deco


def md(s: str) -> tuple[str, str]:
    return ("md", textwrap.dedent(s).strip("\n"))


def code(*parts: str) -> tuple[str, str]:
    """Each part is dedented on its own, so a shared helper can prefix an indented cell."""
    return ("code", "\n".join(textwrap.dedent(p).strip("\n") for p in parts))


def to_ipynb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "metadata": {}, "source": lines,
                        "outputs": [], "execution_count": None})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


# ============================================================ notebooks
# (one function per chapter, in chapter order)

@notebook("03_tokenization")
def _nb03():
    return [
        md("""
        # Chapter 3 — Tokenization

        Every number and worked example in Chapter 3, recomputed with `tfs/tokenizer.py` and the
        chapter's own code. Each cell **asserts** what it checks, so `python tests/run_notebooks.py 03`
        fails if the chapter and the code disagree. GPT-2's real tokenizer would need a download, so
        the exercises that use it run on a byte-level BPE trained here on the course's own prose
        (a small analogue: same shape of result, different exact counts).
        """),
        code(SETUP + "from collections import Counter\nfrom tfs.tokenizer import BPETokenizer, get_pair_counts, merge\n"),
        md("""
        ## 3.2 — the hand-worked example: `"low low low lower lowest"`

        The chapter trains BPE on letters (`_` is a space). The same greedy rule — most frequent pair,
        first-seen wins a tie (what `Counter.most_common` and the chapter's `max(counts, key=counts.get)`
        both do) — on character tokens.
        """),
        code("""
        seq = list("low low low lower lowest".replace(" ", "_"))
        assert len(seq) == 24 and len(set(seq)) == 8              # 24 characters, vocab of 8
        steps = []
        for _ in range(4):
            counts = Counter(zip(seq, seq[1:]))
            (a, b), n = counts.most_common(1)[0]
            top = counts.most_common(3)
            seq = merge(seq, (a, b), a + b)
            steps.append((a + b, n, len(seq), top))
            print(f"merge {a!r:7}+{b!r:6} (count {n})  -> length {len(seq):2d}   top pairs {top}")
        assert [s[0] for s in steps] == ["lo", "low", "_low", "_low_low"]
        assert [s[1] for s in steps] == [5, 5, 4, 2]
        assert [s[2] for s in steps] == [19, 14, 10, 9]                    # lengths after each merge
        # step 1: (l,o) 5, (o,w) 5, (w,_) 3, (w,e) 2;  step 3: (_,low) 4 beats (low,_) 3, (low,e) 2
        c0 = Counter(zip(list("low_low_low_lower_lowest"), list("low_low_low_lower_lowest")[1:]))
        assert (c0["l", "o"], c0["o", "w"], c0["w", "_"], c0["w", "e"]) == (5, 5, 3, 2)
        assert steps[2][3] == [(("_", "low"), 4), (("low", "_"), 3), (("low", "e"), 2)]
        assert 8 + len(steps) == 12                                        # final vocab: 12 tokens
        print(f"24 characters -> {len(seq)} tokens = {24 / len(seq):.2f}x compression")
        assert round(24 / len(seq), 1) == 2.7
        """),
        md("""
        ## 3.2 — the chapter's 40-line BPE, run as printed

        Its `encode` replays merges in learned order; `tfs.tokenizer` instead applies whichever learned
        merge has the lowest rank at each step. Both must give the same ids, and both must round-trip.
        """),
        code('''
        def get_pair_counts_ch(ids):
            return Counter(zip(ids, ids[1:]))

        def train_bpe(text, vocab_size):
            ids = list(text.encode("utf-8"))
            merges = {}
            for new_id in range(256, vocab_size):
                counts = get_pair_counts_ch(ids)
                if not counts:
                    break
                pair = max(counts, key=counts.get)
                if counts[pair] < 2:
                    break
                ids = merge(ids, pair, new_id)
                merges[pair] = new_id
            return merges

        def encode_ch(text, merges):
            ids = list(text.encode("utf-8"))
            for pair, new_id in merges.items():
                ids = merge(ids, pair, new_id)
            return ids

        def decode_ch(ids, merges):
            vocab = {i: bytes([i]) for i in range(256)}
            for (a, b), new_id in merges.items():
                vocab[new_id] = vocab[a] + vocab[b]
            return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

        text = "the theory of the theatre is the theme " * 40
        merges = train_bpe(text, vocab_size=280)
        ids = encode_ch("the theatre theme", merges)
        print(len(ids), ids, repr(decode_ch(ids, merges)))
        assert len("the theatre theme".encode()) == 17 and len(ids) < 17   # "far shorter than the 17 raw bytes"
        assert decode_ch(ids, merges) == "the theatre theme"                # perfectly reversible
        tok = BPETokenizer().train(text, vocab_size=280)
        assert tok.merges == merges and tok.encode("the theatre theme") == ids   # same merges, same ids
        '''),
        md("""
        ## 3.2 — why merge order matters (quiz 1)

        `"the"` can only be formed after `"th"` exists. Applying the same merges in the wrong order gives a
        different, inconsistent tokenization.
        """),
        code("""
        rev = dict(reversed(list(merges.items())))                 # same merges, reversed order
        wrong = encode_ch("the theatre theme", rev)
        print("learned order", len(ids), "  reversed order", len(wrong))
        assert wrong != ids and decode_ch(wrong, merges) == "the theatre theme"
        """),
        md("""
        ## 3.2 — byte-level BPE never needs `<UNK>` (quiz 2), and Unicode's size

        Any string falls back to its bytes. And text unlike the training corpus can need more tokens
        than it has characters — counted in bytes, a Devanagari letter is 3 and an emoji is 4.
        `unicodedata` (Unicode 15.1 in this Python) gives the "~150,000 assigned code points" figure:
        characters that are neither unassigned, private-use nor surrogates.
        """),
        code("""
        import unicodedata
        for s in ["🤖 emoji", "बिल्ली चटाई पर बैठी", "\\x00\\xff corrupted"]:
            e = tok.encode(s)
            assert tok.decode(e) == s and max(e) < tok.vocab_size
            print(f"{s!r:28} {len(s):3d} chars -> {len(e):3d} tokens")
        assert len(tok.encode("🤖")) == 4 > len("🤖")             # more tokens than characters
        assigned = sum(unicodedata.category(chr(c)) not in ("Cn", "Co", "Cs") for c in range(0x110000))
        print("Unicode", unicodedata.unidata_version, "assigned characters:", assigned)
        assert 140_000 < assigned < 160_000
        """),
        md("""
        ## Pre-tokenization: the GPT-2 split

        The `regex` package (for `\\p{L}`, `\\p{N}`) is not a dependency here, so this uses the standard
        library equivalent on ASCII input: `[^\\W\\d_]` for letters, `\\d` for digits.

        **DISCREPANCY.** The chapter's comment says the split is
        `['Hello', ' world', "'s", ' ', '123', ' items', '!']`. The pattern's ` ?\\p{N}+` alternative takes
        the leading space with the digits, exactly as ` ?\\p{L}+` does for words, so the real output has
        `' 123'` as one piece: `['Hello', ' world', "'s", ' 123', ' items', '!']`.
        """),
        code('''
        import re
        try:
            import regex
            GPT2_PAT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+""")
        except ImportError:
            GPT2_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\\W\\d_]+| ?\\d+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+""")
        pieces = GPT2_PAT.findall("Hello world's 123 items!")
        print(pieces)
        assert pieces == ["Hello", " world", "'s", " 123", " items", "!"]     # chapter prints ' ', '123'
        assert all(len(GPT2_PAT.findall(s)) == 2 for s in ("dog.", "3cats"))  # never one token across categories
        '''),
        md("""
        ## 3.3 / quiz — what a bigger vocabulary costs in parameters
        """),
        code("""
        d = 4096
        added = (256_000 - 32_000) * d
        print(f"32k -> 256k at d=4096: {added:,} params per embedding matrix")
        assert added == 224_000 * 4096 and round(added / 1e9, 1) == 0.9
        tied = (128_000 - 32_000) * d
        print(f"32k -> 128k, tied: {tied:,}   untied: {2 * tied:,}")
        assert tied == 393_216_000 and round(2 * tied / 1e6) == 786
        # "1000 tokens ~ 750 words ~ 1.5 pages": 0.75 words/token, 500 words/page
        assert 1000 * 0.75 == 750 and 750 / 500 == 1.5
        """),
        md("""
        ## Exercise 3.1 — train a BPE, read its merges, plot compression against vocab size

        The corpus is the visible prose of Chapters 1 and 2 (≈27 kB); held-out text is Chapter 6. Training
        once to the largest size and truncating the merge list gives exactly the tokenizer each smaller
        vocabulary would have trained (BPE is greedy, so the first *n* merges never depend on later ones).
        Other sessions edit these chapters, so the asserts check the *shape* of the result, not exact counts.
        """),
        code(CORPUS_HELPER, """
        train_text = course_text("ch01-tensors.html", "ch02-matmul.html")[:30000]
        held_out = course_text("ch06-mha.html")[:6000]
        big = BPETokenizer().train(train_text, vocab_size=2000)
        first = [big._vocab[256 + i] for i in range(30)]
        print("first 30 merges:", first)
        assert all(len(t) == 2 for t in first[:5])                 # frequent bigrams first ...
        assert any(t in (b"the", b"the ", b" the") for t in first)  # ... then whole common words
        assert any(b"ing" in big._vocab[i] for i in range(256, 256 + 200))   # ... and suffixes
        all_merges = dict(big.merges)
        ratios = {}
        for V in (300, 500, 1000, 2000):
            big.merges = {p: i for p, i in all_merges.items() if i < V}
            big._build_vocab()
            ratios[V] = big.compression_ratio(held_out)
            print(f"vocab {V:5d} ({256 + len(big.merges):5d} used): {ratios[V]:.3f} bytes/token on held-out text")
        big.merges = all_merges; big._build_vocab()
        r = [ratios[V] for V in (300, 500, 1000, 2000)]
        assert r == sorted(r)                                      # always improving ...
        per_token = [(r[1] - r[0]) / 200, (r[2] - r[1]) / 500, (r[3] - r[2]) / 1000]
        print("gain per added vocab entry:", [f"{g:.5f}" for g in per_token])
        assert per_token[0] > per_token[1] > per_token[2]          # ... with diminishing returns
        print(f"(training stopped at {len(all_merges)} merges: no pair occurs twice after that)")
        """),
        md("""
        ## Exercise 3.2 and §3.5 — forensics and the trailing-space trap, with pre-tokenization

        The tokenizer above runs BPE on raw text, so it learns merges like `"is "` with the space at the
        *end*. Real tokenizers split with the GPT-2 regex first and run BPE inside each piece, which glues
        the space to the *front* of the next word (`" is"`). This small trainer does exactly that, on the
        same corpus, and the forensics below use it.
        """),
        code("""
        def train_pretok(text, vocab_size):
            words = Counter(GPT2_PAT.findall(text))
            seqs = {w: list(w.encode("utf-8")) for w in words}
            merges = {}
            for new_id in range(256, vocab_size):
                counts = Counter()
                for w, n in words.items():
                    for p in zip(seqs[w], seqs[w][1:]):
                        counts[p] += n
                if not counts:
                    break
                pair, f = counts.most_common(1)[0]
                if f < 2:
                    break
                seqs = {w: merge(s, pair, new_id) for w, s in seqs.items()}
                merges[pair] = new_id
            t = BPETokenizer(); t.merges = merges; t._build_vocab()
            return t

        pre = train_pretok(train_text, 1000)
        enc = lambda s: [i for piece in GPT2_PAT.findall(s) for i in pre._encode_ordinary(piece)]
        lead = sum(pre._vocab[i][:1] == b" " for i in pre.merges.values())
        trail = sum(pre._vocab[i][-1:] == b" " and len(pre._vocab[i]) > 1 for i in pre.merges.values())
        print(f"pre-tokenized BPE: {lead} tokens start with a space, {trail} end with one")
        assert lead > 100 and trail == 0
        forms = {s: enc(s) for s in ("hello", " hello", "Hello", "HELLO")}
        for s, e in forms.items():
            print(f"{s!r:10} {e}  {[pre.decode([i]) for i in e]}")
        assert len({tuple(e) for e in forms.values()}) == 4        # casing and leading space change the ids
        assert len(enc("🤖")) >= 2                                   # an emoji costs several tokens
        a, b = enc("The answer is"), enc("The answer is ")
        print([pre.decode([i]) for i in a], "|", [pre.decode([i]) for i in b])
        assert pre.decode([a[-1]]) == " is"                           # the space travels with the word ...
        assert b[:-1] == a and pre.decode([b[-1]]) == " "             # ... so a trailing space is a lone token
        """),
        md("""
        ## 3.5 — the multilingual tax, and Exercise 3.3's bill

        An English-trained tokenizer has merges for English and none for Devanagari, so the Hindi sentence
        stays near raw bytes (3 per letter). The ratio here is larger than the chapter's 3–7× because this
        tokenizer saw no Hindi at all; the direction is the point.
        """),
        code("""
        en, hi = "The cat sat on the mat", "बिल्ली चटाई पर बैठी"
        ne, nh = len(big.encode(en)), len(big.encode(hi))
        print(f"English {ne} tokens, Hindi {nh} tokens: {nh / ne:.1f}x")
        assert nh / ne > 3
        price = 3 / 1e6                                           # $3 per million input tokens
        print(f"same meaning: ${ne * price * 1e6:.0f} vs ${nh * price * 1e6:.0f} per million copies")
        assert nh * price > 3 * ne * price
        """),
    ]


@notebook("04_embeddings")
def _nb04():
    return [
        md("""
        # Chapter 4 — Embeddings and the Residual Stream

        Every number in Chapter 4, recomputed. Each cell **asserts** what it checks, so
        `python tests/run_notebooks.py 04` fails if the chapter and the code disagree. GPT-2's weights
        would need a download, so Exercise 4.2 (nearest neighbours) and the logit lens are not reproduced
        here; Exercise 4.3 runs on this course's model (`tfs/model.py`) trained for a few seconds on the
        bundled offline corpus.
        """),
        code(SETUP + "from tfs.model import Config, Model\n"),
        md("## 4.1 — an embedding is a lookup: shapes and the GPT-2 table"),
        code("""
        V, C = 50257, 768
        emb = nn.Embedding(V, C)
        print(tuple(emb.weight.shape), f"{emb.weight.numel():,} parameters")
        assert emb.weight.numel() == 38_597_376 and round(emb.weight.numel() / 1e6, 1) == 38.6
        ids = torch.tensor([[15496, 995, 0], [464, 3797, 3332]])          # (B=2, T=3)
        x = emb(ids)
        assert ids.shape == (2, 3) and x.shape == (2, 3, 768)              # (B, T) -> (B, T, C)
        onehot = torch.zeros(1, V); onehot[0, 15496] = 1.0
        assert torch.allclose(onehot @ emb.weight, emb.weight[15496][None])  # identical answers
        """),
        md("## 4.1 — the worked example: the same row, two ways, and the waste"),
        code("""
        W = torch.arange(24, dtype=torch.float64).reshape(6, 4) / 10        # id i -> [0.4i, ..., 0.4i+0.3]
        assert W[4].tolist() == [1.6, 1.7, 1.8, 1.9] and W[5].tolist() == [2.0, 2.1, 2.2, 2.3]
        oh = torch.zeros(6, dtype=torch.float64); oh[3] = 1
        print("matmul", (oh @ W).tolist(), " lookup", W[3].tolist())
        assert torch.allclose(oh @ W, torch.tensor([1.2, 1.3, 1.4, 1.5], dtype=torch.float64))
        assert torch.equal(oh @ W, W[3])
        mults, zero = 6 * 4, 5 * 4
        print(f"{zero} of {mults} multiplies by zero = {zero / mults:.0%}")
        assert round(100 * zero / mults) == 83
        gpt2 = 50257 * 768
        print(f"GPT-2: matmul {gpt2:,} multiplies, lookup 768 reads -> {gpt2 // 768:,}x the arithmetic")
        assert gpt2 == 38_597_376 and gpt2 // 768 == 50257 and 50257 - 1 == 50256
        """),
        md("""
        ## 4.1 — the gradient reaches exactly one row per token, and what AdamW does afterwards

        A token that appears once gets one gradient. Its row then keeps moving: on momentum, and from
        weight decay, which shrinks it every step.

        **DISCREPANCY (minor).** The chapter says the row "keeps drifting for a few hundred steps on
        momentum". After the gradient stops, Adam's step is about m̂/√v̂, which decays like
        (β₁/√β₂)ᵗ: under 1% of its first size after ~45 steps with PyTorch's default betas, and the
        measurement below agrees. Momentum drift is over within tens of steps; what keeps moving the row
        for hundreds of steps is weight decay (lr·λ·w every step).
        """),
        code("""
        def rare_token_moves(weight_decay, betas, steps=400):
            torch.manual_seed(0)
            e = nn.Embedding(10, 4)
            opt = torch.optim.AdamW(e.parameters(), lr=1e-2, weight_decay=weight_decay, betas=betas)
            moves = []
            for step in range(steps):
                ids = torch.tensor([3]) if step == 0 else torch.tensor([1])   # token 3 appears once, ever
                before = e.weight[3].detach().clone()
                opt.zero_grad()
                (e(ids) * torch.randn(1, 4)).sum().backward()
                if step == 0:
                    rows = (e.weight.grad.abs().sum(1) > 0).nonzero().flatten().tolist()
                    assert rows == [3]                                     # exactly one row has a gradient
                opt.step()
                moves.append((e.weight[3].detach() - before).norm().item())
            return moves

        for betas in ((0.9, 0.999), (0.9, 0.95)):
            m = rare_token_moves(0.0, betas)
            gone = next(t for t, v in enumerate(m) if v < 0.01 * m[0])
            print(f"betas {betas}: momentum drift falls below 1% of its first step after {gone} steps")
            assert gone < 100                                              # tens of steps, not hundreds
        m = rare_token_moves(0.1, (0.9, 0.95))
        print(f"with weight decay 0.1: still moving {m[300]:.1e} per step at step 300")
        assert m[300] > 1e-4                                               # decay keeps shrinking the row
        """),
        md("## 4.2 — weight tying: V·C parameters instead of 2·V·C"),
        code("""
        cfg = Config(vocab_size=1000, d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_head=16, d_ff=128, max_T=64)
        tied = Model(cfg)
        untied = Model(Config(**{**cfg.__dict__, "tie_embed": False}))
        n = lambda m: sum(p.numel() for p in m.parameters())
        print("tied", n(tied), " untied", n(untied))
        assert tied.head.weight is tied.embed.weight and tied.head.weight.shape == (1000, 64)  # both (V, C)
        assert n(untied) - n(tied) == 1000 * 64 and n(tied) == cfg.n_params()
        """),
        md("## 4.3 — the residual stream has one shape on every line"),
        code("""
        shapes = []
        hooks = [b.register_forward_hook(lambda mod, inp, out: shapes.append(tuple(out[0].shape)))
                 for b in tied.blocks]
        tied(torch.randint(0, 1000, (3, 17)))
        for h in hooks: h.remove()
        print(shapes)
        assert shapes == [(3, 17, 64)] * cfg.n_layer                      # (B, T, C) after every block
        """),
        md("## 4.4 — why x + f(x): 0.9⁸⁰, and the 1 + f′(x) term"),
        code("""
        print(f"0.9**80 = {0.9 ** 80:.2e}")
        assert round(0.9 ** 80, 4) == 0.0002
        x = torch.tensor(2.0, requires_grad=True)
        (x + torch.tanh(0.5 * x)).backward()
        fprime = 0.5 * (1 - torch.tanh(torch.tensor(1.0)) ** 2)
        assert torch.isclose(x.grad, 1 + fprime)                           # d(x + f)/dx = 1 + f'(x)
        """),
        md("## Quiz — how much of an 8B model is embedding tables?"),
        code("""
        table = 128256 * 4096
        print(f"one table {table:,}; two {2 * table / 1e9:.2f} B = {2 * table / 8e9:.0%} of 8 B")
        assert round(table / 1e6) == 525 and round(2 * table / 1e9, 2) == 1.05
        assert round(100 * 2 * table / 8e9) == 13 and 2 * table / 1e9 > 1.0   # over 100% of a 1B model
        """),
        md("""
        ## Exercise 4.1 — embedding = one-hot matmul, and the speed ratio

        The chapter predicts the lookup is typically 100–1000× faster at V = 50,000. Timings depend on
        the machine and its load: this notebook measured 120–1000× on a busy 8-core CPU. The assert is a
        deliberately loose 30× so that a loaded machine does not fail it; the printed ratio is the result.
        """),
        code("""
        import time
        e10 = nn.Embedding(10, 4)
        assert torch.equal(e10(torch.tensor([3])), F.one_hot(torch.tensor([3]), 10).float() @ e10.weight)
        E = nn.Embedding(50000, 768)
        ids = torch.randint(0, 50000, (8, 32))
        def bench(fn, reps=5):
            fn(); t = time.perf_counter()
            for _ in range(reps): fn()
            return (time.perf_counter() - t) / reps
        with torch.no_grad():
            assert torch.allclose(E(ids), F.one_hot(ids, 50000).float() @ E.weight)
            t_lookup = bench(lambda: E(ids))
            t_matmul = bench(lambda: F.one_hot(ids, 50000).float() @ E.weight)
        print(f"lookup {t_lookup * 1e3:.3f} ms   one-hot matmul {t_matmul * 1e3:.1f} ms   ratio {t_matmul / t_lookup:,.0f}x")
        assert t_matmul / t_lookup > 30
        """),
        md("""
        ## Exercise 4.3 — break the residual stream

        The same model with `x = attn(norm(x))` and `x = mlp(norm(x))` in every block (residual removed),
        trained side by side on the bundled offline corpus. The no-residual model trains dramatically
        worse, as the chapter says.

        **DISCREPANCY.** Two of the exercise's predictions do not hold for this pre-norm (RMSNorm) model:
        (1) the gap is not "marginal at 2 layers" — it is already large at 2 layers; (2) the no-residual
        model's first-layer gradients are *not* orders of magnitude smaller — they are about as large or
        larger. With a norm in front of every sublayer, each block rescales its input, so the gradient
        does not vanish; what is lost is the identity path that carries each token's own embedding
        forward. (The vanishing-gradient picture holds for an unnormalised stack — see the Chapter 10
        notebook, Exercise 10.3.)
        """),
        code("""
        from tfs.train import _fallback_corpus, get_batch
        from tfs.tokenizer import CharTokenizer
        text = _fallback_corpus()[:60000]
        ctok = CharTokenizer(text)
        data = torch.tensor(ctok.encode(text))

        def no_residual(self, x, cos, sin, cache=None):
            a, c = self.attn(self.n1(x), cos, sin, cache)
            x = a                                       # was: x = x + a
            x = self.mlp(self.n2(x))                    # was: x = x + self.mlp(self.n2(x))
            return x, c

        def run(n_layer, residual, steps=60):
            torch.manual_seed(0)
            m = Model(Config(vocab_size=ctok.vocab_size, d_model=48, n_layer=n_layer, n_head=4,
                             n_kv_head=4, d_head=12, d_ff=128, max_T=64))
            if not residual:
                for b in m.blocks:
                    b.forward = no_residual.__get__(b)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
            torch.manual_seed(1)
            for step in range(steps):
                x = get_batch(data, 16, 64)
                _, loss, _ = m(x, targets=x)
                opt.zero_grad(); loss.backward()
                if step == 0:
                    g0 = m.blocks[0].attn.q_proj.weight.grad.norm().item()
                opt.step()
            return loss.item(), g0

        res = {(L, r): run(L, r) for L in (2, 4) for r in (True, False)}
        for (L, r), (loss, g0) in res.items():
            print(f"{L} layers  residual={r!s:5}  loss after 60 steps {loss:.3f}   block[0] grad {g0:.2e}")
        for L in (2, 4):
            assert res[L, False][0] > res[L, True][0] + 0.5            # dramatically worse, even at 2 layers
            assert res[L, False][1] > 0.1 * res[L, True][1]            # first-layer grads NOT orders smaller
        """),
    ]


@notebook("07_positions")
def _nb07():
    return [
        md("""
        # Chapter 7 — Position: Sinusoidal → Learned → RoPE → ALiBi

        Every equation, worked example and exercise answer in Chapter 7, recomputed with
        `tfs/model.py`'s RoPE (`build_rope_cache`, `rotate_half`, `apply_rope`). Each cell **asserts**
        what it checks, so `python tests/run_notebooks.py 07` fails if the chapter and the code disagree.
        """),
        code(SETUP + "from tfs.model import build_rope_cache, rotate_half, apply_rope\n"),
        md("""
        ## 7.1 / Exercise 7.1 — attention is permutation-equivariant; RoPE and the causal mask break it
        """),
        code("""
        torch.manual_seed(0)
        B, T, d = 1, 6, 16
        Wq, Wk, Wv = (torch.randn(d, d, dtype=torch.float64) / 4 for _ in range(3))
        def attn(x, rope=False, causal=False):
            q, k, v = (x @ Wq)[:, None], (x @ Wk)[:, None], (x @ Wv)[:, None]   # (B, 1, T, d)
            if rope:
                cos, sin = build_rope_cache(x.shape[1], d)
                q, k = apply_rope(q, k, cos.double(), sin.double())
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)[:, 0]
        x = torch.randn(B, T, d, dtype=torch.float64)
        perm = torch.randperm(T)
        same = lambda kw: torch.allclose(attn(x, **kw)[:, perm], attn(x[:, perm], **kw))
        print("no position, no mask:", same({}), "  + RoPE:", same({"rope": True}), "  + causal mask:", same({"causal": True}))
        assert same({}) and not same({"rope": True}) and not same({"causal": True})
        # and the per-position MLP cannot fix it: it commutes with any permutation
        mlp = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, d)).double()
        assert torch.allclose(mlp(x)[:, perm], mlp(x[:, perm]))
        """),
        md("""
        ## 7.2 — sinusoidal encoding: PE(pos + k) is a fixed linear map (a rotation) of PE(pos)
        """),
        code("""
        def sinusoidal(T, d):
            pos = torch.arange(T)[:, None]
            i = torch.arange(0, d, 2)[None, :]
            ang = pos / (10000 ** (i / d))
            pe = torch.zeros(T, d)
            pe[:, 0::2] = torch.sin(ang)
            pe[:, 1::2] = torch.cos(ang)
            return pe.double()
        pe = sinusoidal(200, 16)
        k = 7
        w = 1 / 10000 ** (torch.arange(0, 16, 2).double() / 16)        # per-pair frequency
        M = torch.zeros(16, 16, dtype=torch.float64)                    # block-diagonal 2x2 rotations by k*w
        for j, a in enumerate(k * w):
            c, s = torch.cos(a), torch.sin(a)
            M[2 * j:2 * j + 2, 2 * j:2 * j + 2] = torch.tensor([[c, -s], [s, c]])
        assert torch.allclose(pe[k:], pe[:-k] @ M, atol=1e-5)           # the same M for every position
        print("PE[pos+7] == PE[pos] @ M for all", 200 - k, "positions")
        """),
        md("## 7.3 — a learned table has a hard ceiling (GPT-2: 1024 rows)"),
        code("""
        wpe = nn.Embedding(1024, 768)
        assert wpe.weight.numel() == 1024 * 768                           # max_T * d parameters
        try:
            wpe(torch.tensor([1024])); raised = False
        except IndexError:
            raised = True
        assert raised                                                      # position 1025 does not exist
        cos, sin = build_rope_cache(40_000, 64)                             # RoPE: any position, no table
        assert torch.isfinite(cos).all() and cos.shape == (40_000, 64)
        """),
        md("""
        ## 7.4 — the 2-D case: Rₘᵀ Rₙ = Rₙ₋ₘ, so only the offset survives
        """),
        code("""
        R = lambda a: torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=torch.float64)
        for m, n in [(0, 3), (5, 2), (11, 40)]:
            assert torch.allclose(R(m).T @ R(n), R(n - m))
        assert torch.allclose(R(0.7).T @ R(0.7), torch.eye(2, dtype=torch.float64))   # orthogonal: length kept
        """),
        md("## 7.4 — the worked example (d_head = 8): frequencies, a rotating q, and the cancelling positions"),
        code("""
        theta = [10000 ** (-2 * i / 8) for i in range(4)]
        for i, t in enumerate(theta):
            print(f"pair i={i}: theta={t:.6f}   full turn every {2 * math.pi / t:,.2f} tokens")
        assert [round(t, 6) for t in theta] == [1.0, 0.1, 0.01, 0.001]
        assert [round(2 * math.pi / t, 2) for t in theta] == [6.28, 62.83, 628.32, 6283.19]
        q1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        rows = [(R(m) @ q1).tolist() for m in range(4)]
        print([[round(v, 4) for v in r] for r in rows], [round(math.degrees(m), 1) for m in range(4)])
        assert [[round(v, 4) for v in r] for r in rows] == [[1.0, 0.0], [0.5403, 0.8415], [-0.4161, 0.9093], [-0.99, 0.1411]]
        assert [round(math.degrees(m), 1) for m in range(4)] == [0.0, 57.3, 114.6, 171.9]
        q, k = torch.tensor([0.6, 0.8], dtype=torch.float64), torch.tensor([1.0, 0.0], dtype=torch.float64)
        dots = [(R(m) @ q) @ (R(n) @ k) for m, n in [(0, 2), (1, 3), (5, 7), (10, 12)]]
        print("offset 2:", [round(v.item(), 6) for v in dots], "  offset 3:", round(((R(1) @ q) @ (R(4) @ k)).item(), 6))
        assert all(round(v.item(), 6) == 0.47775 for v in dots)
        assert round(((R(1) @ q) @ (R(4) @ k)).item(), 6) == -0.481099
        """),
        md("""
        ## The chapter's own code cell, in float32

        **DISCREPANCY (wording).** The text says the four scores are "bit-identical because the gap was
        always 2". Mathematically they are equal; computed in float32 (as the chapter's `rot` does, via
        `torch.tensor`) they agree to ~3×10⁻⁸ but are *not* bit-identical. The chapter's own assert
        (`< 1e-6`) is the right test; "bit-identical" is not.
        """),
        code("""
        def rot(v, m, theta=1.0):
            a = m * theta
            return torch.tensor([v[0]*math.cos(a) - v[1]*math.sin(a),
                                 v[0]*math.sin(a) + v[1]*math.cos(a)])
        q, k = torch.tensor([0.6, 0.8]), torch.tensor([1.0, 0.0])
        vals = [torch.dot(rot(q, m), rot(k, n)).item() for m, n in [(0, 2), (1, 3), (5, 7), (10, 12)]]
        print([f"{v:.10f}" for v in vals])
        assert all(round(v, 6) == 0.47775 for v in vals)                  # the printed claim
        assert abs(torch.dot(rot(q, 0), rot(k, 2)) - torch.dot(rot(q, 10), rot(k, 12))) < 1e-6
        assert len(set(vals)) > 1                                         # ... but not bit-identical
        """),
        md("""
        ## Exercise 7.2 — the relative-position property for the real implementation (d = 64)

        **Minor wording.** The chapter expects the three offset-3 scores to "agree to ~6 decimal places".
        In float32 with scores around ±6 they agree to ~6 *significant figures* (differences of ~1e-5);
        six decimal places is beyond float32 at that magnitude. The offset-4 score differs in the first digit.
        """),
        code("""
        torch.manual_seed(0)
        d = 64
        cos, sin = build_rope_cache(200, d)
        q, k = torch.randn(1, 1, 1, d), torch.randn(1, 1, 1, d)
        def score(m, n):
            qm = q * cos[m] + rotate_half(q) * sin[m]
            kn = k * cos[n] + rotate_half(k) * sin[n]
            return (qm * kn).sum().item()
        s = [score(5, 8), score(50, 53), score(100, 103)]
        print(s, score(5, 9))
        assert max(s) - min(s) < 1e-4 * max(1.0, abs(s[0]))               # agree to ~6 significant digits
        assert abs(score(5, 9) - s[0]) > 1e-3                              # offset 4 is different
        # apply_rope on (B, nh, T, hd) does the same thing at every position
        qq, kk = apply_rope(q.expand(1, 1, 200, d), k.expand(1, 1, 200, d), cos, sin)
        assert abs((qq[0, 0, 5] * kk[0, 0, 8]).sum().item() - s[0]) < 1e-5
        """),
        md("""
        ## 7.4 — two conventions, both correct: half-split (Llama/HF) vs adjacent pairs (the paper)

        They differ by a fixed permutation of the feature axis, so one equals the other in a permuted
        basis — and they are not interchangeable on the same weights.
        """),
        code("""
        def rope_adjacent(x, pos, base=10000.0):
            d = x.shape[-1]
            inv = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
            a = pos[:, None].float() * inv                               # (T, d/2)
            x1, x2 = x[..., 0::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x1 * a.cos() - x2 * a.sin()
            out[..., 1::2] = x1 * a.sin() + x2 * a.cos()
            return out
        d, T = 16, 10
        x = torch.randn(T, d)
        cos, sin = build_rope_cache(T, d)
        half = x * cos + rotate_half(x) * sin
        P = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])     # adjacent -> half-split layout
        inv = torch.argsort(P)
        assert torch.allclose(rope_adjacent(x[:, inv], torch.arange(T))[:, P], half, atol=1e-6)   # equivalent
        assert not torch.allclose(rope_adjacent(x, torch.arange(T)), half, atol=1e-3)            # not interchangeable
        print("half-split RoPE == adjacent-pair RoPE in the permuted basis; on the same weights they differ")
        """),
        md("""
        ## 7.5 — ALiBi: a per-head linear distance penalty with geometric slopes
        """),
        code("""
        n_head, T = 8, 6
        slopes = torch.tensor([2 ** (-8 * (h + 1) / n_head) for h in range(n_head)])
        print("slopes", slopes.tolist())
        assert slopes[:3].tolist() == [0.5, 0.25, 0.125]                     # 1/2, 1/4, 1/8, ...
        i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
        bias = slopes[:, None, None] * (j - i)                                # (H, T, T)
        assert (bias.masked_select((j <= i).expand(n_head, T, T)) <= 0).all()   # a penalty, never a bonus
        assert torch.equal(bias[:, 3, 1], bias[:, 5, 3])                      # depends only on j - i
        """),
        md("""
        ## Exercise 7.3 — the RoPE wavelength spectrum at d_head = 128, base 10,000 vs 500,000
        """),
        code("""
        d = 128
        for base, want_long, want_n in ((10_000, 5.4e4, 18), (500_000, 2.6e6, 32)):
            wl = [2 * math.pi * base ** (2 * i / d) for i in range(d // 2)]
            n_long = sum(w > 4096 for w in wl)
            cut = (d / 2) * math.log(4096 / (2 * math.pi)) / math.log(base)     # the closed form
            print(f"base {base:>7,}: shortest {wl[0]:.2f}, longest {wl[-1]:.3g} tokens, "
                  f"{n_long}/64 pairs longer than 4096 (closed form: i > {cut:.2f})")
            assert round(wl[0], 2) == 6.28 and round(wl[-1], -int(math.log10(want_long)) + 1) == want_long
            assert n_long == want_n == sum(i > cut for i in range(d // 2))
        """),
    ]


# ============================================================ END NOTEBOOKS


def main() -> int:
    filters = sys.argv[1:]
    names = [n for n in NOTEBOOKS if not filters or any(f in n for f in filters)]
    if not names:
        print("no matching notebooks")
        return 1
    for name in names:
        path = os.path.join(NB_DIR, name + ".ipynb")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(to_ipynb(NOTEBOOKS[name]), f, indent=1, ensure_ascii=False)
            f.write("\n")
        n_assert = sum(src.count("assert ") for kind, src in NOTEBOOKS[name] if kind == "code")
        print(f"  wrote notebooks/{name}.ipynb  ({len(NOTEBOOKS[name])} cells, {n_assert} asserts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
