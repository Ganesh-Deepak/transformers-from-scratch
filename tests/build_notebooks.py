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


@notebook("08_the_block")
def _nb08():
    return [
        md("""
        # Chapter 8 — The Transformer Block

        Every number in Chapter 8, recomputed with `tfs/model.py` (`RMSNorm`, `SwiGLU`, `Block`,
        `Config.n_params`). Each cell **asserts** what it checks, so `python tests/run_notebooks.py 08`
        fails if the chapter and the code disagree. Exercise 8.4 needs GPT-2's weights and is not
        reproduced.
        """),
        code(SETUP + "from tfs.model import Config, Model, RMSNorm, SwiGLU\n"),
        md("""
        ## 8.2 — the worked example: LayerNorm and RMSNorm on x = [3, 1, 4, 8]

        **DISCREPANCY (typo).** The chapter's code-cell comment prints the RMSNorm output as
        `[ 0.633, 0.211, 0.843, 1.687]`; the prose above it (correctly) says `0.632`.
        3 / √22.5 = 0.63246, which rounds to 0.632 with or without the 1e-5 epsilon.
        """),
        code("""
        x = torch.tensor([3., 1., 4., 8.])
        mu, var = x.mean(), x.var(unbiased=False)
        assert mu.item() == 4 and var.item() == 6.5 and round(var.sqrt().item(), 4) == 2.5495
        ln = (x - mu) / torch.sqrt(var + 1e-5)
        ms = x.pow(2).mean()
        assert ms.item() == 22.5 and round(ms.sqrt().item(), 4) == 4.7434
        rms = x * torch.rsqrt(ms + 1e-5)
        r3 = lambda t: [round(v, 3) for v in t.tolist()]
        print("LN ", r3(ln), "  RMS", r3(rms))
        assert r3(ln) == [-0.392, -1.177, 0.0, 1.569]
        assert r3(rms) == [0.632, 0.211, 0.843, 1.687]                               # not 0.633
        assert abs(ln.mean()) < 1e-6 and abs(ln.std(unbiased=False) - 1) < 1e-4
        assert round(rms.mean().item(), 3) == 0.843 and abs(rms.pow(2).mean().sqrt() - 1) < 1e-4
        cos = F.cosine_similarity(ln, rms, dim=0).item()
        print(f"cosine(LN, RMS) = {cos:.4f}")
        assert round(cos, 2) == 0.54
        # the module in tfs/model.py gives the same RMSNorm
        assert torch.allclose(RMSNorm(4, eps=1e-5)(x), rms, atol=1e-6)
        # shift every element: LayerNorm is unchanged, RMSNorm moves
        ln2 = (x + 10 - (x + 10).mean()) / torch.sqrt((x + 10).var(unbiased=False) + 1e-5)
        assert torch.allclose(ln, ln2, atol=1e-5) and not torch.allclose(RMSNorm(4)(x + 10), RMSNorm(4)(x), atol=1e-2)
        """),
        md("""
        ## 8.2 — how close LN and RMSNorm are on realistic vectors (d = 768)

        The chapter: median cosine 0.9997 over 2000 random zero-mean vectors, worst case 0.994, and 0.45
        once the mean is pushed 2σ off zero (1/√5 = 0.447 analytically).
        """),
        code("""
        torch.manual_seed(0)
        def cos_ln_rms(v):
            l = F.layer_norm(v, (v.shape[-1],), eps=1e-5)
            r = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)
            return F.cosine_similarity(l, r, dim=-1)
        c = cos_ln_rms(torch.randn(2000, 768))
        c_shift = cos_ln_rms(torch.randn(2000, 768) + 2.0)
        print(f"median {c.median():.4f}  worst {c.min():.4f}  shifted by 2 sigma: {c_shift.median():.3f}")
        assert round(c.median().item(), 4) == 0.9997 and 0.99 < c.min().item() < 0.999
        assert round(c_shift.median().item(), 2) == 0.45 and abs(c_shift.median().item() - 1 / math.sqrt(5)) < 0.01
        """),
        md("""
        ## 8.2 — LayerNorm ≠ BatchNorm: a token's output never depends on the rest of the batch
        """),
        code("""
        torch.manual_seed(0)
        a, others = torch.randn(1, 5, 32), torch.randn(7, 5, 32)
        lnm = nn.LayerNorm(32)
        assert torch.allclose(lnm(a), lnm(torch.cat([a, others]))[:1], atol=1e-6)
        bn = nn.BatchNorm1d(32).train()
        alone = bn(a.reshape(-1, 32))
        assert not torch.allclose(alone, bn(torch.cat([a, others]).reshape(-1, 32))[:5], atol=1e-3)
        """),
        md("""
        ## 8.2 — pre-LN never normalises the stream, so its magnitude grows with depth (hence a final norm)
        """),
        code("""
        torch.manual_seed(0)
        m = Model(Config(vocab_size=256, d_model=64, n_layer=8, n_head=4, n_kv_head=2, d_head=16, d_ff=128, max_T=64))
        rms_by_layer = []
        hooks = [b.register_forward_hook(lambda mod, i, o: rms_by_layer.append(o[0].pow(2).mean().sqrt().item()))
                 for b in m.blocks]
        m(torch.randint(0, 256, (4, 64)))
        for h in hooks: h.remove()
        print("stream RMS after each block:", [f"{v:.5f}" for v in rms_by_layer])
        assert rms_by_layer == sorted(rms_by_layer) and rms_by_layer[-1] > rms_by_layer[0]
        assert isinstance(m.norm, RMSNorm)                     # the final norm before the LM head
        """),
        md("## 8.3 — the MLP is a per-position function: changing other tokens cannot change its output"),
        code("""
        cfg = Config(vocab_size=256, d_model=64, n_layer=1, n_head=4, n_kv_head=2, d_head=16, d_ff=172, max_T=64)
        mlp = SwiGLU(cfg)
        h = torch.randn(1, 10, 64)
        h2 = h.clone(); h2[0, 3:] = torch.randn(7, 64)
        assert torch.allclose(mlp(h)[0, :3], mlp(h2)[0, :3])
        """),
        md("## 8.3 — two-thirds of the model is in the MLP: 12C² per layer, GPT-2 small, Llama-2-7B"),
        code("""
        C = 768
        attn, mlp4 = 4 * C * C, 8 * C * C
        assert mlp4 == 2 * attn and attn + mlp4 == 12 * C * C
        body = 12 * 12 * C ** 2
        print(f"GPT-2 small: 12*12*768^2 = {body / 1e6:.1f}M + 38.6M embedding = {(body + 50257 * 768) / 1e6:.1f}M")
        assert round(body / 1e6) == 85 and round((body + 50257 * 768) / 1e6) == 124
        llama = Config(vocab_size=32000, d_model=4096, n_layer=32, n_head=32, n_kv_head=32, d_head=128,
                       d_ff=11008, tie_embed=False)
        per = 3 * 4096 * 11008
        print(f"Llama-2-7B: {llama.n_params():,} params; MLP {per / 1e6:.2f}M/layer x32 = {32 * per / 1e9:.2f}B"
              f" = {32 * per / llama.n_params():.1%} (of 6.7B: {32 * per / 6.7e9:.1%})")
        assert llama.n_params() == 6_738_415_616
        assert round(per / 1e6, 2) == 135.27 and round(4096 * 11008 / 1e6, 2) == 45.09
        assert round(32 * per / 1e9, 2) == 4.33 and round(100 * 32 * per / 6.7e9) == 65
        assert 11008 / 4096 == 2.6875 and 11008 % 256 == 0
        """),
        md("## Exercise 8.3 — `swiglu_ff_dim`, and Llama-2-70B's 28672"),
        code("""
        def swiglu_ff_dim(d_model, multiple_of=256, ffn_dim_multiplier=None):
            hidden = int(2 * 4 * d_model / 3)
            if ffn_dim_multiplier is not None:
                hidden = int(ffn_dim_multiplier * hidden)
            return multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        print(swiglu_ff_dim(4096), swiglu_ff_dim(8192), swiglu_ff_dim(8192, 4096, 1.3))
        assert swiglu_ff_dim(4096) == 11008 and swiglu_ff_dim(8192) == 22016
        assert swiglu_ff_dim(8192, multiple_of=4096, ffn_dim_multiplier=1.3) == 28672   # Llama-2-70B's config
        # parameter match: SwiGLU at (8/3)C has the parameters of a GELU MLP at 4C
        assert abs(3 * 4096 * swiglu_ff_dim(4096) / (2 * 4096 * 4 * 4096) - 1) < 0.01
        """),
        md("""
        ## Exercise 8.1 — both norms on a random (4, 128, 512) input, verified and timed

        The chapter: cosine typically > 0.98; RMSNorm "usually 20–40% faster in a pure-Python benchmark".
        The speed ratio depends heavily on the machine and on how `var` is implemented; on this CPU the
        hand-written RMSNorm is several times faster than the hand-written LayerNorm, while PyTorch's
        fused `nn.LayerNorm` beats both. The assert is only that the hand-written RMSNorm is faster.
        """),
        code("""
        import time
        class LayerNorm(nn.Module):
            def __init__(self, d, eps=1e-5):
                super().__init__()
                self.g = nn.Parameter(torch.ones(d)); self.b = nn.Parameter(torch.zeros(d)); self.eps = eps
            def forward(self, x):
                mu = x.mean(-1, keepdim=True)
                var = x.var(-1, keepdim=True, unbiased=False)
                return (x - mu) / torch.sqrt(var + self.eps) * self.g + self.b
        torch.manual_seed(0)
        x = torch.randn(4, 128, 512)
        mine, ref, rn = LayerNorm(512), nn.LayerNorm(512), RMSNorm(512)
        with torch.no_grad():
            assert torch.allclose(mine(x), ref(x), atol=1e-5)
            lo, ro = mine(x), rn(x)
            assert lo.mean(-1).abs().max() < 1e-5 and (lo.std(-1, unbiased=False) - 1).abs().max() < 1e-3
            assert (ro.pow(2).mean(-1).sqrt() - 1).abs().max() < 1e-3
            cs = F.cosine_similarity(lo, ro, dim=-1)
            print(f"cosine LN vs RMS: min {cs.min():.4f}")
            assert cs.min() > 0.98
            def bench(f, reps=200):
                for _ in range(5): f(x)
                t = time.perf_counter()
                for _ in range(reps): f(x)
                return (time.perf_counter() - t) / reps
            t_ln, t_rms = bench(mine), bench(lambda v: v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6))
        print(f"hand-written LN {t_ln * 1e3:.2f} ms   hand-written RMS {t_rms * 1e3:.2f} ms   ({t_ln / t_rms:.1f}x)")
        assert t_rms < t_ln
        """),
        md("""
        ## Exercise 8.2 — pre-LN vs post-LN gradients across 24 layers

        Built with PyTorch's standard `nn.TransformerEncoderLayer` (`norm_first=False` is post-LN,
        `True` is pre-LN; LayerNorm, default init), one forward + backward on random data, gradient norm
        of each layer's attention input projection.

        **DISCREPANCY (magnitude).** The direction matches the chapter: post-LN gradients are largest at
        the top and smaller at layer 0, pre-LN gradients are roughly flat. But the size of the effect is
        ~5–10×, not "2–4 orders of magnitude". (With this course's own init — weights at std 0.02 and
        un-normalised embeddings — a post-LN variant does show a ~10³ ratio, but only because layer 0
        alone sees the tiny raw embedding; layers 1–23 are flat. That is an init artefact, not geometric
        decay.) The well-known result (Xiong et al. 2020) is the direction; the "2–4 orders" figure
        should be softened or re-measured.
        """),
        code("""
        def grad_profile(norm_first, L=24, d=64):
            torch.manual_seed(0)
            layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=norm_first)
            enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
            out = enc(torch.randn(8, 32, d))
            if norm_first:
                out = F.layer_norm(out, (d,))                 # pre-LN needs its final norm
            (out @ torch.randn(d)).pow(2).mean().backward()
            return [l.self_attn.in_proj_weight.grad.norm().item() for l in enc.layers]
        post, pre = grad_profile(False), grad_profile(True)
        print("post-LN", " ".join(f"{v:.1e}" for v in post[::4]), f"  top/bottom {post[-1] / post[0]:.1f}x")
        print("pre-LN ", " ".join(f"{v:.1e}" for v in pre[::4]), f"  top/bottom {pre[-1] / pre[0]:.1f}x")
        assert post[-1] / post[0] > 2                          # post-LN: smaller at layer 0 ...
        assert post[-1] / post[0] < 100                        # ... but not by 2-4 orders of magnitude
        assert 0.3 < pre[-1] / pre[0] < 3                      # pre-LN: roughly flat
        """),
    ]


@notebook("10_backprop")
def _nb10():
    return [
        md("""
        # Chapter 10 — The Backward Pass, Derived

        Every hand-derived gradient in Chapter 10, checked against autograd, plus every worked number and
        exercise answer. Each cell **asserts** what it checks, so `python tests/run_notebooks.py 10` fails
        if the chapter and the code disagree. Float64 throughout unless a cell says otherwise, as the
        chapter recommends.
        """),
        code(SETUP + "torch.set_default_dtype(torch.float64)\n"),
        md("## 10.1 — matmul: the shapes force dA = dC Bᵀ and dB = Aᵀ dC"),
        code("""
        A = torch.randn(4, 3, requires_grad=True); Bm = torch.randn(3, 5, requires_grad=True)
        Cm = A @ Bm; g = torch.randn_like(Cm)
        Cm.backward(g)
        assert A.grad.shape == A.shape == (4, 3) and Bm.grad.shape == (3, 5)
        assert torch.allclose(A.grad, g @ Bm.detach().T) and torch.allclose(Bm.grad, A.detach().T @ g)
        """),
        md("## 10.2 — the softmax Jacobian at x = [1.0, 2.0, 0.5]"),
        code("""
        x = torch.tensor([1.0, 2.0, 0.5])
        s = x.softmax(0)
        J_auto = torch.autograd.functional.jacobian(lambda t: t.softmax(0), x)
        J_formula = torch.diag(s) - torch.outer(s, s)
        print("s =", [round(v, 4) for v in s.tolist()])
        print(J_formula.numpy().round(4))
        assert [round(v, 4) for v in s.tolist()] == [0.2312, 0.6285, 0.1402]
        assert torch.allclose(J_auto, J_formula)
        assert J_formula.numpy().round(4).tolist() == [[0.1778, -0.1453, -0.0324],
                                                       [-0.1453, 0.2335, -0.0881],
                                                       [-0.0324, -0.0881, 0.1206]]
        assert torch.allclose(J_formula, J_formula.T) and torch.allclose(J_formula.sum(1), torch.zeros(3))
        """),
        md("## 10.2 — the vector-Jacobian product, which is what actually runs"),
        code("""
        def softmax_backward(s, g):
            return s * (g - (g * s).sum(-1, keepdim=True))
        x = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
        s = x.softmax(0); g = torch.tensor([0.3, -1.2, 0.7])
        s.backward(g)
        assert torch.allclose(x.grad, softmax_backward(s.detach(), g), atol=1e-6)
        assert torch.allclose(x.grad, J_formula.T @ g)                     # = J^T g, never forming J
        """),
        md("## 10.3 — cross-entropy + softmax collapses to p − y (logits [2.0, 1.0, 0.1], class 0)"),
        code("""
        z = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        F.cross_entropy(z, torch.tensor([0])).backward()
        p = z.detach().softmax(-1)[0]
        y = torch.tensor([1.0, 0, 0])
        print("p", [round(v, 4) for v in p.tolist()], " p - y", [round(v, 4) for v in (p - y).tolist()])
        assert [round(v, 4) for v in p.tolist()] == [0.659, 0.2424, 0.0986]
        assert [round(v, 4) for v in (p - y).tolist()] == [-0.341, 0.2424, 0.0986]
        assert torch.allclose(z.grad[0], p - y)
        """),
        md("## 10.4 — RMSNorm's correction term, at x = [3, 1, 4, 8], g = [1, 0, 0, 0]"),
        code("""
        def rmsnorm_backward(x, g, gamma, eps=0.0):
            d = x.shape[-1]
            r = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
            gg = g * gamma
            return (gg - x * (gg * x).sum(-1, keepdim=True) / (d * r ** 2)) / r
        x = torch.tensor([3., 1., 4., 8.], requires_grad=True)
        gamma = torch.ones(4); g = torch.tensor([1., 0., 0., 0.])
        (x / x.pow(2).mean().sqrt() * gamma).backward(g)
        print("autograd", [round(v, 6) for v in x.grad.tolist()])
        assert round(x.pow(2).mean().sqrt().item(), 4) == 4.7434
        assert [round(v, 6) for v in x.grad.tolist()] == [0.189737, -0.007027, -0.028109, -0.056218]
        assert torch.allclose(x.grad, rmsnorm_backward(x.detach(), g, gamma), atol=1e-12)
        print("dot(grad, x) =", torch.dot(x.grad, x.detach()).item())
        assert abs(torch.dot(x.grad, x.detach())) < 1e-12                   # orthogonal to x
        # and the same holds for the float32 module in tfs/model.py, with a learned gamma
        from tfs.model import RMSNorm
        xr = torch.randn(5, 16, dtype=torch.float32, requires_grad=True)
        nrm = RMSNorm(16, eps=0.0).float()
        with torch.no_grad(): nrm.g.uniform_(0.5, 1.5)
        gr = torch.randn(5, 16, dtype=torch.float32)
        nrm(xr).backward(gr)
        assert torch.allclose(xr.grad, rmsnorm_backward(xr.detach(), gr, nrm.g.detach()), atol=1e-5)
        """),
        md("## 10.5 — attention backward: dQ, dK, dV against autograd (T = 4, d = 3)"),
        code("""
        def attention_backward(q, k, v, P, dO):
            dk_ = q.shape[-1]
            dV = P.transpose(-2, -1) @ dO
            dP = dO @ v.transpose(-2, -1)
            dS = P * (dP - (dP * P).sum(-1, keepdim=True))
            dQ = dS @ k / dk_ ** 0.5
            dK = dS.transpose(-2, -1) @ q / dk_ ** 0.5
            return dQ, dK, dV
        torch.manual_seed(0)
        q, k, v = (torch.randn(4, 3, requires_grad=True) for _ in range(3))
        P = (q @ k.T / 3 ** 0.5).softmax(-1)
        O = P @ v
        dO = torch.randn_like(O)
        O.backward(dO)
        mine = attention_backward(q.detach(), k.detach(), v.detach(), P.detach(), dO)
        for auto, man in zip((q.grad, k.grad, v.grad), mine):
            assert torch.allclose(auto, man, atol=1e-5)
        print("dQ, dK, dV all match autograd")
        """),
        md("## 10.6 — the residual connection over depth: 0.9ᴸ, and 1 + f′"),
        code("""
        for L, want, weaker in ((20, 0.1216, 8), (50, 0.0052, 190), (100, 0.0000266, 37_600)):
            print(f"{L:3d} layers: 0.9^{L} = {0.9 ** L:.7f}  ({1 / 0.9 ** L:,.0f}x weaker)")
            assert round(0.9 ** L, 7 if L == 100 else 4) == want
            assert abs(1 / 0.9 ** L - weaker) / weaker < 0.03            # 8x, ~190x, ~37,600x
        x = torch.tensor(2.0, requires_grad=True)
        (x + 0.3 * x ** 2).backward()
        assert abs(x.grad.item() - 2.2) < 1e-12                           # 1 + f'(2) = 1 + 1.2
        """),
        md("## Exercise 10.1 — the chapter's `check` harness on every layer, including a linear layer"),
        code("""
        def check(fn_forward, manual_backward, *inputs):
            ins = [t.clone().requires_grad_(True) for t in inputs]
            out = fn_forward(*ins)
            g = torch.randn_like(out)
            out.backward(g)
            mine = manual_backward(*[t.detach() for t in inputs], g)
            for auto, man in zip([t.grad for t in ins], mine):
                assert torch.allclose(auto, man, atol=1e-5), (auto, man)
            return "ok"
        torch.manual_seed(0)
        X, W, b = torch.randn(6, 5), torch.randn(5, 4), torch.randn(4)
        assert check(lambda x, w, b: x @ w + b, lambda x, w, b, g: (g @ w.T, x.T @ g, g.sum(0)), X, W, b) == "ok"
        assert check(lambda x: x.softmax(-1), lambda x, g: (softmax_backward(x.softmax(-1), g),), torch.randn(3, 7)) == "ok"
        ga = torch.rand(8) + 0.5
        assert check(lambda x: x / x.pow(2).mean(-1, keepdim=True).sqrt() * ga,
                     lambda x, g: (rmsnorm_backward(x, g, ga),), torch.randn(3, 8)) == "ok"
        def attn_fwd(q, k, v):
            return (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(-1) @ v
        def attn_bwd(q, k, v, g):
            P = (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(-1)
            return attention_backward(q, k, v, P, g)
        assert check(attn_fwd, attn_bwd, *(torch.randn(2, 6, 4) for _ in range(3))) == "ok"
        print("linear, softmax, RMSNorm, attention: ok")
        """),
        md("""
        ## Exercise 10.2 — gradient checking with central differences

        **Minor correction.** The chapter says forward differences give ~1e-6 error and central
        differences ~1e-12 at ε = 1e-6. The truncation errors are O(ε) and O(ε²), as stated, but in
        float64 the central difference is limited by rounding, ~machine-ε/ε ≈ 1e-10: measured below it
        lands at ~1e-11 to 1e-10, not 1e-12. Still four orders better than the forward difference, which
        is the point. In float32 the same check is useless, as the chapter warns.
        """),
        code("""
        def grad_check(f, x, eps=1e-6, dtype=torch.float64):
            x = x.to(dtype)
            num = torch.zeros_like(x)
            for i in range(x.numel()):
                xp, xm = x.clone(), x.clone()
                xp.view(-1)[i] += eps; xm.view(-1)[i] -= eps
                num.view(-1)[i] = (f(xp) - f(xm)) / (2 * eps)
            xa = x.clone().requires_grad_(True); f(xa).backward()
            return ((num - xa.grad).norm() / (num.norm() + xa.grad.norm())).item()
        torch.manual_seed(0)
        loss = lambda t: (t.softmax(-1) * torch.arange(6.0, dtype=t.dtype)).sum() + t.pow(2).mean().sqrt()
        rel64 = grad_check(loss, torch.randn(6))
        rel32 = grad_check(loss, torch.randn(6), eps=1e-3, dtype=torch.float32)
        print(f"relative error: float64 {rel64:.1e}   float32 {rel32:.1e}")
        assert rel64 < 1e-7 and rel32 > 1e-5
        x1 = torch.tensor(1.0)
        fwd = ((torch.sin(x1 + 1e-6) - torch.sin(x1)) / 1e-6 - math.cos(1)).abs().item()
        cen = ((torch.sin(x1 + 1e-6) - torch.sin(x1 - 1e-6)) / 2e-6 - math.cos(1)).abs().item()
        print(f"d sin/dx at 1: forward-difference error {fwd:.1e}, central {cen:.1e}")
        assert 1e-7 < fwd < 1e-5 and cen < 1e-9 and fwd / cen > 1e3
        """),
        md("""
        ## Exercise 10.3 — a 40-layer tanh stack, with and without the residual

        Gradient of the loss with respect to each layer's input (the signal backprop carries).

        **DISCREPANCY (init-dependent figure).** The chapter expects "4–6 orders of magnitude between
        layer 40 and layer 1" without residuals. The size depends entirely on the weight scale, which the
        exercise does not fix: with PyTorch's default `nn.Linear` init (std ≈ 1/√(3d)) the decay is
        ~10 orders of magnitude; with std = 1/√d it is barely one. The qualitative result holds either way:
        geometric decay without the residual, and with it the gradient does not decay at all (it grows a
        little toward layer 1).
        """),
        code("""
        def profile(residual, std=None, d=64, L=40):
            torch.manual_seed(0)
            Ws = [nn.Linear(d, d, bias=False) for _ in range(L)]
            if std is not None:
                for W in Ws: nn.init.normal_(W.weight, 0, std)
            h = torch.randn(32, d, requires_grad=True); hs = []
            for W in Ws:
                hs.append(h); h.retain_grad()
                h = h + torch.tanh(W(h)) if residual else torch.tanh(W(h))
            (h @ torch.randn(d)).sum().backward()
            return [t.grad.norm().item() for t in hs]                      # layer 1 ... layer 40
        for name, std in (("default init", None), ("std 1/sqrt(d)", 1 / 8)):
            plain, res = profile(False, std), profile(True, std)
            orders = math.log10(plain[-1] / plain[0])
            print(f"{name:14}: no residual, layer40/layer1 = 10^{orders:.1f};  residual: {res[0] / res[-1]:.1f}x")
            assert res[0] >= res[-1]                                        # residual: no vanishing
            if std is None:
                assert orders > 6                                           # more than the chapter's 4-6
            else:
                assert orders < 4                                           # less than the chapter's 4-6
        """),
        md("""
        ## Exercise 10.4 — break cross-entropy on purpose

        `a` (logits) and `c` (`log(softmax)` then `nll_loss`) agree; `b` softmaxes twice. At ×50 the small
        probabilities underflow to exactly 0, so `c` hits log(0) = −inf, while the fused `a` stays finite.
        """),
        code("""
        torch.set_default_dtype(torch.float32)              # the failure is a float32 one
        torch.manual_seed(0)
        logits = torch.randn(4, 10, requires_grad=True)
        target = torch.randint(0, 10, (4,))
        def loss_and_grad(fn):
            if logits.grad is not None: logits.grad = None
            out = fn(); out.backward()
            return out.item(), logits.grad.clone()
        a, ga = loss_and_grad(lambda: F.cross_entropy(logits, target))
        b, gb = loss_and_grad(lambda: F.cross_entropy(logits.softmax(-1), target))
        c, gc = loss_and_grad(lambda: F.nll_loss(logits.softmax(-1).log(), target))
        print(f"a {a:.4f}  b {b:.4f}  c {c:.4f};  |grad a - grad b| = {(ga - gb).norm():.3f}")
        assert abs(a - c) < 1e-5 and torch.allclose(ga, gc, atol=1e-6)
        assert abs(a - b) > 0.1 and (ga - gb).norm() > 0.1
        assert b > math.log(10) - 0.2 * math.log(10)                       # double softmax: nearly uniform
        big = (logits.detach() * 50).requires_grad_(True)
        t_low = big.argmin(-1)                                              # targets whose probability underflows
        a50 = F.cross_entropy(big, t_low)
        c50 = F.nll_loss(big.softmax(-1).log(), t_low)
        print(f"x50: a = {a50.item():.1f}   c = {c50.item()}   zeros in softmax: {(big.softmax(-1) == 0).sum().item()}")
        assert (big.softmax(-1) == 0).any() and math.isinf(c50.item()) and math.isfinite(a50.item())
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
