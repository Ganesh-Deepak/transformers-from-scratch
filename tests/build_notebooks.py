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
