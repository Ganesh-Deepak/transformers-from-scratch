"""
Chapter 13 reference code: what happens around the sampler.

    best-of-n and self-consistency      pass_at_n, pass_at_k, majority_vote_accuracy
    stop sequences while streaming      StopStreamer
    batch-dependent nondeterminism      sum_fp32, dot_fp32, find_argmax_flip
    grammar-constrained decoding        Pattern, token_mask, tokenizations, greedy_tokenize

Every number Chapter 13 quotes is produced by a function here and asserted in
tests/test_inference.py. Pure Python where exact IEEE behaviour matters (the
fp32 sums are done one addition at a time so they reproduce on any machine).
"""

from __future__ import annotations

import itertools
import math
import random
import struct
from math import comb


# ============================================================ best-of-n


def pass_at_n(p: float, n: int) -> float:
    """Chance that at least one of n independent samples is correct."""
    return 1.0 - (1.0 - p) ** n


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c are correct (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def majority_vote_accuracy(probs, correct: int, n: int) -> float:
    """
    Exact probability that a plurality vote over n samples returns the correct
    answer. probs[i] is the chance one sample gives answer i. A tie between the
    correct answer and others counts as a fair coin among the tied answers.
    """
    m = len(probs)
    total = 0.0
    # every way to split n votes among m answers
    for cut in itertools.combinations(range(n + m - 1), m - 1):
        counts, prev = [], -1
        for c in cut:
            counts.append(c - prev - 1)
            prev = c
        counts.append(n + m - 1 - prev - 1)
        top = max(counts)
        if counts[correct] != top:
            continue
        ties = sum(1 for c in counts if c == top)
        coef = math.factorial(n)
        pr = 1.0
        for c, p in zip(counts, probs):
            coef //= math.factorial(c)
            pr *= p ** c
        total += coef * pr / ties
    return total


# ============================================================ stop sequences


class StopStreamer:
    """
    Stream text to a client while watching for stop strings that may arrive
    split across tokens. Text that could still turn into a stop string is
    held back; everything else is released immediately.
    """

    def __init__(self, stops):
        self.stops = [s for s in stops if s]
        self.buf = ""
        self.stopped = False

    def _holdback(self) -> int:
        """Length of the longest suffix of buf that is a proper prefix of a stop."""
        best = 0
        for s in self.stops:
            for L in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:L]):
                    best = max(best, L)
                    break
        return best

    def push(self, piece: str) -> str:
        """Feed one decoded token's text; return the text safe to send now."""
        if self.stopped:
            return ""
        self.buf += piece
        hits = [(self.buf.find(s), s) for s in self.stops if s in self.buf]
        if hits:
            at = min(h[0] for h in hits)
            out, self.buf, self.stopped = self.buf[:at], "", True
            return out
        keep = self._holdback()
        out = self.buf[:len(self.buf) - keep]
        self.buf = self.buf[len(self.buf) - keep:]
        return out

    def flush(self) -> str:
        """End of generation without a stop: release whatever was held back."""
        out, self.buf = ("" if self.stopped else self.buf), ""
        return out


def stream(pieces, stops):
    """Run a whole token stream through a StopStreamer; return what each push emitted."""
    st = StopStreamer(stops)
    emitted = [st.push(p) for p in pieces]
    tail = st.flush()
    return emitted, tail, st.stopped


# ============================================================ fp32 reductions


def f32(x: float) -> float:
    """Round a Python float (fp64) to the nearest fp32 value."""
    return struct.unpack("f", struct.pack("f", x))[0]


def sum_fp32(values, n_chunks: int = 1) -> float:
    """
    Sum in fp32 the way a split-K kernel does: cut the vector into n_chunks
    contiguous pieces, sum each one left to right, then add the partial sums.
    Every addition is rounded to fp32, exactly as hardware would.
    """
    n = len(values)
    size = -(-n // n_chunks)
    partials = []
    for s in range(0, n, size):
        acc = 0.0
        for v in values[s:s + size]:
            acc = f32(acc + v)
        partials.append(acc)
    acc = 0.0
    for p in partials:
        acc = f32(acc + p)
    return acc


def dot_fp32(a, b, n_chunks: int = 1) -> float:
    return sum_fp32([f32(x * y) for x, y in zip(a, b)], n_chunks)


def random_fp32(n: int, seed: int):
    r = random.Random(seed)
    return [f32(r.gauss(0.0, 1.0)) for _ in range(n)]


def find_argmax_flip(d: int = 1024, seeds=range(200), chunks=(1, 8)):
    """
    Two candidate tokens whose true logits differ by less than fp32 reduction
    noise. Search deterministically for a hidden state h where computing the
    two logits with split count chunks[0] ranks them one way and chunks[1] the
    other way. Returns the first such case, with the exact (fp64) logits.
    """
    w0 = random_fp32(d, 10_000)
    for seed in seeds:
        h = random_fp32(d, seed)
        exact0 = math.fsum(x * y for x, y in zip(h, w0))
        # an unrelated weight row, with its largest-|h| weight adjusted so the
        # exact logit lands within ~1e-5 of token 0's: a genuine near-tie
        w1 = random_fp32(d, 20_000 + seed)
        j = max(range(d), key=lambda i: abs(h[i]))
        rest = math.fsum(h[i] * w1[i] for i in range(d) if i != j)
        w1[j] = f32((exact0 - rest) / h[j])
        exact1 = math.fsum(x * y for x, y in zip(h, w1))
        if exact0 == exact1:
            continue
        a = [dot_fp32(h, w0, c) for c in chunks]
        b = [dot_fp32(h, w1, c) for c in chunks]
        win = [0 if x > y else 1 if y > x else -1 for x, y in zip(a, b)]
        if -1 not in win and win[0] != win[1]:
            return dict(seed=seed, exact=(exact0, exact1), split=dict(zip(chunks, zip(a, b))),
                        winner=dict(zip(chunks, win)))
    return None


# ============================================================ constrained decoding


DIGITS = "0123456789"


class Pattern:
    """
    A tiny grammar: a sequence of literal strings and digit runs, e.g.
    ['{"n": ', DIGITS, '}'] accepts {"n": 0} ... {"n": 123}. A state is
    (segment index, chars matched in it). Enough to show how a character-level
    automaton turns into a per-token mask, which is all a grammar engine does.
    """

    def __init__(self, segments):
        self.seg = segments

    def start(self):
        return (0, 0)

    def step(self, state, ch):
        i, off = state
        while i < len(self.seg):
            s = self.seg[i]
            if s is DIGITS:
                if ch in DIGITS:
                    return (i, off + 1)
                if off == 0:
                    return None                      # a digit run needs at least one digit
                i, off = i + 1, 0                    # run over: try the next segment
                continue
            if off < len(s):
                return (i, off + 1) if s[off] == ch else None
            i, off = i + 1, 0
        return None

    def walk(self, state, text):
        for ch in text:
            state = self.step(state, ch)
            if state is None:
                return None
        return state

    def accepts(self, text):
        st = self.walk(self.start(), text)
        if st is None:
            return False
        i, off = st
        s = self.seg[i]
        complete = (off > 0) if s is DIGITS else (off == len(s))
        return complete and i == len(self.seg) - 1


def token_mask(pattern: Pattern, state, vocab):
    """Which vocabulary entries may come next: every character must keep the automaton alive."""
    return [t for t in vocab if pattern.walk(state, t) is not None]


def token_mask_whole_terminals(pattern: Pattern, state, vocab):
    """
    The naive engine: a token is allowed only if it stays inside ONE grammar
    segment. It forbids tokens like '7}' that finish a number and close the
    object in one go -- tokens real tokenizers are full of.
    """
    i, _ = state
    ok = []
    for t in vocab:
        st = pattern.walk(state, t)
        if st is not None and st[0] == i:
            ok.append(t)
    return ok


def tokenizations(text: str, vocab) -> int:
    """How many different token sequences spell exactly this text."""
    ways = [0] * (len(text) + 1)
    ways[0] = 1
    for i in range(len(text)):
        if ways[i]:
            for t in vocab:
                if text.startswith(t, i):
                    ways[i + len(t)] += ways[i]
    return ways[-1]


def greedy_tokenize(text: str, vocab):
    """Longest match first -- a stand-in for the tokenizer's canonical split."""
    out, i = [], 0
    by_len = sorted(vocab, key=len, reverse=True)
    while i < len(text):
        t = next(t for t in by_len if text.startswith(t, i))
        out.append(t)
        i += len(t)
    return out


TOY_VOCAB = ['{', '{"', '"', 'n', '":', '": ', ' ', ':', 'n"', '"n', '"n":',
             '1', '2', '7', '12', '42', '7}', '}', 'a', 'no', 'x', '}\n']
TOY_PATTERN = ['{"n": ', DIGITS, '}']
