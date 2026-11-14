"""
Pretraining data — Chapter 11.5.

Every pretraining pipeline is the same four steps, in this order:

    1. dedup exactly        exact_dedup()                        hash of normalised text
    2. dedup approximately  MinHasher + lsh_candidates + dedup()  MinHash / LSH (Broder 1997)
    3. filter for quality   gopher_rules() then QualityClassifier  heuristics, then a model
    4. mix and schedule     mixture_weights, epochs_per_source,   weights per source,
                            wsd_lr                                anneal on the best data

The numbers the chapter quotes (Jaccard, signature agreement, LSH catch
probabilities, epochs per source) are computed here and asserted in
tests/test_training.py.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict

import torch

# ============================================================ 1. exact dedup

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lower-case and collapse whitespace: the cheapest canonical form."""
    return _WS.sub(" ", text.lower()).strip()


def exact_dedup(docs):
    """Keep the first document of every group with identical normalised text."""
    seen, keep = set(), []
    for i, d in enumerate(docs):
        h = hashlib.sha1(normalise(d).encode("utf-8")).digest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


# ============================================================ 2. near-dup: MinHash + LSH

def shingles(text: str, n: int = 5) -> set:
    """The set of word n-grams. FineWeb uses n = 5."""
    w = normalise(text).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


_P = (1 << 61) - 1          # a Mersenne prime; all hashing is done mod p


class MinHasher:
    """
    k random hash functions h_j(x) = (a_j * x + b_j) mod p over a 64-bit hash of
    each shingle, with a_j, b_j uniform in [1, p). The signature keeps, per
    function, the MINIMUM over the document's shingles.

    The one fact that makes this work: for a random hash function,
        P[ min h(A) == min h(B) ] = |A ∩ B| / |A ∪ B| = Jaccard(A, B)
    because the minimum over A ∪ B is equally likely to be any of its elements,
    and the two minima agree exactly when that element is in both sets.
    So the fraction of agreeing signature slots estimates the Jaccard similarity.

    The functions must really scramble the order. With small a and small x,
    a*x + b never wraps mod p, every h_j sorts shingles the same way as x, and
    the slots stop being independent: the estimate comes out far too high.
    """

    def __init__(self, num_perm: int = 112, seed: int = 0):
        rng = random.Random(seed)
        self.a = [rng.randrange(1, _P) for _ in range(num_perm)]
        self.b = [rng.randrange(1, _P) for _ in range(num_perm)]
        self.num_perm = num_perm

    def signature(self, sh: set) -> list:
        xs = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") % _P
              for s in sh]
        return [min((a * x + b) % _P for x in xs) for a, b in zip(self.a, self.b)]


def signature_agreement(s1, s2) -> float:
    return sum(x == y for x, y in zip(s1, s2)) / len(s1)


def lsh_catch_probability(s: float, bands: int, rows: int) -> float:
    """
    Split the signature into `bands` bands of `rows` slots. Two documents become
    candidates if ALL rows of ANY band agree:
        P(candidate) = 1 - (1 - s^rows)^bands
    an S-curve whose midpoint is near (1/bands)^(1/rows).
    """
    return 1 - (1 - s ** rows) ** bands


def lsh_threshold(bands: int, rows: int) -> float:
    return (1 / bands) ** (1 / rows)


def lsh_candidates(signatures, bands: int, rows: int):
    """Pairs (i, j), i < j, that share at least one whole band."""
    assert bands * rows <= len(signatures[0])
    buckets = defaultdict(list)
    for i, sig in enumerate(signatures):
        for bnd in range(bands):
            buckets[(bnd, tuple(sig[bnd * rows:(bnd + 1) * rows]))].append(i)
    pairs = set()
    for ids in buckets.values():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                pairs.add((ids[x], ids[y]))
    return pairs


def dedup(docs, n: int = 5, bands: int = 14, rows: int = 8, seed: int = 0):
    """
    Near-duplicate removal. Candidates from LSH are merged with union-find
    (duplication is transitive: if A~B and B~C, keep one of the three), and
    one document per cluster survives. Defaults are FineWeb's: 5-grams,
    112 hashes = 14 bands x 8 rows.
    """
    mh = MinHasher(bands * rows, seed)
    sigs = [mh.signature(shingles(d, n)) for d in docs]
    parent = list(range(len(docs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in lsh_candidates(sigs, bands, rows):
        parent[find(j)] = find(i)
    return sorted({find(i) for i in range(len(docs))})


# ============================================================ 3. quality

STOP_WORDS = ("the", "be", "to", "of", "and", "that", "have", "with")


def gopher_rules(text: str):
    """
    The MassiveWeb heuristics of Rae et al. 2021 (Gopher), Appendix A.
    Returns the list of rules a document FAILS (empty list = keep).
    """
    words = text.split()
    lines = [l for l in text.splitlines() if l.strip()]
    fails = []
    if not 50 <= len(words) <= 100_000:
        fails.append("word count outside 50..100,000")
    if words:
        mean_len = sum(len(w) for w in words) / len(words)
        if not 3 <= mean_len <= 10:
            fails.append("mean word length outside 3..10")
        if (text.count("#") / len(words) > 0.1) or (
                (text.count("...") + text.count("…")) / len(words) > 0.1):
            fails.append("symbol-to-word ratio > 0.1")
        if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
            fails.append("< 80% of words contain a letter")
    if lines:
        if sum(l.lstrip().startswith(("•", "-", "*")) for l in lines) / len(lines) > 0.9:
            fails.append("> 90% of lines start with a bullet")
        if sum(l.rstrip().endswith(("...", "…")) for l in lines) / len(lines) > 0.3:
            fails.append("> 30% of lines end with an ellipsis")
    lw = {w.lower().strip(".,;:!?\"'()") for w in words}
    if sum(s in lw for s in STOP_WORDS) < 2:
        fails.append("fewer than 2 stop words")
    return fails


class QualityClassifier:
    """
    The FineWeb-Edu idea at toy scale: an expensive judge (there, Llama-3-70B
    scoring 460k pages 0-5) labels a sample; a CHEAP model learns to imitate it
    and is run over everything. Here the cheap model is logistic regression on
    hashed word unigrams and bigrams, so it runs anywhere.
    """

    def __init__(self, dim: int = 2**14, seed: int = 0):
        self.dim = dim
        torch.manual_seed(seed)
        self.w = torch.zeros(dim, requires_grad=True)
        self.b = torch.zeros((), requires_grad=True)

    def features(self, text: str) -> torch.Tensor:
        w = normalise(text).split()
        grams = w + [a + " " + b for a, b in zip(w, w[1:])]
        x = torch.zeros(self.dim)
        for g in grams:
            x[int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little") % self.dim] += 1
        return x / max(len(grams), 1) ** 0.5

    def fit(self, texts, labels, steps: int = 200, lr: float = 0.5):
        X = torch.stack([self.features(t) for t in texts])
        y = torch.tensor(labels, dtype=torch.float32)
        opt = torch.optim.SGD([self.w, self.b], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ self.w + self.b, y)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def score(self, text: str) -> float:
        return torch.sigmoid(self.features(text) @ self.w + self.b).item()


# ============================================================ 4. mixtures and schedules

def mixture_weights(sizes, tau: float = 1.0):
    """
    Temperature sampling over sources: p_i proportional to n_i^(1/tau).
    tau = 1 samples in proportion to size; larger tau flattens toward uniform,
    upsampling the small sources (how multilingual mixes boost rare languages).
    """
    z = [n ** (1 / tau) for n in sizes]
    s = sum(z)
    return [x / s for x in z]


def epochs_per_source(weights, sizes, total_tokens):
    """How many times each source is seen if you draw total_tokens with these weights."""
    return [w * total_tokens / n for w, n in zip(weights, sizes)]


def wsd_lr(step, base_lr, warmup, total, decay_frac=0.1, final_ratio=0.0, shape="linear"):
    """
    Warmup-Stable-Decay. Linear warmup, a long constant plateau, and a short
    decay over the last `decay_frac` of training — the phase in which runs
    switch to their highest-quality data ("annealing"). shape="sqrt" is the
    1 - sqrt decay of Hägele et al. 2024.
    """
    if step < warmup:
        return base_lr * (step + 1) / warmup
    d0 = int(total * (1 - decay_frac))
    if step < d0:
        return base_lr
    p = min((step - d0) / max(total - d0, 1), 1.0)
    f = p if shape == "linear" else math.sqrt(p)
    return base_lr * (1 - (1 - final_ratio) * f)
