"""
Evaluating LLMs — Chapter 26. Every number in the chapter comes from a function
here, and tests/test_posttraining.py asserts each one.

    resolution        standard_error, ci95, mde_unpaired
    scoring           normalize_answer, exact_match, token_f1, majority_vote
    sampling metrics  pass_at_k (unbiased), naive_pass_at_k, pass_hat_k
    significance      bootstrap_ci, paired_bootstrap_diff, n_items_unpaired, n_items_paired
    contamination     ngrams, ngram_overlap, min_k_prob
    judges            debias_pairwise, length_controlled_winrate, elo_win_prob
    labels            cohens_kappa
"""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from statistics import NormalDist

Z = NormalDist().inv_cdf


# ============================================================ resolution of a benchmark

def standard_error(p: float, n: int) -> float:
    """SE of an accuracy p measured on n independent items: sqrt(p(1-p)/n)."""
    return math.sqrt(p * (1 - p) / n)


def ci95(p: float, n: int) -> tuple[float, float]:
    h = Z(0.975) * standard_error(p, n)
    return p - h, p + h


def mde_unpaired(p: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest difference two models at accuracy ~p can be told apart by on n items each."""
    return (Z(1 - alpha / 2) + Z(power)) * math.sqrt(2 * p * (1 - p) / n)


# ============================================================ scoring one answer

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lower-case, drop punctuation and articles, squeeze spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str, normalize: bool = True) -> int:
    if normalize:
        return int(normalize_answer(pred) == normalize_answer(gold))
    return int(pred == gold)


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def majority_vote(answers: list[str]) -> str:
    """maj@k / cons@k: the most common (normalised) answer; ties go to the first seen."""
    norm = [normalize_answer(a) for a in answers]
    counts = Counter(norm)
    best = max(counts.values())
    return next(a for a in norm if counts[a] == best)


# ============================================================ pass@k (Chen et al. 2021)

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k from n samples of which c are correct:
        1 - C(n-c, k) / C(n, k)
    computed as a product so it never overflows (the HumanEval paper's form).
    """
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1 - k / i
    return 1 - prod


def naive_pass_at_k(n: int, c: int, k: int) -> float:
    """Plug-in estimate 1 - (1 - c/n)^k. Biased LOW, because 1-(1-p)^k is concave in p."""
    return 1 - (1 - c / n) ** k


def pass_hat_k(n: int, c: int, k: int) -> float:
    """tau-bench's pass^k: probability that k trials ALL succeed, unbiased: C(c,k)/C(n,k)."""
    return math.comb(c, k) / math.comb(n, k) if c >= k else 0.0


def expected_estimator(estimator, n: int, p: float, k: int) -> float:
    """E over c ~ Binomial(n, p) of estimator(n, c, k). Exact, no sampling."""
    return sum(math.comb(n, c) * p ** c * (1 - p) ** (n - c) * estimator(n, c, k)
               for c in range(n + 1))


# ============================================================ significance

def bootstrap_ci(scores: list[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean: resample ITEMS with replacement."""
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def paired_bootstrap_diff(a: list[float], b: list[float], n_boot: int = 10_000,
                          alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """CI for mean(b) - mean(a) when both models answered the SAME items: resample item ids."""
    assert len(a) == len(b)
    return bootstrap_ci([y - x for x, y in zip(a, b)], n_boot, alpha, seed)


def unpaired_bootstrap_diff(a, b, n_boot=10_000, alpha=0.05, seed=0):
    """The same difference, ignoring the pairing: resample each model's items independently."""
    rng = random.Random(seed)
    n, m = len(a), len(b)
    d = sorted(sum(b[rng.randrange(m)] for _ in range(m)) / m -
               sum(a[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return d[int(alpha / 2 * n_boot)], d[int((1 - alpha / 2) * n_boot) - 1]


def n_items_unpaired(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items per model to detect p1 -> p2 with independent samples (two-proportion z-test)."""
    z = Z(1 - alpha / 2) + Z(power)
    return math.ceil(z * z * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)


def n_items_paired(discordant: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Items to detect a gain delta when both models see the same items. Per item the
    difference d is in {-1, 0, +1}; Var(d) = P(models disagree) - delta^2.
    """
    z = Z(1 - alpha / 2) + Z(power)
    return math.ceil(z * z * (discordant - delta ** 2) / delta ** 2)


# ============================================================ contamination

def ngrams(tokens: list[str], n: int) -> set:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(item: str, corpus_ngrams: set, n: int = 13) -> float:
    """Fraction of the item's n-grams that also occur in the training corpus."""
    g = ngrams(item.lower().split(), n)
    return len(g & corpus_ngrams) / len(g) if g else 0.0


def min_k_prob(token_logps: list[float], k: float = 0.2) -> float:
    """
    Min-K% Prob (Shi et al. 2023): mean log-prob of the k fraction of LEAST likely tokens.
    Text seen in training has few very-surprising tokens, so its score is higher.
    """
    m = max(1, int(len(token_logps) * k))
    return sum(sorted(token_logps)[:m]) / m


# ============================================================ judges

def debias_pairwise(v_ab: str, v_ba: str) -> str:
    """
    Ask the judge twice with the order swapped. v_ab is its verdict with the model's
    answer shown FIRST ('first'|'second'), v_ba with it shown SECOND.
    A win only counts if it survives the swap; otherwise it is a tie.
    """
    model_wins_ab = v_ab == "first"
    model_wins_ba = v_ba == "second"
    if model_wins_ab and model_wins_ba:
        return "win"
    if not model_wins_ab and not model_wins_ba:
        return "loss"
    return "tie"


def _fit_logistic(X: list[list[float]], y: list[int], iters: int = 50, ridge: float = 1e-6):
    """Newton's method for logistic regression (2 or 3 features). Plain Python."""
    d = len(X[0])
    w = [0.0] * d
    for _ in range(iters):
        g = [0.0] * d
        H = [[ridge * (i == j) for j in range(d)] for i in range(d)]
        for xi, yi in zip(X, y):
            p = 1 / (1 + math.exp(-sum(a * b for a, b in zip(w, xi))))
            for i in range(d):
                g[i] += (yi - p) * xi[i]
                for j in range(d):
                    H[i][j] += p * (1 - p) * xi[i] * xi[j]
        # solve H step = g (Gauss-Jordan, d is tiny)
        M = [H[i][:] + [g[i]] for i in range(d)]
        for c in range(d):
            piv = max(range(c, d), key=lambda r: abs(M[r][c]))
            M[c], M[piv] = M[piv], M[c]
            for r in range(d):
                if r != c:
                    f = M[r][c] / M[c][c]
                    M[r] = [a - f * b for a, b in zip(M[r], M[c])]
        w = [wi + M[i][d] / M[i][i] for i, wi in enumerate(w)]
    return w


def length_controlled_winrate(wins: list[int], len_model: list[int], len_base: list[int]) -> dict:
    """
    The idea of length-controlled AlpacaEval (Dubois et al. 2024), reduced to its core:
    fit  P(win) = sigmoid(theta + phi * tanh(len_diff / sd))  and report the win rate
    with the length term set to zero, i.e. sigmoid(theta): "how often would it win if its
    answers were as long as the baseline's?" (The real metric also has a per-instruction
    difficulty term.)
    """
    diffs = [a - b for a, b in zip(len_model, len_base)]
    sd = (sum(d * d for d in diffs) / len(diffs)) ** 0.5 or 1.0
    X = [[1.0, math.tanh(d / sd)] for d in diffs]
    theta, phi = _fit_logistic(X, wins)
    return {"raw": sum(wins) / len(wins), "lc": 1 / (1 + math.exp(-theta)), "theta": theta, "phi": phi}


def elo_win_prob(delta: float) -> float:
    """Arena-style ratings: P(A beats B) = 1 / (1 + 10^(-(R_A - R_B)/400))."""
    return 1 / (1 + 10 ** (-delta / 400))


# ============================================================ labels for a private eval

def cohens_kappa(a: list, b: list) -> float:
    """Agreement between two annotators corrected for agreement by chance."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe)


def perplexity(mean_nll: float) -> float:
    return math.exp(mean_nll)
