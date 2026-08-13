"""
Byte-level BPE tokenizer — Chapter 3.

Pure Python, no dependencies. Deliberately readable rather than fast:
training on a few MB takes a minute or two, which is fine for the course.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Iterable


def get_pair_counts(ids: list[int], counts: Counter | None = None) -> Counter:
    """Count every adjacent pair in a list of token ids."""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """
    Byte-level byte-pair encoding.

    Base vocabulary is the 256 possible byte values, so *any* string is
    representable and there is never an <UNK> token.

        tok = BPETokenizer()
        tok.train(text, vocab_size=1024)
        assert tok.decode(tok.encode(s)) == s
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.special: dict[str, int] = {}
        self._vocab: dict[int, bytes] = {}

    # ---------------------------------------------------------------- train

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte alphabet)")

        ids = list(text.encode("utf-8"))
        self.merges = {}

        for new_id in range(256, vocab_size):
            counts = get_pair_counts(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break                       # nothing repeats any more
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            if verbose and (new_id - 256) % 100 == 0:
                print(f"  merge {new_id - 256:5d}: {pair} -> {new_id}  (freq {freq})")

        self._build_vocab()
        return self

    def _build_vocab(self) -> None:
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]
        for tok, tid in self.special.items():
            vocab[tid] = tok.encode("utf-8")
        self._vocab = vocab

    # ------------------------------------------------------------- specials

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        """Reserve ids above the learned merges for control tokens."""
        start = 256 + len(self.merges)
        for offset, tok in enumerate(tokens):
            if tok not in self.special:
                self.special[tok] = start + offset
        self._build_vocab()

    # --------------------------------------------------------------- encode

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if allow_special and self.special:
            return self._encode_with_special(text)
        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        if len(ids) < 2:
            return ids
        # Repeatedly apply whichever learned merge appears EARLIEST in the
        # merge order and is still present. This is the standard greedy BPE
        # encode and is what makes encoding consistent with training.
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def _encode_with_special(self, text: str) -> list[int]:
        # Split on special tokens, encode the ordinary chunks, splice ids back.
        import re

        pattern = "(" + "|".join(re.escape(t) for t in self.special) + ")"
        out: list[int] = []
        for chunk in re.split(pattern, text):
            if not chunk:
                continue
            if chunk in self.special:
                out.append(self.special[chunk])
            else:
                out.extend(self._encode_ordinary(chunk))
        return out

    # --------------------------------------------------------------- decode

    def decode(self, ids: Iterable[int]) -> str:
        parts = [self._vocab[i] for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------ utilities

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    def compression_ratio(self, text: str) -> float:
        """Bytes per token. Higher is better compression."""
        n = len(self.encode(text, allow_special=False))
        return len(text.encode("utf-8")) / max(n, 1)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "merges": [[list(k), v] for k, v in self.merges.items()],
                    "special": self.special,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.merges = {tuple(k): v for k, v in data["merges"]}
        tok.special = data.get("special", {})
        tok._build_vocab()
        return tok


class CharTokenizer:
    """
    Character-level tokenizer. Not used by real models, but it is what Chapter 10
    Exercise 10.1 uses: with vocab ~65 a tiny model reaches val loss ~1.5 on
    Shakespeare, and the samples look far better than BPE at the same compute
    (each token is one character, so the model never emits a broken word-fragment).
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.special: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def compression_ratio(self, text: str) -> float:
        return 1.0


if __name__ == "__main__":
    # Windows consoles default to cp1252 and will crash on emoji. Notebooks are
    # fine; plain `python foo.py` is not. This makes stdout UTF-8 everywhere.
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # A varied corpus. (A corpus that is one sentence repeated N times is
    # degenerate: BPE learns tokens spanning the whole thing and reports an
    # absurd compression ratio. Real text is what makes the number meaningful.)
    corpus = """
    The quick brown fox jumps over the lazy dog. Attention is all you need.
    A transformer maintains a residual stream of shape (batch, time, channel).
    Every layer reads from that stream and adds its result back into it.
    Tokenization turns text into integers; embeddings turn integers into vectors.
    The theory of the theatre is thematic. These theories theorise thoroughly.
    Machine learning models learn representations from large amounts of data.
    Softmax normalises a vector of real numbers into a probability distribution.
    def forward(self, x): return self.proj(self.attn(self.norm(x))) + x
    Positional information must be injected because attention is permutation
    equivariant: without it, "dog bites man" and "man bites dog" are identical.
    """ * 24

    tok = BPETokenizer().train(corpus, vocab_size=600)
    tok.add_special_tokens(["<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"])

    s = "the theatre theme 🤖 ñ"
    ids = tok.encode(s)
    print(f"vocab_size   {tok.vocab_size}")
    print(f"text         {s!r}")
    print(f"ids          {ids}")
    print(f"tokens       {[tok.decode([i]) for i in ids]}")
    print(f"roundtrip    {tok.decode(ids) == s}")
    print(f"compression  {tok.compression_ratio(corpus):.2f} bytes/token")
