"""
Score a chapter's prose for AI tells (github.com/blader/humanizer patterns) and readability.

    python tests/humanize/scan.py course/ch05-attention.html [more files]   # one line per file
    python tests/humanize/scan.py --detail course/ch05-attention.html       # list every hit

Counts are over the prose segments only (seg.py), so code, tables and figures do not count.
Lower tells and higher reading ease are better. Readability is Flesch reading ease with a
vowel-group syllable estimate: rough, but consistent before and after, which is what matters.
"""

from __future__ import annotations

import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from seg import segments, text_of  # noqa: E402

TELLS = {
    "not X but Y (§1)": [r"\bnot (just|only|merely|simply)\b", r"\bit'?s not\b[^.]{0,80}\bit'?s\b",
                         r"\bisn'?t\b[^.]{0,60}\bit'?s\b", r"\bdoes not mean\b", r"\bnot because\b"],
    "dashes (§8)": [r"—", r"–"],
    "deep sayings (§3)": [r"\bthe real (question|point|lesson|problem|reason)\b", r"\bat its core\b", r"\bfundamentally\b",
                          r"\bthe heart of\b", r"\bthe key (insight|idea|point)\b", r"\bwhat really matters\b"],
    "run-ups (§4)": [r"\blet's\b", r"\blet us\b",r"\bhere'?s (the|what|why|how)\b", r"\bnow (let|we will) look\b", r"\bthe thing is\b"],
    "arguing with no one (§5)": [r"\bto be clear\b", r"\byou might think\b", r"\bone might\b", r"\bit would be easy to\b",
                                 r"\bthis is not to say\b"],
    "stock words (§12)": [r"\b(crucial|crucially|delve|pivotal|robust|showcase|underscore[sd]?|intricate|landscape|"
                          r"meticulous(ly)?|testament|tapestry|vibrant|enhance[sd]?|garner|interplay|seamless(ly)?|"
                          r"leverage[sd]?|additionally|notably|essentially|ultimately|simply put)\b"],
    "inflation (§13)": [r"\bplays? a (key|crucial|vital|pivotal) role\b", r"\bgame[- ]chang", r"\brevolutioni[sz]", r"\bpowerful\b"],
    "-ing tails (§15)": [r",\s(highlighting|underscoring|emphasizing|ensuring|reflecting|showcasing|making it)\b"],
    "is-avoidance (§18)": [r"\bserves as\b", r"\bstands as\b", r"\bboasts\b", r"\bfunctions as\b"],
}
BOLD = re.compile(r"<b>")


def syllables(w: str) -> int:
    w = w.lower().strip(".,;:!?()\"'")
    if not w:
        return 0
    g = re.findall(r"[aeiouy]+", w)
    n = len(g) - (1 if w.endswith("e") and len(g) > 1 else 0)
    return max(1, n)


def score(path: str, detail: bool = False) -> dict:
    s = open(path, encoding="utf-8").read()
    segs = segments(s)
    text = "\n".join(text_of(re.sub(r"<code[^>]*>.*?</code>", " X ", x[3], flags=re.S)) for x in segs)
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", text)
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text) if len(x.split()) >= 3]
    nw = max(1, len(words))
    hits = {}
    for name, pats in TELLS.items():
        found = []
        for p in pats:
            found += [m.group(0) for m in re.finditer(p, text, re.I)]
        hits[name] = found
    bold = sum(len(BOLD.findall(x[3])) for x in segs)
    syl = sum(syllables(w) for w in words)
    asl = nw / max(1, len(sents))
    fre = 206.835 - 1.015 * asl - 84.6 * syl / nw
    long_words = sum(1 for w in words if syllables(w) >= 3) / nw
    tells = sum(len(v) for v in hits.values())
    r = {"file": path.split("/")[-1].split("\\")[-1], "words": nw, "tells": tells, "per1k": tells / nw * 1000,
         "bold_per1k": bold / nw * 1000, "sent_len": asl, "ease": fre, "long%": long_words * 100, "hits": hits}
    if detail:
        for k, v in hits.items():
            if v:
                print(f"  {k:26s} {len(v):3d}  {sorted(set(x.lower() for x in v))[:12]}")
    return r


def line(r):
    return (f"{r['file']:26s} {r['words']:6d} words  tells {r['tells']:4d} ({r['per1k']:4.1f}/1k)  "
            f"bold {r['bold_per1k']:4.1f}/1k  sentence {r['sent_len']:4.1f} words  ease {r['ease']:5.1f}  long words {r['long%']:4.1f}%")


if __name__ == "__main__":
    detail = "--detail" in sys.argv
    for p in [a for a in sys.argv[1:] if not a.startswith("--")]:
        r = score(p, detail)
        print(line(r))
