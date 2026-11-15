"""
Humanizing pass for the Transformers chapters: split a chapter into prose segments,
apply rewrites by segment id, and refuse any rewrite that changes a fact.

    python tests/humanize/seg.py dump  course/ch05-attention.html        # writes tests/humanize/work/ch05.seg.txt
    python tests/humanize/seg.py apply course/ch05-attention.html work/ch05.new.txt   # verify, then write

A segment is the inner HTML of one leaf prose element: <p>, <li> (no nested list),
.qq / .opt / .expl, a worked-example step's text, <figcaption>, a non-orient .mn note.
Code, <pre>, maths blocks, tables, SVG, headings and the orient note are never touched.

The rewrite file holds blocks:
    @@ 17
    new inner HTML for segment 17
    @@ 18
    ...
Only listed segments change. For each, the verifier requires the SAME multiset of:
numbers, <code> contents, href targets, <sub>/<sup> contents, span.math contents,
span.shp badges and <a ...> tags; and no em/en dashes in the new prose.
"""

from __future__ import annotations

import html
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")

LEAF = [
    ("p", re.compile(r"<p(?:\s[^>]*)?>(.*?)</p>", re.S)),
    ("li", re.compile(r"<li(?:\s[^>]*)?>(.*?)</li>", re.S)),
    ("qq", re.compile(r'<div class="qq">(.*?)</div>', re.S)),
    ("opt", re.compile(r'<div class="opt">(.*?)</div>', re.S)),
    ("expl", re.compile(r'<div class="expl">(.*?)</div>', re.S)),
    ("step", re.compile(r'<div class="i">[^<]*</div><div>(.*?)</div></div>', re.S)),
    ("cap", re.compile(r"<figcaption>(.*?)</figcaption>", re.S)),
    ("mn", re.compile(r'<div class="mn">(.*?)</div>', re.S)),
]
BAD_INNER = re.compile(r"<(div|ul|ol|li|p|pre|table|svg|figure)\b", re.I)


def blocked_ranges(s: str):
    """Regions never edited: pre, svg, script, style, tables, maths blocks, the orient note."""
    out = []
    for pat in (r"<pre\b.*?</pre>", r"<svg\b.*?</svg>", r"<script\b.*?</script>", r"<style\b.*?</style>",
                r"<table\b.*?</table>", r'<div class="mathblock">.*?</div>', r'<div class="mn orient">.*?</div>\s*\n',
                r"<head>.*?</head>", r"<h[1-6][^>]*>.*?</h[1-6]>"):
        for m in re.finditer(pat, s, re.S):
            out.append((m.start(), m.end()))
    return out


def segments(s: str):
    blocked = blocked_ranges(s)
    inside = lambda a, b: any(x <= a and b <= y for x, y in blocked)  # noqa: E731
    segs = []
    for kind, rx in LEAF:
        for m in rx.finditer(s):
            a, b = m.start(1), m.end(1)
            inner = m.group(1)
            if inside(a, b) or BAD_INNER.search(inner) or not text_of(inner).strip():
                continue
            segs.append((a, b, kind, inner))
    segs.sort()
    # drop any segment nested inside another (e.g. a <p> inside an .expl caught twice)
    out, last_end = [], -1
    for sg in segs:
        if sg[0] >= last_end:
            out.append(sg)
            last_end = sg[1]
    return out


def text_of(h: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", h))


def protected(h: str) -> Counter:
    c = Counter()
    for m in re.finditer(r"<code[^>]*>(.*?)</code>", h, re.S):
        c["code:" + m.group(1)] += 1
    for m in re.finditer(r'href="([^"]*)"', h):
        c["href:" + m.group(1)] += 1
    for m in re.finditer(r"<a\b[^>]*>", h):
        c["atag:" + m.group(0)] += 1
    for m in re.finditer(r"<(sub|sup)>(.*?)</\1>", h, re.S):
        c[m.group(1) + ":" + m.group(2)] += 1
    for m in re.finditer(r'<span class="math">(.*?)</span>', h, re.S):
        c["math:" + m.group(1)] += 1
    for m in re.finditer(r'<span class="shp">.*?</span>', h, re.S):
        c["shp:" + m.group(0)] += 1
    t = re.sub(r"<code[^>]*>.*?</code>", " ", h, flags=re.S)
    # "bf16", "e4m3" are names, not numbers; a comma joins digits only as a thousands separator
    for m in re.finditer(r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?", text_of(t)):
        c["num:" + m.group(0)] += 1
    return c


def dump(path: str):
    s = open(path, encoding="utf-8").read()
    segs = segments(s)
    os.makedirs(WORK, exist_ok=True)
    tag = os.path.basename(path)[:4]
    out = os.path.join(WORK, tag + ".seg.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for i, (_, _, kind, inner) in enumerate(segs):
            f.write(f"@@ {i} {kind}\n{inner.strip()}\n")
    words = sum(len(text_of(x[3]).split()) for x in segs)
    print(f"{path}: {len(segs)} segments, {words} words -> {out}")
    return segs


ALLOW: dict = {}   # segment id -> facts the rewrite may drop on purpose ("@@ 0 !drop:70")


def parse_edits(p: str) -> dict:
    edits, cur, buf = {}, None, []
    ALLOW.clear()
    for line in open(p, encoding="utf-8").read().split("\n"):
        m = re.match(r"^@@ (\d+)\b(.*)$", line)
        if m:
            if cur is not None:
                edits[cur] = "\n".join(buf).strip()
            cur, buf = int(m.group(1)), []
            drops = re.findall(r"!drop:(\S+)", m.group(2))
            if drops:
                ALLOW[cur] = Counter("num:" + d for d in drops)
        else:
            buf.append(line)
    if cur is not None:
        edits[cur] = "\n".join(buf).strip()
    return edits


def apply(path: str, edits_path: str, write: bool = True) -> int:
    s = open(path, encoding="utf-8").read()
    segs = segments(s)
    edits = parse_edits(edits_path)
    problems = []
    for i, new in edits.items():
        if i >= len(segs):
            problems.append(f"segment {i}: no such segment")
            continue
        old = segs[i][3]
        po, pn = protected(old), protected(new)
        po = po - ALLOW.get(i, Counter())          # deliberate, declared removals of unsupported claims
        if po != pn:
            lost, added = po - pn, pn - po
            problems.append(f"segment {i}: facts changed  lost={dict(lost)}  added={dict(added)}")
        if re.search(r"—|–|&mdash;|&ndash;", text_of(re.sub(r"<code[^>]*>.*?</code>", "", new, flags=re.S))
                     if False else re.sub(r"<code[^>]*>.*?</code>", "", new, flags=re.S)):
            problems.append(f"segment {i}: still has a dash")
        if BAD_INNER.search(new):
            problems.append(f"segment {i}: new text contains block markup")
        if new.count("<") != new.count(">"):
            problems.append(f"segment {i}: unbalanced angle brackets")
    if problems:
        print("\n".join("  REFUSED " + p for p in problems))
        print(f"{len(problems)} problem(s); nothing written")
        return 1
    out = []
    pos = 0
    for i, (a, b, _, inner) in enumerate(segs):
        out.append(s[pos:a])
        if i in edits:
            lead = inner[:len(inner) - len(inner.lstrip())]
            trail = inner[len(inner.rstrip()):]
            out.append(lead + edits[i] + trail)
        else:
            out.append(inner)
        pos = b
    out.append(s[pos:])
    new_s = "".join(out)
    if write:
        open(path, "w", encoding="utf-8", newline="\n").write(new_s)
    print(f"{path}: {len(edits)} of {len(segs)} segments rewritten, every fact preserved")
    return 0


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "dump":
        dump(path)
    elif cmd == "apply":
        sys.exit(apply(path, sys.argv[3]))
    elif cmd == "check":
        sys.exit(apply(path, sys.argv[3], write=False))
