"""Write docs/HUMANIZE-REPORT.md: before (work/chNN.orig.html) and after (course/) for every chapter done so far."""
import glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from scan import score  # noqa: E402

rows = []
for orig in sorted(glob.glob(os.path.join(HERE, "work", "ch*.orig.html"))):
    tag = os.path.basename(orig)[:4]
    cur = glob.glob(os.path.join(ROOT, "course", tag + "-*.html"))[0]
    a, b = score(orig), score(cur)
    rows.append((os.path.basename(cur), a, b))
out = ["# Humanizing the Transformers prose: before and after", "",
       "Rules: github.com/blader/humanizer (SKILL.md). Each chapter went through draft, check and final;",
       "`tests/humanize/seg.py` refuses any rewrite that changes a number, inline code, link, maths span or",
       "shape badge, so every fact is preserved by construction. Scores from `tests/humanize/scan.py` over the",
       "prose only (code, tables, figures excluded). Tells = AI-writing patterns per 1,000 words. Reading ease =",
       "Flesch (higher is easier; 60-70 is plain English).", "",
       "| chapter | tells/1k before | after | bold/1k before | after | words/sentence before | after | reading ease before | after |",
       "|---|---|---|---|---|---|---|---|---|"]
for f, a, b in rows:
    out.append(f"| {f} | {a['per1k']:.1f} | {b['per1k']:.1f} | {a['bold_per1k']:.1f} | {b['bold_per1k']:.1f} | "
               f"{a['sent_len']:.1f} | {b['sent_len']:.1f} | {a['ease']:.1f} | {b['ease']:.1f} |")
open(os.path.join(ROOT, "docs", "HUMANIZE-REPORT.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("\n".join(out[-len(rows):]))
