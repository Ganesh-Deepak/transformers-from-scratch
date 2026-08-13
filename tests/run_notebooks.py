"""
Execute every notebook's code cells in order and report failures.

    python tests/run_notebooks.py                # all notebooks
    python tests/run_notebooks.py 01 05          # only matching ones

This is a smoke test, not a full Jupyter run: cells are concatenated and exec'd
in a fresh namespace per notebook. It catches syntax errors, NameErrors, failed
asserts, and shape bugs -- which is what we care about.
"""

import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB_DIR = os.path.join(ROOT, "notebooks")
sys.path.insert(0, ROOT)


def run_notebook(path: str) -> tuple[bool, str]:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    ns: dict = {"__name__": "__nb__"}
    buf = io.StringIO()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue
        try:
            with redirect_stdout(buf):
                exec(compile(src, f"{os.path.basename(path)}::cell{i}", "exec"), ns)
        except Exception:
            return False, f"cell {i} failed:\n{traceback.format_exc()}"
    return True, buf.getvalue()


def main() -> int:
    filters = sys.argv[1:]
    names = sorted(n for n in os.listdir(NB_DIR) if n.endswith(".ipynb"))
    if filters:
        names = [n for n in names if any(f in n for f in filters)]
    if not names:
        print("no notebooks found")
        return 1

    fails = 0
    for name in names:
        ok, out = run_notebook(os.path.join(NB_DIR, name))
        if ok:
            lines = [l for l in out.splitlines() if l.strip()]
            print(f"  PASS  {name}  ({len(lines)} output lines)")
        else:
            fails += 1
            print(f"  FAIL  {name}")
            print("        " + out.replace("\n", "\n        ")[:1600])
    print(f"\n{len(names) - fails} passed, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
