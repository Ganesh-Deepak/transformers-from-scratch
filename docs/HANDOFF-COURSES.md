# Handoff: building the ten new courses

Last updated 2026-11-15. Read this first when you resume the course build. It records where
everything is, how a chapter is produced, and exactly where the work stopped.

## Where things are

| What | Where |
|---|---|
| Transformers course (live, 28 chapters) | `D:\ai-llms` → github.com/Ganesh-Deepak/transformers-from-scratch |
| The hub (library page, registry, curricula, build tools) | `D:\ai-llms\sites\ganesh-deepak.github.io` → github.com/Ganesh-Deepak/Ganesh-Deepak.github.io |
| The ten course repos | `D:\ai-llms\sites\<repo>` (llm-evaluation, post-training, gpu-programming, inference-engineering, retrieval, agents, production-ai, ai-security, ml-maths, multimodal), each its own git repo and Pages site |
| Curriculum of each course (source of truth) | hub `curricula/<slug>.txt`; readable copy in hub `docs/curricula/<slug>.md` |
| Authoring rules | `docs/NEW-COURSE-PROMPT.md` (content standard, components, layout), `docs/PLATFORM.md` (how courses connect) |
| Shared test suites | `D:\ai-llms\tests` (`check_ui.js`, `check_responsive.js`, `check_interaction.js`, `shoot.js`), all accept `--root <course dir>` |

`D:\ai-llms\sites\` is ignored by the Transformers repo; every folder inside it is its own repository.

## Status

| # | Course | Status |
|---|---|---|
| 0 | Transformers | live, all chapters graded A; prose being simplified (see "Humanizing" below) |
| 1 | Evaluation (`llm-evaluation`) | **Ch 1–4 written, tested, pushed.** Ch 5–6 code written (`evalkit/stats.py`), prose not started. Ch 7–18 are outline pages. |
| 2–10 | all others | outline sites only (every chapter page lists its planned sections) |

### Where Evaluation stopped

`evalkit/stats.py` is written and committed (bootstrap percentile/BCa, clustered SE, design
effect, cluster bootstrap, McNemar exact/chi², paired/unpaired bootstrap, permutation test,
`n_paired`, `n_unpaired`, `mde_paired`, `simulate_pair`). Numbers already computed for Ch 5
(not yet in a test):

- 200 items at true 72%, seed 3: mean 0.720, SE 0.0318, normal [0.658, 0.782], percentile
  bootstrap [0.655, 0.780], BCa [0.650, 0.775], Wilson [0.654, 0.778].
- Skewed per-item cost (lognormal(-5, 1.2), 60 items, seed 4): mean 0.0191, percentile
  [0.01236, 0.02735], BCa [0.01331, 0.03001].
- Design effect 1 + (m−1)ρ: (5, 0.1) 1.4 → 714 effective of 1,000; (5, 0.3) 2.2 → 455;
  (10, 0.3) 3.7 → 270; (20, 0.5) 10.5 → 95; (5, 1.0) 5 → 200.
- One clustered sample (200 prompts × 5 samples, p 0.7, ρ 0.3, seed 1): mean 0.681, naive SE
  0.0147, clustered SE 0.0225, cluster bootstrap [0.634, 0.725].
- Coverage of a 95% interval over 1,000 trials: (200×5, ρ 0.3) naive 83.3%, clustered 95.1%;
  (100×10, ρ 0.5) naive 60.2%, clustered 94.1%.

Next: write `ch05-uncertainty.html` from those numbers, add `test_ch05_*` and notebook
`05_uncertainty`, then Ch 6 (McNemar, paired vs unpaired, power), Ch 7 (`rank.py`: Holm, BH,
Bradley–Terry, Elo, IRT 2PL), Ch 8 (`variance.py`), Part III (`judges.py`: Rogan–Gladen, PPI++),
Part IV (`systems.py`), Part V (`online.py`: CUPED, mSPRT), Ch 18 capstone.

## How one chapter is produced (the loop that worked)

1. **Code first.** Add the functions to the course package (`evalkit/…`), shapes or units noted.
2. **Compute every number** the chapter will quote by running the code. Never write a number
   from memory or mental arithmetic: four of the first five Evaluation chapters had a hand
   calculation that the code proved wrong.
3. **Write the tests** asserting those numbers (`tests/test_evalkit.py`, one `test_chNN_*` per
   section group). Run them.
4. **Write the chapter** (`course/chNN-slug.html`) with the exact numbers. Skeleton, components and
   order: `NEW-COURSE-PROMPT.md` §3 and §5. Each chapter: lead, orient note (cross-course links in
   `a.xref` form), "whole chapter in N sentences", numbered sections with worked examples, ~3 quiz
   questions per major section (`data-answer` is 0-based), 5–6 exercises with solutions, "mental
   model to keep", sources. Scripts at the bottom: `chapters.js`, `course.js`, `viz.js`, `widgets.js`.
5. **Notebook**: add a function to `tests/nb_chapters.py`, then `python tests/build_notebooks.py NN`
   and `python tests/run_notebooks.py NN`.
6. **Re-check any claim the simulations touch**: change the prose to the measured value, not the
   other way round.
7. **Widgets**: course-specific, in `course/assets/widgets.js`, registered into the shared `VIZ`
   table (they reuse viz.js helpers and theme handling). Colours: the five `--dim-*` hues mean the
   course's roles (Evaluation: system under test, baseline, human labels, judge, uncertainty).
8. **Check**: `node check_ui.js --root ../sites/<repo>/course --pkg <pkg>` from `D:\ai-llms\tests`
   (outline pages fail only "quiz markers built" until written; that is expected).
9. **Commit per Part, push.** At the end of a course: `check_responsive --root`, `check_interaction
   --root` (runs the per-chapter sweep), `shoot.js --root … --out D:/…` (screenshots on D:, never C:),
   set `status: live` in the hub curriculum, `python tools/build.py`, `python tools/check_platform.py`,
   commit and push the hub, verify live.

The hub build (`tools/build.py`) never overwrites a chapter page that lacks the outline marker
comment, and never deletes course-owned assets such as `widgets.js`.

## Gotchas found on this machine

- **C: is nearly full.** Screenshot runs go to `D:` (`--out D:/…`). The suites now delete their
  Chrome profiles on exit; if a run is killed, delete `%TEMP%\tfs-*` folders.
- **Escapes in Python heredocs**: writing JS or regex through `python - <<'EOF'` loses or doubles
  backslashes. Write JS with the Write/Edit tools, or check the result with grep.
- **`print` of emoji/checkmarks crashes** plain `python` on this console: reconfigure stdout to UTF-8.
- `git config core.autocrlf false` is set in every `sites/*` repo to keep LF endings.
- Tests in the Transformers repo must stay green after any shared-asset change: run
  `check_ui`, `check_responsive`, `check_interaction --all`, `shoot.js` there too, then
  `python tools/build.py --sync-assets` in the hub and `check_platform.py`.

## Humanizing the Transformers prose

The user asked for the Transformers prose to read less like AI and more simply, following
github.com/blader/humanizer (`SKILL.md`: 26 patterns; strongest are "not X but Y", one-line
closers, sayings that sound deep, staged run-ups, arguing with no one, forced triads, dashes,
stock AI words, bold as decoration). Tooling and progress for that pass are in
`tests/humanize/` and `docs/HUMANIZE-REPORT.md` (one row per finished chapter).

Per chapter (from `D:\ai-llms`):

1. `python tests/humanize/seg.py dump course/chNN-x.html` writes `tests/humanize/work/chNN.seg.txt`
   (every prose segment, numbered). `python tests/humanize/scan.py --detail course/chNN-x.html` lists tells.
2. Write `tests/humanize/work/chNN.new.txt` with `@@ <id>` blocks holding the new inner HTML of the
   segments that change (draft). Plain words, shorter sentences, no dashes, bold only on real key terms,
   no "not X but Y", no closers or staged lines. Keep every number, `<code>`, link, `<sub>/<sup>`, span.math.
   To delete an unsupported claim's number on purpose: `@@ 0 !drop:70`.
3. `python tests/humanize/seg.py check ...chNN-x.html tests/humanize/work/chNN.new.txt`; fix every REFUSED line.
4. `cp course/chNN-x.html tests/humanize/work/chNN.orig.html`, then `seg.py apply ...`, then `scan.py --detail`
   (remaining hits are usually titles such as "Let's build…", or the literal word "underscore").
5. Every few chapters: `node tests/check_ui.js` (from tests/), `python tests/humanize/report.py`, commit.

Done so far: see the rows in `docs/HUMANIZE-REPORT.md`.
