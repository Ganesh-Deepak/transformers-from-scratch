# Prompt: author a new course in the "Field Manual" system

Paste everything below the line into a fresh session, after filling in the three
`<<< >>>` placeholders at the top. It is self-contained. For a large course, run
one session per Part in parallel (see §10).

---

You are authoring a new interactive technical course. An existing course —
**Transformers From Scratch** — already defines the design system, the
pedagogical standard, the tooling, the test suite and the platform contract that
lets all courses act as one library. Your job is to produce a new course **in
that exact system**, not to design a new one.

    REPO      https://github.com/Ganesh-Deepak/transformers-from-scratch
    LIVE      https://ganesh-deepak.github.io/transformers-from-scratch/

    <<< COURSE TITLE >>>
    <<< COURSE SLUG, e.g. "evals"; the repo will be ganesh-deepak/<repo-name> >>>
    <<< ONE-PARAGRAPH BRIEF from docs/ROADMAP.md: who it is for, what they can do
        at the end, what it deliberately does not cover >>>

---

## 0. First action, before you write anything

**The outline already exists.** Every planned course has a reviewed curriculum in the
hub repository (`Ganesh-Deepak/ganesh-deepak.github.io`): `curricula/<slug>.txt` is the
source, `docs/curricula/<slug>.md` the readable form, and the course repository already
holds a generated outline site (one page per chapter with its sections, "Builds on"
links and what the reader builds). Write the chapters against that outline. If the
argument needs a change (a chapter split, merged or moved), change `curricula/<slug>.txt`
first and rerun `tools/build.py` in the hub, so the registry, search, prerequisite map and
every cross-course link stay correct. When a chapter's prose is written it replaces its
outline page; when the whole course is written, set `status: live` in the curriculum and
the build stops generating its pages.

Clone the reference course and read these files end to end. They are the
specification; this prompt is a summary of them.

    docs/PLATFORM.md            how every course plugs into one library (§9 below)
    docs/CONTENT-AUDIT.md       the grading standard every chapter is held to
    course/assets/style.css     the whole design system, heavily commented
    course/assets/course.js     layout, theming, study console, quizzes, sidebar
    course/assets/axes.js       the semantic highlighter and the shape inspector
    course/assets/chapters.js   the outline manifest: single source of truth
    course/ch20-moe.html        the best finished chapter (content standard)
    course/ch06-mha.html        the best widget-driven chapter
    tfs/moe.py + notebooks/20_moe.ipynb + tests/test_tfs.py (Ch 20 section)
                                how a chapter's numbers are backed by code

Then **copy `course/assets/` verbatim** (`style.css`, `course.js`, `axes.js`,
`viz.js`, `chapters.js` as a template, `fonts/`) and **the whole `tests/`
toolkit** (`check_ui.js`, `check_responsive.js`, `check_interaction.js`,
`shoot.js`, `run_notebooks.py`, `package.json`).

**Do not restyle. Do not "improve" the palette. Do not swap the fonts.** The
visual system is locked across every course on the platform. Your creative
budget goes into content, figures, widgets and exercises. If something in the
design genuinely must change, say so and ask first. Never change it silently.

---

## 1. The look, so you can tell when it is wrong

A 1960s aerospace field manual that happens to be interactive: printed in two
inks, a reading sheet laid on a kraft desk. Bold, not restrained. If your page
looks like a generic docs site, you have drifted.

| | light | dark |
|---|---|---|
| Desk (around the page) | kraft `#E7D8BC` under a halftone dot screen | `#1A1513` with a faint dot screen |
| **Reading sheet** | solid paper `#F8F0E0`, 2px oxblood edge, hard 9px shadow | warm charcoal `#27211E`, visible edge |
| Cards, code | cream `#FFFBF3` on **hard `4px 4px 0` shadows, zero blur** | `#342C27` |
| Structure | oxblood `#5A241F` masthead, borders, rules | terracotta stamp `#B34B33` |
| Display / body / mono | Space Grotesk 700 · Literata · JetBrains Mono (self-hosted) | same |

Rules that carry the look, in priority order:

1. **No texture under letterforms.** The halftone lives on the desk only; every
   line of text sits on the solid sheet. (Dots behind small type made the light
   theme unreadable once.)
2. **No soft shadows.** Every shadow is `Npx Npx 0`.
3. **2px borders** on every card-like object; oxblood does the structural work,
   so the violet accent is only ever a link and interaction colour.
4. Labels are Space Grotesk, uppercase, 700, tracked `.13`–`.16em`.
5. Dark mode is **layered, never one black field**: desk, sheet and cards must be
   visibly different surfaces, and secondary text stays bright (ink-3 ≥ 7:1 on
   the sheet). A near-black ground with the sheet equal to it was rejected as
   "too dark to read anything".

---

## 2. Layout — locked, and enforced by tests

- **The sheet** is `main`, as wide as the space beside the sidebar allows.
- **Prose is 75% of that width**, centred, never narrower than 37rem or the
  screen, capped at 90rem. Body type grows with it (17 → 21px). The reader asked
  for a wide column; do not "fix" it back to a narrow measure.
- **One body size for everything you read inside `main`**: `--fs`. Components
  scale from it, never from fixed `rem`: h2 1.5×, h3 1.22×, lead 1.1×, callout
  text .95×, tables .88×, quiz options .92×, code .76×, maths 1.08×. A heading is
  always larger than the text under it.
- **Three kinds of block**, one rule each:
  1. *Full width* — widgets (`data-viz`), figures, panels.
  2. *Fit content* — code, tables (`.tw`), maths blocks: as wide as what they
     hold, never narrower than the text column, never wider than the sheet,
     centred on the text column.
  3. *Reading boxes* — quizzes, exercises, worked examples, the study console:
     the text column plus at most 4.5rem a side.
- **Margin notes are inline asides** (`.mn`), not a side rail: at 75% prose width
  there is no room for one.
- The **sidebar** groups chapters by Part: each Part is a solid band in its own
  ink (`--part-0..7`), folds away on click, and the current chapter is a raised
  card whose section list folds too. Inks are navigation-only.
- The **shape inspector** (Phase 3 of `axes.js`) is a docked right rail on wide
  screens (it reserves its column; text reflows) and a bottom sheet on phones.
  It appears only on pages that contain semantic badges.
- No page may scroll sideways from **320px** up.

You do not author any of this; the vendored CSS and JS do it. You only need to
avoid fighting it: no inline layout styles, no wrappers that change widths.

---

## 3. Component vocabulary — use these, invent nothing

**Zero `<style>` blocks.** Inline `style=` only for local spacing
(`style="margin-bottom:0"` on a callout's last child, `style="margin:.4rem 0"` on
a `<pre>` inside a worked example). Never colour, type, borders or layout.

### Page skeleton

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ch 6 — Multi-Head Attention: The Reshape Dance</title>
<meta name="color-scheme" content="light dark">
<script>(function(){try{var t=localStorage.getItem("tfs-theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t);if(localStorage.getItem("tfs-side")==="collapsed"&&matchMedia("(min-width:82.001rem)").matches)document.documentElement.classList.add("pre-collapsed");}catch(e){}})();</script>
<link rel="stylesheet" href="assets/style.css?v=1">
</head>
<body>
<main>
  <h1>6 · Multi-Head Attention: The Reshape Dance</h1>
  <p class="lead">One paragraph: what this chapter does, and what makes it hard.</p>
  <div class="mn orient"><span class="k">Builds on</span>…<div class="sep"></div><span class="k">Leads to</span>…<div class="sep"></div><span class="k">Why this chapter</span>…</div>
  …
</main>
<script src="assets/chapters.js?v=1"></script>
<script src="assets/course.js?v=1"></script>
<script src="assets/viz.js?v=1"></script>
<script src="assets/axes.js?v=1"></script>
<script>initChapter("06");</script>
</body>
</html>
```

Keep the head script byte-identical (it applies the stored theme before first
paint; the theme key is shared across the whole platform). Bump every `?v=N`
together whenever an asset changes, or browsers serve stale CSS.

### Callouts

```html
<div class="box intuition"><span class="t">The core trade</span><p>…</p></div>
```

| class | for |
|---|---|
| `box intuition` | the mental model that makes the rest obvious (also "the whole chapter in N sentences") |
| `box warn` | a thing that will bite you later |
| `box gotcha` | a thing that is silently wrong and raises no error |
| `box math-note` | the one line of maths that matters, explained |
| `box sota` | what the frontier does now (date it) |

### Quiz, exercise, worked example, table, maths, code

```html
<div class="quiz" data-title="Checkpoint 1 — routing and balance">
  <div class="qitem" data-answer="1">
    <div class="qq">Question, with <code>code</code> where it helps.</div>
    <div class="opts"><div class="opt">…</div><div class="opt">…</div></div>
    <div class="expl"><b>Answer.</b> Why it is right, and why each distractor is tempting.</div>
  </div>
</div>

<div class="exercise">
  <div class="eh"><span class="n">6.1</span><span class="ti">Prove your heads are independent</span><span class="lvl">essential</span></div>
  <div class="eb"><p>…</p><details class="reveal"><summary>Solution</summary><div>…</div></details></div>
</div>

<div class="example"><span class="t">Worked example — …</span>
  <div class="step"><div class="i">1</div><div>…</div></div>
</div>

<div class="tw"><table><thead>…</thead><tbody>…</tbody></table></div>
<div class="mathblock">…<span class="lbl">what it says in words</span></div>
<pre><code>…python, highlighted automatically…</code></pre>
<pre><code class="nohl">…diagrams, output…</code></pre>
```

`data-answer` is **0-based**. Every table goes inside `.tw`. **Never write raw
LaTeX** (`d_{head}`, `\sqrt{x}`): there is no renderer, readers see it literally,
and `check_ui.js` fails on it. Use `d<sub>head</sub>`, `√`, `×`, `·`.

### Figures

Inline SVG; every colour is a CSS variable with a fallback
(`fill="var(--ink-3,#664436)"`) so figures re-theme. Never a bare hex.

---

## 4. Colour is semantic — spend it on meaning only

The five `--dim-*` hues **encode one categorical meaning for the whole course**
(in Transformers: which tensor axis — batch, time, feature, head, head-dim). If
your subject has a natural categorical dimension (memory tier, pipeline stage,
agent role, failure class), map it onto `--dim-1..5` and keep it for every page.
If it does not, use neutrals and leave the hues alone. Never two meanings.

The hues are validated perceptually, not by hue angle: OKLab distance ×100 under
the Machado-2009 protan and deutan simulations must be **≥ 8 for every pair**
(colour-blind readers) and **≥ 15** with normal vision, and each hue must stay
clear of the accent and the stamp. `check_ui.js` enforces this in both themes.
A sixth category gets a non-hue treatment (dotted underline, outline), never a
near-miss colour.

Widgets that do not draw your semantic categories use neutral, accent and stamp
colours, never the dim hues. The `--part-*` inks are for navigation only.

If your course has shape-like expressions, reuse the highlighter: register your
vocabulary in `TFS_AXIS_SYMBOLS` / `TFS_AXES` and mark spans as
`<span class="shp">(<b data-axis="batch">B</b>, …)</span>`. It is never
colour-only (dimming, underline, ring, a "pinned" label). Never bake this markup
into `<pre>`; `tfsAnnotate()` annotates code at runtime.

---

## 5. The content standard — what an "A" chapter is

Every chapter is graded in `docs/CONTENT-AUDIT.md` (create one for your course).
**A** = worth the read as a book for a working engineer; **B** = solid but missing
listed items; **C** = too thin for its importance. Ship only A chapters.

An A chapter has, in order:

1. `<h1>N · Title</h1>`, a `.lead` naming what is genuinely hard, and the
   `.mn.orient` block (*Builds on / Leads to / Why this chapter*; "Builds on" only
   points backwards, cross-course links use the `xref` form in §9).
2. A **"whole chapter in N sentences"** `box intuition` near the top.
3. `<h2>N.1 · …</h2>` sections in dependency order, each with at least one worked
   example that shows every number, and a widget or drawn figure where it helps.
4. **At least 3 quiz questions per major section**, explanations that address the
   distractors, spread through the chapter, not all at the end.
5. **5–7 exercises**, including at least one cost/systems estimate, one
   implementation, and one "predict, then measure". Answers in `details.reveal`.
6. A "mental model to keep" section, then Exercises, then
   `<h2>Sources &amp; further reading</h2>` (`ul.tiny`: primary papers first, then
   the best secondary explainers, then the companion notebook and tests).

Non-negotiables of method:

- **Every number in the text is computed by code in the repository and asserted.**
  Reference code lives in a package (`yourpkg/…py`) with shapes annotated on every
  line; a test asserts each claim; a companion notebook recomputes every worked
  example and table with an `assert` per cell. A chapter whose notebook does not
  exist yet marks the reference `<code class="planned">…</code>`; the integrity
  check fails on any other reference to a missing file.
- **Derive, never assert.** If a constant appears, show where it comes from.
- **Facts must be verifiable.** Cite the paper, the vendor spec or the model
  config. If you cannot verify it, do not state it. Date anything about the
  frontier, in a `box sota` or a dated note.
- **Annotate every shape and name every axis** at each step.
- **Close the loop on anything generative.** A healthy loss curve proved nothing
  once: the buggy run had the *lower* loss. Show the output, or the assertion that
  would have caught the bug.
- **Order by dependency, not topic.** Each chapter uses what the last built.

### The outline manifest

```js
{ id: "06", f: "ch06-mha.html", t: "Multi-Head Attention: The Reshape Dance",
  s: "One-line subtitle", tag: "70 min" }
```

Chapters are grouped into `PARTS` (each Part gets the next `--part-N` ink). `id`
matches the filename digits and runs without gaps (it may start at `00`).
Unwritten chapters are `wip: true` (listed but inert). **Renumbering is
error-prone**: when you insert a chapter, rename files, update `chapters.js`, and
fix every `href` and every visible "Ch NN" label by deriving labels from the
hrefs, never by regex over bare numbers (`Ch&nbsp;15` and "Ch 6, 13" slip past
naive patterns).

---

## 6. Interactivity and state

- At least one `data-viz` widget in most chapters. Widgets follow `viz.js`
  conventions: read colours through `readTheme`, redraw on theme change while
  preserving their controls' state, and call `tfsEnhance()` after drawing any
  semantic badges.
- Every stateful control gets a scenario in `check_interaction.js` (fake clock;
  scenarios may return promises when an observer must run). "It saved the value
  but never restarted the timer" passed every other suite once.
- `prefers-reduced-motion` means no transitions at all (`transition: none`, not a
  tiny duration).

---

## 7. Definition of done

```bash
cd tests && npm install jsdom
node check_ui.js            # structure, WCAG both themes, CVD-validated hues,
                            # content integrity (files, links, Builds-on, no LaTeX)
node check_responsive.js    # no sideways scroll from 320px, real Chrome
node check_interaction.js --all   # every control does what it says, every chapter
node shoot.js               # every page × 8 viewports × 2 themes: screenshots plus
                            # layout audit (overflow, centring, heading hierarchy,
                            # box-wider-than-content, misalignment, clipped text and
                            # SVG labels, tiny type, touch targets, fonts, JS errors)
python <your python tests> && python run_notebooks.py
```

All must pass with zero problems. Then look: `node shoot.js --full` writes contact
sheets of every page; read them in both themes. The suites cannot see taste, and
three of the worst bugs in the reference course (prose pinned left in a wide sheet,
headings smaller than body text, code boxes 2× wider than their code) were found by
looking, then turned into rules.

Check by hand too: keyboard only (tab to everything, Esc clears), greyscale (every
state distinguishable), JS disabled (prose and colours correct), both themes on a
real screen.

---

## 8. Deploy

GitHub Pages from `main`, repository root. After each push, confirm the build
finished (`gh api repos/<owner>/<repo>/pages/builds/latest`) and that the live
pages serve the new `?v=`.

---

## 9. The platform contract (docs/PLATFORM.md)

Every course is one entry in a single library at `ganesh-deepak.github.io`. All
courses share one origin, so they share settings and can read each other's
progress. Build yours ready for that from the first commit:

- **Slug and storage.** Your progress key is `<slug>-progress-v1` (set `KEY` in
  `course.js`). Do not rename shared keys (`tfs-theme`, `tfs-side*`,
  `tfs-inspector`): a reader's theme and sidebar follow them across courses.
- **Registry.** Vendor the hub's `platform/courses.js` as
  `course/assets/courses.js` and add your course's entry (slug, title, url,
  store key, chapter count, ink, status).
- **Cross-course links** use one form, so the prerequisite map and the
  "already done" marks work:

  ```html
  <a class="xref" data-course="transformers" data-ch="15"
     href="/transformers-from-scratch/ch15-kvcache.html">Transformers&nbsp;Ch&nbsp;15</a>
  ```

  Put them in `.mn.orient` "Builds on" whenever a chapter depends on another
  course. Prefer linking to the chapter that teaches a thing over re-teaching it.
- **Design-system version.** Record which version of the shared assets you
  vendored; `check_platform.js` reports drift.

---

## 10. Working in parallel

For a course of 15+ chapters, split authoring by Part across parallel sessions,
each on its own branch, with disjoint files:

- only one session edits `chapters.js`, `index.html` and the shared assets;
- each session puts its tests in its own file (`tests/test_<part>.py`);
- nobody bumps `?v=` (the merging session does it once);
- each session updates only its chapters' entries in `CONTENT-AUDIT.md`.

Merge, run every suite, bump the cache version, deploy, verify live.

Commit in coherent chunks with messages that explain **why**, not what.
