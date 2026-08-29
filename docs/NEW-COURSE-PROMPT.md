# Prompt: author a new course in the "Field Manual" system

Paste everything below the line into a fresh session, after filling in the two
`<<< >>>` placeholders at the top. It is self-contained.

---

You are authoring a new interactive technical course. An existing course —
**Transformers From Scratch** — already defines the design system, the
pedagogical structure, the tooling and the test suite. Your job is to produce a
new course **in that exact system**, not to design a new one.

    REPO      https://github.com/Ganesh-Deepak/transformers-from-scratch
    LIVE      https://ganesh-deepak.github.io/transformers-from-scratch/

    <<< COURSE TITLE >>>
    <<< ONE-PARAGRAPH BRIEF: who it is for, what they can do at the end,
        what it deliberately does not cover >>>

---

## 0. First action, before you write anything

Fetch the reference course and read these five files end to end. They are the
specification; this prompt is only a summary of them.

    course/assets/style.css     the whole design system, heavily commented
    course/assets/course.js     layout scaffold, theming, quiz engine, highlighter
    course/assets/axes.js       the tensor-axis highlighting system
    course/assets/chapters.js   the outline manifest — single source of truth
    course/ch06-mha.html        the best worked example of a finished chapter

Then **copy `course/assets/` into your new course verbatim** — `style.css`,
`course.js`, `axes.js`, `viz.js`, and the whole `fonts/` directory. Copy
`tests/check_ui.js` and `tests/check_responsive.js` too.

**Do not restyle. Do not "improve" the palette. Do not swap the fonts.** The
visual system is locked. Your entire creative budget goes into content, figures,
widgets and exercises. If you believe something in the design genuinely must
change, say so and ask first — do not change it silently.

---

## 1. The look, so you can tell when it is wrong

A 1960s aerospace field manual that happens to be interactive: printed on kraft
stock in two inks. If your page looks like a generic docs site, you have drifted.

| | |
|---|---|
| Ground | kraft `#E7D8BC` under a halftone dot screen (1px dots on a 5px grid) |
| Structure | oxblood `#5A241F` — masthead, every 2px border, every rule |
| Cards | cream `#FBF4E4`, stamped on **hard `4px 4px 0` shadows, zero blur** |
| Dark | near-neutral `#14100F` ground + terracotta `#B34B33` stamp |
| Display | Space Grotesk 700, tracking −0.025 to −0.035em |
| Body | Literata (serif, real italics) |
| Mono | JetBrains Mono — every shape annotation, every code block |

Rules that carry the whole look, in priority order:

1. **No soft shadows anywhere.** Every shadow is `Npx Npx 0` with zero blur.
2. **2px borders**, not hairlines, on every card-like object.
3. The **halftone dot screen** is on `body` and on `.sidebar`. It is what turns
   a flat colour into printed stock.
4. **Oxblood does the structural work**, which frees the violet accent to be
   nothing but a link colour.
5. Labels are Space Grotesk, uppercase, 700, letter-spacing `.13`–`.16em`.
6. Buttons collapse their shadow on `:active` and translate by the same offset,
   so they physically press into the paper.

**Dark mode is not a token swap.** The first attempt at it failed exactly that
way: every layer went brown, the masthead landed at 1.10:1 against the paper and
was invisible. On a dark ground the saturation must live in the *structure*, not
the field — near-achromatic ground (chroma ~5), vivid stamp (chroma ~128).
Surface separation in dark is tuned by **L\***, not contrast ratio: near black
the `+0.05` in the WCAG formula swamps small steps, so 1.09 and 1.25 look
identical. Cards must lift by ΔL\* ≈ 8.5.

---

## 2. Component vocabulary — use these, invent nothing

All styling already exists. Write semantic HTML with these classes.

**Zero `<style>` blocks** — the reference course has none across 27 pages, which
is why a full redesign is a one-file change. Keep it that way.

Inline `style=` is permitted **only for local spacing** (`style="margin-bottom:0"`
on the last child of a callout, `style="margin:.4rem 0"` on a `<pre>` inside a
worked example — those two account for almost all 406 uses in the reference).
**Never** put colour, typography, borders or layout in an inline style. If you
reach for one, the class you actually want probably already exists.

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
<link rel="stylesheet" href="assets/style.css?v=10">
</head>
<body>
<main>
  <h1>6 · Multi-Head Attention: The Reshape Dance</h1>
  <p class="lead">One paragraph. What this chapter does, and what makes it hard.</p>
  ...
</main>
<script src="assets/chapters.js?v=10"></script>
<script src="assets/course.js?v=10"></script>
<script src="assets/viz.js?v=10"></script>
<script src="assets/axes.js?v=10"></script>
<script>initChapter("06");</script>
</body>
</html>
```

The inline `<script>` in `<head>` is an anti-FOUC guard — it applies the stored
theme before first paint. Keep it byte-identical. `course.js` builds the topbar,
sidebar, reading-progress bar, chapter nav, copy buttons and quiz engine at
runtime; you do not author any of that.

### Callouts

```html
<div class="box intuition"><span class="t">The core trade</span><p>…</p></div>
```

Six variants, each with a settled meaning. Use them for what they mean:

| class | for |
|---|---|
| `box intuition` | the mental model that makes the rest obvious |
| `box warn` | a thing that will bite you later |
| `box gotcha` | a thing that is silently wrong and raises no error |
| `box math-note` | the one line of maths that matters, explained |
| `box history` | why it ended up this way |
| `box sota` | what the frontier actually does now |

### Margin notes — the signature layout element

```html
<div class="mn"><span class="k">Why it hides</span>Both merges give the same shape. Nothing raises.</div>
<div class="mn orient"><span class="k">Builds on</span><a href="ch01-tensors.html">Ch&nbsp;01</a> &mdash; view and transpose<div class="sep"></div><span class="k">Leads to</span>…</div>
```

Absolutely positioned beside the prose column with **no `top`**, so each note
lines up with the paragraph it follows. Below 66rem they fold back inline. Put
an `.mn.orient` block directly after the `.lead` of every chapter with
*Builds on / Leads to / Why this chapter*.

### Quiz

```html
<div class="quiz" data-title="Checkpoint 1 — the dance">
  <div class="qitem" data-answer="1">
    <div class="qq">Question text, with <code>code</code> where it helps.</div>
    <div class="opts">
      <div class="opt">(4, 256, 12, 64)</div>
      <div class="opt">(4, 12, 256, 64)</div>
    </div>
    <div class="expl"><b>(4, 12, 256, 64).</b> Why — and why each wrong option is tempting.</div>
  </div>
</div>
```

`data-answer` is a **0-based index**. The engine adds A/B/C/D markers, scoring
and reveal. The explanation must explain the *distractors*, not just restate the
answer — being wrong and noticing is the mechanism that makes it stick.

### Exercise

```html
<div class="exercise">
  <div class="eh"><span class="n">6.1</span><span class="ti">Prove your heads are independent</span><span class="lvl">essential</span></div>
  <div class="eb">
    <p>What to build, and the assertion that proves it.</p>
    <pre><code>def mha_loop(x, W_q, W_k, W_v, W_o, nh):
    ...</code></pre>
    <details class="reveal"><summary>Solution</summary><div>…</div></details>
  </div>
</div>
```

`.n` is `chapter.index`. `.lvl` is a short free-form label, not a fixed enum —
the reference uses `essential`, `code`, `thinking`, `hard`, `warm-up`,
`exploration`, `applied`, and occasionally a time estimate (`~1 hr`). Pick a
small vocabulary for your course and stay consistent within it.

### Worked example, tables, figures, maths

```html
<div class="example"><span class="t">Worked example — …</span>
  <div class="step"><div class="i">1</div><div>…</div></div>
</div>

<div class="tw"><table><thead>…</thead><tbody>…</tbody></table></div>

<figure><svg viewBox="0 0 660 150" xmlns="http://www.w3.org/2000/svg">…</svg>
  <figcaption>…</figcaption></figure>

<div class="mathblock">…<span class="lbl">what it says in words</span></div>
```

`.tw` is the scroll wrapper — **every table must be wrapped in it** or it will
overflow the page on mobile.

### Code

Plain `<pre><code>…</code></pre>`. Python is syntax-highlighted automatically.
Add `class="nohl"` on the `<code>` to opt out (use for ASCII diagrams and
non-Python output). Copy buttons are added for you.

---

## 3. SVG figures — the one place you must be careful

Inline SVG only. **Every colour must be a CSS variable with a fallback**, so
figures re-theme with the page:

```html
<text fill="var(--ink-3,#775446)">label</text>
<rect fill="var(--surface,#FBF4E4)" stroke="var(--dim-2,#1B4FA8)" stroke-width="1.3"/>
```

Never hardcode a bare hex. The reference course had three inline maths colours
picked for a dark background that landed at **1.4:1** on cream when the theme
changed — invisible. Use the tokens and this cannot happen.

Available: `--ink`, `--ink-2/3/4`, `--paper`, `--surface`, `--surface-2`,
`--line`, `--line-2`, `--accent`, `--dim-1..5`, `--ok`, `--warn`, `--stop`,
`--stamp`, `--stamp-ink`.

---

## 4. Colour is semantic — do not spend it on decoration

The four/five `--dim-*` hues **encode meaning**, in this course's case which
tensor axis you are looking at. Boldness is spent there and nowhere else. If
your subject has a natural categorical dimension (pipeline stage, memory tier,
agent role, failure class), map it onto `--dim-1..5` and be consistent for the
whole course. If it does not, use neutrals and leave the dim hues alone.

**Five is the ceiling.** The hues sit at 20° / 90° / 175° / 218° / 324°, with
the violet accent at 276°, `--warn` at 55° and `--stop` at 5°. Every remaining
gap is under 45°. A sixth category must get a **non-hue** treatment (dotted
underline, outline, pattern) — a near-miss colour is worse than no colour.
`check_ui.js` enforces ≥30° separation.

### The axis highlighter (`axes.js`)

If your course has anything shaped like a tensor shape, reuse it. Register your
vocabulary in `TFS_AXIS_SYMBOLS`, mark up spans as:

```html
<span class="shp">(<b data-axis="batch">B</b>, <b data-axis="time">T</b>, <b data-axis="feat">C</b>)</span>
```

Hover, tap or keyboard-select any axis and every occurrence on the page lights
up. Four channels move together so it is **never colour-only**: everything else
dims to `.28`, the active one gains a 2px underline and a tinted background, and
pinned adds a solid ring.

Two hard-won constraints, both of which cost real debugging time:

- **Never bake this markup into `<pre>`.** `course.js` highlights code with
  `code.innerHTML = highlightPython(code.textContent)`, which silently strips
  it. Code blocks are annotated at runtime by `tfsAnnotate()` *after*
  highlighting, walking text nodes so the syntax spans survive.
- **Whitelist, never regex alone.** A naive matcher wraps `def forward(self, x)`,
  `(q, k, v)` and `[None, None]` as if they were shapes. Only annotate when
  *every* non-numeric part resolves to a known symbol.

---

## 5. Pedagogical contract

The reference course is 26 chapters, ~35 h, 177 exercises and questions. Match
the density, not necessarily the length. Every chapter must have, in order:

1. `<h1>` as `N · Title`, then a `.lead` naming what is genuinely hard here.
2. An `.mn.orient` block: *Builds on / Leads to / Why this chapter*.
3. `<h2>` sections numbered `N.1 ·`, `N.2 ·` …
4. **At least one interactive widget** (`data-viz`) or a drawn SVG figure per
   major section. Widgets are the point, not decoration.
5. **At least two `.quiz` checkpoints**, mid-chapter — not all at the end.
6. **3–6 `.exercise` blocks**, each with a `details.reveal` solution.
7. A `<h2>Sources &amp; further reading</h2>` with a `ul.tiny` — primary papers
   first, then the good secondary explainers, then the companion notebook.

Non-negotiables of voice and method:

- **Derive, never assert.** If a constant appears, show where it comes from.
  The reference course derives the `√d_k` in attention rather than stating it.
- **Every claim is derived or measured.** No hand-waving benchmarks.
- **Annotate every shape and name every axis** at each step of any
  transformation. Dimension confusion is the single most common failure mode.
- **Order by dependency, not by topic.** Each chapter uses what the last built.
- Prerequisites are stated honestly and kept minimal; where a chapter needs a
  piece of maths, it derives the one line that matters instead of assuming it.
- Close the loop on anything generative: a plausible-looking loss curve proves
  nothing. Show the output, or the assertion that would have caught the bug.

### The outline manifest

`assets/chapters.js` is the single source of truth for ordering, the sidebar,
prev/next nav and the progress bar:

```js
{ id: "06", f: "ch06-mha.html", t: "Multi-Head Attention: The Reshape Dance",
  s: "One-line subtitle", tag: "70 min" }
```

`id` must match the filename's digits and run `01..NN` with no gaps —
`check_ui.js` asserts all of it. Mark unwritten chapters `wip: true` and they
render listed-but-inert instead of as dead links.

---

## 6. Responsiveness contract — this is locked, and it is load-bearing

No page may scroll sideways at any width from **320px** up. `check_responsive.js`
enforces it. These rules are the reasons it currently holds:

- Prose is capped at `--measure: 37rem` (~70 characters). **Code, tables,
  figures, widgets and callouts break out** to the wider `--breakout`, which
  grows to 64rem above 100rem and 74rem above 120rem.
- `main > * { max-width: min(var(--measure), 100%) }` — the `min()` is
  load-bearing. A bare `37rem` is 592px and overflows every phone.
- `main` spans the full breakout and prose is narrowed by that child rule.
  Doing it the other way round fails: `max-width:100%` on a child resolves
  against `main` and silently caps it.
- **Every flex/grid container that can hold a `<pre>` needs `min-width: 0`.**
  Grid and flex items default to `min-width: auto` and refuse to shrink below
  their content, so a wide code block pushes the whole page sideways instead of
  scrolling inside its own box. This is the single most common regression.
- `body { overflow-x: clip }` as a backstop.
- Wide tables live in `.tw`; wide code scrolls inside its own `<pre>`.

Breakpoints, all already implemented:

| width | what changes |
|---|---|
| 82rem | sidebar becomes an off-canvas drawer |
| 66rem | margin notes fold back into the column |
| 44rem | `.grid2` collapses to one column |
| 34rem | chapter nav collapses to one column |
| 30rem | chapter-card tags hide |

`.wrap` reserves the sidebar with **padding, not `margin-left`**, so
`margin-inline: auto` still centres the content in the space that remains.

---

## 7. Theming — the trap that already bit once

There are **four** token blocks and they must stay in sync:

```
:root                                  light defaults
@media (prefers-color-scheme: dark) { :root:not([data-theme]) { … } }
:root[data-theme="light"]              explicit override
:root[data-theme="dark"]               explicit override
```

The media query is scoped `:not([data-theme])` so the OS preference applies only
when the reader has not chosen. Relying on specificity alone is fragile — it
silently failed for `background` while working for `color`. `check_ui.js` reads
the two explicit blocks, so a token added to only some blocks passes the test
and still breaks the page. Add to all four.

When you edit a theme with a script, verify per block. A regex keyed on
indentation put light values into the dark block once and the test did not catch
it, because the test only reads two of the four.

---

## 8. Definition of done

```bash
cd tests && npm install jsdom && npm run check:all
```

That runs three suites, and each answers a different question:

| suite | question | how |
|---|---|---|
| `check_ui.js` | is it BUILT and legible? | jsdom — structure, WCAG, manifest, links |
| `check_responsive.js` | does anything OVERFLOW? | real Chrome, fixed-width iframe |
| `check_interaction.js` | does it WORK when pressed? | real Chrome, fake clock |

The third one exists because the first two never press a button. A duration
select that saved its value but never restarted the running sprint passed both
of them and still shipped broken. It injects a fake clock before `course.js`
loads, so `__advance(60000)` moves a countdown one minute instantly and
exactly — a fifteen-minute timer is testable in milliseconds. Add a scenario
for every stateful control you build; `--all` replays the core one on every
chapter in the manifest.

Then `python serve.py` and click through, because the suites cannot see taste.

`check_ui.js` verifies, in **both** themes: every colour pair against WCAG
(4.5:1 body, 3.0:1 large/UI), ≥30° hue separation between the semantic hues,
the accent excluded from them, manifest integrity, that every internal link
resolves, and that each page builds its topbar, sidebar, nav, copy buttons and
quiz markers with no script errors.

Then check by hand, because the tests do not cover these:

- [ ] Both themes, on a real screen. Dark is not a token swap.
- [ ] Every widget responds to its controls and re-themes on toggle.
- [ ] Keyboard only: tab to every interactive thing, Escape clears state.
- [ ] Greyscale (devtools → Rendering → disable colour): every state still
      distinguishable. If it is not, you have encoded meaning in hue alone.
- [ ] `prefers-reduced-motion`: no animation, no transition.
- [ ] JS disabled: prose, colours and layout still correct.
- [ ] 320px wide: nothing overflows, drawer works.

Commit in coherent chunks with messages that explain **why**, not what.
