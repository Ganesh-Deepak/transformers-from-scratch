# The platform: many courses, one library

The goal: a reader moves from *Transformers From Scratch* to *Evaluation*, to
*GPU Programming*, to *Agents*, and it feels like one place. One look, one
theme setting, one record of what they have finished, and links that go
straight from "you need this" to the chapter that teaches it.

This file is the contract every course is built against. `NEW-COURSE-PROMPT.md`
points here; a course that follows it plugs into the library with no extra work.

---

## 1. The architecture, and why

**One repository per course, one hub.** Each course stays its own repository
and its own GitHub Pages deployment:

    https://ganesh-deepak.github.io/transformers-from-scratch/   (exists)
    https://ganesh-deepak.github.io/llm-evaluation/              (course 1)
    https://ganesh-deepak.github.io/gpu-programming/             (course 3)
    ...
    https://ganesh-deepak.github.io/                             the hub (the library)

The hub is the `ganesh-deepak.github.io` user-site repository, served at the
root of the domain.

Why separate repositories rather than one monorepo:

- **Courses are authored in parallel**, one or more sessions each. Separate
  repositories mean no merge traffic between them and an independent deploy for
  each.
- **The browser treats them as one site anyway.** Every GitHub Pages repository
  under one account is served from the same origin, `ganesh-deepak.github.io`.
  `localStorage` is per origin, so every course can read the same theme, the same
  sidebar preferences and every other course's progress. Unified progress needs
  no server and no accounts.
- **A course remains a finished, standalone artefact.** It still works if the hub
  is down, or when opened from disk. It is a portfolio piece in its own right.

What the hub adds is the layer above: the library page, the course switcher's
data, the cross-course prerequisite map and the shared glossary.

---

## 2. The registry: one manifest for the whole library

The hub serves `platform/courses.js`, the single source of truth for which
courses exist:

```js
const AIE_COURSES = [
  { slug: "transformers", title: "Transformers From Scratch",
    url: "/transformers-from-scratch/", store: "tfs-progress-v1",
    chapters: 28, ink: "--part-2", status: "live" },
  { slug: "evals", title: "Evaluation & Experimentation for LLM Systems",
    url: "/llm-evaluation/", store: "evals-progress-v1",
    chapters: 18, ink: "--part-4", status: "planned" },
  // ...
];
```

`status` is `planned | wip | live`. Each course also *vendors a copy* of this file
as `course/assets/courses.js`, so its topbar works offline. `check_platform.js`
(below) fails when the vendored copy is older than the hub's.

Each course keeps its own `assets/chapters.js` for its own outline. The hub reads
every course's `chapters.js` (same origin, plain `fetch`) to build the library
page and the prerequisite map. Nothing is duplicated by hand.

---

## 3. Shared state: keys, namespaces, migration

All courses share one origin, so `localStorage` keys are a shared namespace and
must be planned:

| key | scope | meaning |
|---|---|---|
| `tfs-theme` | **shared** | `light` / `dark` / unset (system). One setting for the whole library. |
| `tfs-side`, `tfs-side-parts`, `tfs-side-secs` | **shared** | sidebar open/collapsed and folding preferences |
| `tfs-inspector` | **shared** | shape inspector open/closed |
| `tfs-progress-v1` | transformers only | that course's progress (kept for compatibility) |
| `<slug>-progress-v1` | per course | every new course's progress object, same schema |

The `tfs-` prefix on shared keys is kept, not renamed: renaming would reset every
existing reader's settings. New shared keys use the `aie-` prefix.

The progress object's schema is the one `course.js` already writes (read, merge,
write, so two open tabs never erase each other's work). The hub reads every
course's `store` key from the registry to show library-wide progress.

---

## 4. What a reader sees

**In every course's topbar**: a **Library** control next to the course title.
It opens a panel listing every course from the registry with its progress
(`8 / 18 chapters`), the current course marked, planned courses listed but inert
(the same treatment as `wip` chapters). One click moves between courses.

**On the hub (the library page)**:

- every course as a card in its own part ink, with progress and a "continue
  where you left off" link (the per-course resume pointer already exists);
- **tracks**: the sequences from `ROADMAP.md` (Everyone, Systems, Applied, Model,
  Frontier labs), each shown as a path through the courses;
- **the prerequisite map**: chapters across all courses as a dependency graph,
  built from the cross-course references below, so a reader can see that
  *Inference Engineering ch 4* needs *Transformers ch 15*;
- **search** across every course's headings and glossary terms.

**Inside chapters**: cross-course references look and behave like ordinary
links, with the target course's name in front: "*Transformers* Ch&nbsp;15 — the KV
cache". If the reader has finished that chapter, the link says so.

---

## 5. Cross-course references: the markup and the check

Inside a course, chapter links are plain relative `href`s and `check_ui.js`
verifies them. Across courses, use one fixed form:

```html
<a class="xref" data-course="transformers" data-ch="15"
   href="/transformers-from-scratch/ch15-kvcache.html">Transformers&nbsp;Ch&nbsp;15</a>
```

- `href` is root-relative, so it works on the live site and through a local
  server that serves the whole `ganesh-deepak.github.io` layout.
- `data-course` + `data-ch` let the platform script mark finished chapters and
  feed the prerequisite map, without parsing URLs.
- `check_platform.js` resolves every `xref` against the registry and the target
  course's `chapters.js`, and fails on any that point at a missing course or
  chapter. Targets in a `planned` course are allowed but are flagged in the report.

"Builds on" notes (`.mn.orient`) may contain `xref`s. That is how the
prerequisite map gets its edges.

---

## 6. One design system, many courses

Every course uses the Field Manual system verbatim (`NEW-COURSE-PROMPT.md` §1–§4).
What legitimately differs per course:

- **Its semantic hues.** The five `--dim-*` colours mean "which tensor axis" in
  *Transformers*. Another course maps them to *its* natural categories (memory
  tier in GPU Programming, pipeline stage in Inference, agent role in Agents) or
  leaves them unused. Never two meanings inside one course.
- **Its ink in the library.** Each course gets one `--part-N` colour for its card
  and its library badge. Inks are navigation-only, exactly as in the sidebar.

What must not differ: tokens, typography, layout, components, the sheet, the
study console, the quiz and exercise engines, accessibility behaviour. A reader
should never relearn the interface.

**Keeping the shared assets in step.** Each course vendors `course/assets/`
(style, course.js, axes.js, viz.js, fonts). The hub keeps the canonical copies
under `platform/assets/` with a version number. `check_platform.js` compares each
course's vendored files with the canonical version and reports drift. A
design-system change is made once in the hub and rolled out to each course as a
normal commit.

---

## 7. The checks that hold it together

`tests/check_platform.js` (lives in the hub, also runnable from any course):

1. the vendored `courses.js` matches the hub's;
2. every `xref` in every live course resolves to a real chapter;
3. no two courses use the same `store` key; shared keys are only the ones in §3;
4. the vendored design-system assets match the canonical version (or the course
   is marked as pinned to an older one, with a reason);
5. every course's `chapters.js` is valid (`id` sequence, files exist).

Each course still runs its own full suite (`check_ui`, `check_responsive`,
`check_interaction`, `shoot`, its Python tests and notebooks) before it deploys.

---

## 8. Build order

**Status, November 2026:** steps 1 and 2 are done. Transformers has the Library control
(`course.js` reads `COURSE` from `chapters.js` and the vendored `assets/courses.js`); the hub
repository holds the registry, the canonical assets, the library page (tracks, cards with
progress, the prerequisite map, search), `tools/build.py`, `tools/check_platform.py` and
`tools/serve.py`; and all ten new courses have generated outline sites under their final
paths, so switching, shared progress and cross-course links are exercised for real.

1. **Prepare *Transformers From Scratch*** (this repository): add the Library
   control to `course.js`, read `courses.js`, and keep `tfs-progress-v1`. This is
   the template every new course copies.
2. **Create the hub repository** `ganesh-deepak.github.io`: library page, tracks,
   registry, canonical assets, `check_platform.js`.
3. **Author each course** with `NEW-COURSE-PROMPT.md`, which already requires
   the platform contract (slug, store key, `xref` markup, vendored registry).
4. As courses go live, flip their `status` in the registry. The library, the
   switcher and the prerequisite map update without touching any other course.
