# Detailed course critique: *Transformers From Scratch*

## Executive verdict

This is already a rare course: it has the conceptual ambition of a graduate text, the implementation discipline of a good engineering repository, and a visual system built around the exact misconception it wants to prevent—losing track of tensor axes. Its strongest feature is not breadth by itself. It is the dependency chain: shapes lead to matmul, matmul to attention, attention to the block, the block to training, and training to the systems and alignment layers that modern courses often bolt on as disconnected surveys.

The course is strongest as a reference-quality, self-directed technical monograph. Before this pass, it was weaker as a course someone can reliably finish in irregular 5–20 minute windows. A learner could save only chapter completion, despite chapters being labelled 55–90 minutes and the capstone 4–8 hours. That made an interrupted session feel like lost work even when the learner had made real progress.

The shared learning interface now fixes that structural problem without changing any lesson text: all 195 existing top-level sections are resumable “bites”; the learner can run a 5, 10, 15, or 20 minute focus sprint; the exact section is saved automatically; section and exercise completion are independent; quiz state persists; and the home page resumes the precise section rather than merely the chapter.

The course is now substantially more finishable. It is not yet defensibly “the best there is” in every dimension. The remaining gap is mostly instructional coverage, not visual polish: cumulative retrieval, more evenly distributed active practice, additional runnable labs, and accessible mathematical semantics would require future authoring work.

## What was audited

The audit covered all 26 chapter pages, the course roadmap, shared interaction code, curriculum manifest, notebooks, reference implementation, and regression tests.

| Measure | Observed |
|---|---:|
| Chapter pages | 26 |
| Top-level chapter sections | 195 |
| Approximate chapter-page words | 64,636 |
| Quiz blocks | 32 |
| Multiple-choice questions | 91 |
| Exercise cards | 91 |
| Code blocks | 283 |
| Worked-example blocks | 31 |
| Reveal blocks | 69 |
| Tables | 55 |
| Interactive widget instances | 13 across 12 chapters |
| Companion notebooks | 9 |
| Source/further-reading sections | 25 of 26 chapters |

The five longest pages are Chapters 12, 1, 11, 5, and 2. The greatest average text density per top-level section occurs in Chapters 21, 12, 2, 11, and 16. Those are precisely the pages where re-entry and visible subchapter progress matter most.

## Scorecard

| Dimension | Assessment | Critique |
|---|---:|---|
| Curriculum architecture | 9.5/10 | Exceptionally coherent prerequisite ordering; advanced systems topics arrive only after the arithmetic needed to understand them. |
| Technical depth | 9.5/10 | Derivations, shapes, code, measured claims, and implementation tests go far beyond overview-style courses. |
| Conceptual explanation | 9/10 | Strong concrete metaphors and worked traces; difficult ideas are usually earned rather than asserted. |
| Active learning | 7.5/10 | Many questions and exercises in total, but their cadence is uneven and most quizzes use recognition rather than unaided recall. |
| Hands-on transfer | 8/10 | Excellent reference code, tests, and capstone; only 9 of 26 chapters have companion notebooks. |
| Attention-friendly pacing | 9/10 after this pass | Previously chapter-granular and punishing to interrupt; now section-granular, resumable, and focusable. |
| Navigation and continuity | 9/10 after this pass | Strong sidebar and cross-references, now joined by exact resume, section maps, and granular progress. |
| Accessibility | 8/10 after this pass | Excellent contrast and responsive intent; quiz semantics improved. Math and visualization semantics still need a dedicated screen-reader pass. |
| Epistemic transparency | 9/10 | Sources are pervasive and repository claims are tested; frontier claims are explicitly dated. |
| Maintenance readiness | 8/10 | Shared templates and tests are strong; a few public metadata statements have drifted from the implementation. |

## Where the course is excellent

### 1. The sequence teaches causes, not just components

The ordering is the course’s biggest intellectual advantage. Tensor shapes and matmul are not treated as preliminaries to rush past; they become the language used to explain every later operation. Position encoding appears after attention’s position-blindness is established. Backprop appears after a complete forward trace and before training. Precision follows training, so underflow and loss scaling answer a problem the learner has just encountered. GPU roofline arithmetic comes before the KV-cache memory wall, and distributed training comes before expert parallelism.

This creates a real explanatory chain. Many transformer courses provide a list of mechanisms; this one repeatedly answers “why did the next mechanism have to exist?”

### 2. Shape literacy is treated as the core skill

The colour system has semantic discipline: hues encode axes or roles rather than decoration. Shape badges, drawn tensor slabs, explicit reshapes, code comments, and the persistent dimension cheat sheet reinforce the same representation. This is an excellent choice for the target learner because most apparent confusion about attention is actually confusion about which dimension moved, split, or disappeared.

### 3. It connects theory, implementation, and verification

The combination of derivations, executable code, reference modules, and tests is outstanding. The learner is not asked to trust that causal masking, RoPE relativity, online softmax, or cached decoding work. The repository gives them a way to check. The training bug case study—where a plausible lower loss hid a two-token shift—is especially valuable because it teaches the difference between unit correctness and system correctness.

### 4. The advanced half is unusually substantive

FlashAttention, roofline reasoning, KV-cache economics, GQA/MLA, distributed communication costs, MoE routing, DPO, GRPO, and RLVR are not presented as vocabulary lists. The course derives or costs the mechanisms. That makes Parts IV–VI genuinely useful to an engineer who already knows a basic GPT implementation.

### 5. The visual identity supports the subject

The field-manual aesthetic is distinctive without becoming arbitrary decoration. Hard shadows, stamped cards, a restrained structural colour, and self-hosted fonts make the course memorable. More importantly, the saturated dimension palette is reserved for meaning. The design is at its best when the visual identity and the teaching model become the same system.

## Where the course was losing learners

### 1. A 90-minute chapter behaved like one task

Before this pass, the durable state was essentially “chapter done” or “chapter not done.” The reading bar showed progress only during the current visit. A learner interrupted halfway through Chapter 12 or 19 returned to a long page with no durable marker for what had actually been completed.

This is not a cosmetic issue. User-controlled segmentation can improve transfer in demanding multimedia learning, and distributed study is more effective for long-term retention than massed practice. See Mayer and Chandler’s learner-pacing experiments ([Journal of Educational Psychology, 2001](https://doi.org/10.1037/0022-0663.93.2.390)) and Cepeda et al.’s quantitative synthesis of distributed practice ([Psychological Bulletin, 2006](https://doi.org/10.1037/0033-2909.132.3.354)).

**Addressed now:** each existing `h2` section is a separately trackable bite with a visible estimate; sessions can be 5/10/15/20 minutes; focus mode shows one section while temporarily tucking away the rest; the current section is saved automatically; and the home page resumes it directly.

### 2. Interaction cadence is uneven

The totals look excellent—91 questions, 91 exercises, 69 reveals, and 13 widget instances—but they are not evenly distributed. Nineteen chapters have exactly one three-question quiz. Fourteen chapters have no interactive widget instance. Dense, abstract chapters such as precision, GPU architecture, and distributed training have no widget, while Chapters 14 and 19 also have no reveal block.

Not every chapter needs a slider. The issue is the distance between moments when a learner must do something rather than read. Across 195 top-level sections there are only 32 quiz blocks, so most sections have no immediate checkpoint.

**Partially addressed now:** section completion, focus controls, persistent quizzes, retry controls, and independent exercise marks create more frequent action and visible closure. The deeper fix—adding a recall prompt, manipulation, or prediction at the end of every major conceptual section—would change instructional content and was intentionally not done.

### 3. Most quiz activity is recognition, not retrieval

The multiple-choice checks are well written and give explanations, but choosing among visible options is less demanding than reconstructing an answer. The exercises provide stronger generative practice, yet they are usually grouped at chapter ends, when fatigue is highest.

Retrieval practice has repeatedly produced better delayed retention than restudying, including for educational prose and inference questions. See Roediger and Karpicke ([Psychological Science, 2006](https://doi.org/10.1111/j.1467-9280.2006.01693.x)) and Karpicke and Blunt ([Science, 2011](https://doi.org/10.1126/science.1199327)).

**Addressed now at the interface level:** attempts persist, explanations remain unlocked after a break, each quiz reports progress and score, radio-group semantics and arrow-key navigation are correct, and a learner can deliberately retry a check. A future content pass should add short free-recall prompts before selected multiple-choice blocks and a cumulative review queue that resurfaces earlier concepts.

### 4. Hands-on coverage is concentrated

There are 9 companion notebooks for 26 chapters. The chosen notebooks cover several high-value mechanisms, but the coverage drops in areas where experimentation would be especially educational: backprop, training, numeric precision, GPU/roofline arithmetic, distributed training, MoE, and alignment beyond DPO.

The reference implementation and inline code soften this gap, but a code block is not the same learning experience as changing a value, making a prediction, running a test, and explaining the result.

**Recommendation requiring future authoring:** add compact labs for Chapters 10–12, 14, 19–20, and 23–25. Prefer one falsifiable claim per lab over long notebooks. Each lab should begin with a prediction and finish with a test that can fail meaningfully.

### 5. The capstone milestones are still large

The capstone is authentic and well decomposed architecturally, but its five main exercise blocks are labelled roughly 1–2 hours each. For an attention-fragile learner, “Milestone 2 — The model” is still a project, not a next action.

**Addressed now:** capstone milestones are identified as project blocks rather than given misleading short read estimates, and the focus timer makes it legitimate to stop mid-milestone. The strongest future improvement would be tested sub-checkpoints—tokenizer round trip, one block forward shape, loss sanity check, cache equivalence, and DPO loss direction—using the existing test philosophy.

### 6. The three routes are useful but do not reduce local uncertainty

The roadmap offers “understand,” “build,” and “whole thing” routes. That is good macro-navigation. Within a chapter, however, the learner previously had no clear answer to “what is the smallest useful thing I can do now?”

**Addressed now:** the home resume card and chapter console always name one next bite, its approximate effort, and its saved completion state. This turns a curriculum route into an executable next action.

## Consistency and quality issues

These were deliberately left as critique items because the constraint was not to alter course content.

1. **Exercise counts have drifted.** The rendered chapter pages contain 91 exercise cards and 91 quiz questions, for 182 combined items. The home page says 177, and the README says 86 exercises plus 91 questions. One source of truth should generate all three numbers.
2. **The typography note is stale.** The course home says the reading face is Sitka, while the stylesheet and README describe the self-hosted Literata/Space Grotesk/JetBrains Mono system. The latter matches the implementation.
3. **Widget wording can be clearer.** There are 13 widget instances but 12 unique widget types because the tensor widget is reused. “12 interactive widget types” would remove ambiguity.
4. **Exercise difficulty labels are inconsistent.** The audit found descriptive labels such as `essential`, `code`, `thinking`, `hard`, `the main event`, `the interesting one`, and hour estimates. They are engaging, but they do not form a predictable difficulty/effort scheme. Separate “kind,” “difficulty,” and “time” would scan better.
5. **The capstone has no source section.** That is reasonable because it synthesizes earlier material, but a short “builds from” or “verification map” would make the exception intentional.
6. **The mobile test documents a real limitation.** Headless Chrome clamps the narrowest requested sizes to about 504 px, so the test does not prove a true 320 px layout. The CSS is carefully defensive, but a real-device or emulated-viewport check should exist before public release.

## Accessibility critique

The course already does several things unusually well: both themes have automated contrast checks; keyboard shortcuts are documented in code; focus-visible treatment is strong; the sidebar becomes a drawer; code and tables contain their own overflow; and reduced-motion preferences are respected.

This pass corrected the quiz interaction model from generic buttons to labelled radio groups with arrow-key movement, persisted selection state, live feedback, and visible retry. Focus mode is reversible, works without deleting or moving content, and print mode restores the entire lesson.

The remaining accessibility work is specialized:

- Mathematical expressions are primarily visual HTML spans rather than MathML or consistently labelled alternatives.
- Custom SVG/widget output should be audited for names, descriptions, keyboard operation, and non-colour equivalents.
- Shape colour is thoughtfully reinforced during axis highlighting, but each widget needs its own check for non-colour cues.
- The sidebar search filters chapters, not sections or full text; a command palette would help keyboard and screen-magnifier users reach a known concept directly.

## What changed in this pass

No lesson paragraph, derivation, example, exercise, answer, source, or curriculum order was changed.

The shared interface now provides:

- 195 resumable section-sized learning bites;
- 5, 10, 15, and 20 minute focus sprints;
- one-section focus mode with a one-click full-chapter restore;
- automatic exact-place saving and a home-page resume action;
- per-section completion and chapter maps;
- independent completion marks for all 91 exercise cards;
- persistent quiz attempts, score/progress display, explanations, and retry;
- correct radio-group keyboard and screen-reader semantics for quizzes;
- non-reloading chapter-completion controls;
- print behavior that always includes the full course content; and
- regression assertions for the new learning controls.

The design intentionally avoids streak pressure, points, confetti, and punitive timers. Engagement here means reducing the energy required to start, making progress visible at the right scale, and giving the learner permission to stop after a meaningful unit.

## Prioritized roadmap to make the course category-leading

### Priority 0 — completed without changing content

1. Granular section progress and exact resume.
2. Short, selectable focus sprints.
3. Focus mode that preserves the full chapter.
4. Persistent quiz and exercise state.
5. Quiz accessibility and retry.
6. Clear next action on the home page and in every chapter.

### Priority 1 — can reuse existing content

1. Build a local review queue from missed quiz items and resurface one due item at a time.
2. Add full-course search over section headings, glossary terms, code symbols, and sources.
3. Allow progress export/import so browser storage is not a single point of failure.
4. Add bookmarks and private local notes at section level.
5. Generate roadmap counts, duration totals, and widget totals from the manifest/audit rather than hand-maintaining them.
6. Standardize exercise metadata into type, difficulty, and effort.

### Priority 2 — requires instructional authoring

1. Add an unaided recall or prediction prompt to every major conceptual section.
2. Add compact notebooks for numeric precision, roofline arithmetic, distributed communication, MoE routing, and preference optimization.
3. Add high-value manipulatives for Chapters 10, 12, 14, and 19 rather than increasing decoration everywhere.
4. Turn the capstone milestones into test-backed sub-checkpoints that fit 10–20 minute work sessions.
5. Add cumulative, interleaved checks at the end of each Part rather than testing only the current chapter.
6. Give mathematical expressions and custom visualizations first-class accessible alternatives.

## Bottom line

The course’s intellectual core is already excellent. It is deep, opinionated, verifiable, and visually coherent. Its largest weakness was not that the content was boring; it was that the interface assumed long, uninterrupted attention and made partial progress nearly invisible. That has now been corrected at the shared-system level.

The next leap is not more prose or more visual flourish. It is a tighter learning loop: predict, manipulate, retrieve, get feedback, leave, and return at exactly the right place. The new short-session layer supplies the leave-and-return mechanics. Future authoring should concentrate on retrieval cadence, lab coverage, cumulative review, and accessible math.
