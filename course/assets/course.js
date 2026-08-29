/* ============================================================================
   course.js — theme, progress, layout scaffold, quiz engine, code highlighting.
   No dependencies.
   ========================================================================= */

/* ------------------------------------------------------------- storage */

const Store = (() => {
  const KEY = "tfs-progress-v1";
  let mem = {};
  let ok = true;
  try { mem = JSON.parse(localStorage.getItem(KEY) || "{}"); }
  catch (e) { ok = false; mem = {}; }
  const save = () => { if (ok) { try { localStorage.setItem(KEY, JSON.stringify(mem)); } catch (e) { ok = false; } } };
  return {
    get: k => mem[k],
    set: (k, v) => { mem[k] = v; save(); },
    remove: k => { delete mem[k]; save(); },
    all: () => mem,
    reset: () => { mem = {}; save(); },
    available: () => ok,
  };
})();

const doneKey = id => "done:" + id;
const isDone  = id => !!Store.get(doneKey(id));

/* --------------------------------------------------------------- theme */
/* Precedence: explicit user choice > OS preference. The toggle stamps
   data-theme on <html>, which the stylesheet lets win in both directions. */

const Theme = (() => {
  const KEY = "tfs-theme";
  const root = document.documentElement;

  const osPrefersDark = () =>
    window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

  let choice = null;
  try { choice = localStorage.getItem(KEY); } catch (e) { /* blocked */ }

  function apply() {
    if (choice === "light" || choice === "dark") root.setAttribute("data-theme", choice);
    else root.removeAttribute("data-theme");
    document.querySelectorAll(".theme-toggle").forEach(b => {
      b.dataset.isDark = isDark() ? "1" : "0";
      b.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
    });
    window.dispatchEvent(new CustomEvent("themechange"));
  }
  function isDark() {
    return choice ? choice === "dark" : osPrefersDark();
  }
  function toggle() {
    choice = isDark() ? "light" : "dark";
    try { localStorage.setItem(KEY, choice); } catch (e) { /* blocked */ }
    apply();
  }

  // Follow the OS if the user has not made an explicit choice.
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", () => { if (!choice) apply(); });
  }

  return { apply, toggle, isDark };
})();

/* Applied before first paint (see the inline snippet in each page's <head>)
   to avoid a flash of the wrong theme. */
Theme.apply();

const SUN = '<svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4"/></svg>';
const MOON = '<svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.2 8.2 0 1 0 10.2 10.2z"/></svg>';

/* -------------------------------------------------------------- layout */
/* Chapters ship a bare <main>. Wrap it so the live margin has an anchor. */

function buildLayout() {
  const main = document.querySelector("main");
  if (!main || main.parentElement.classList.contains("wrap")) return;
  const wrap = document.createElement("div");
  wrap.className = "wrap";
  main.parentNode.insertBefore(wrap, main);
  wrap.appendChild(main);
}

/* --------------------------------------------------------------- topbar */

function buildTopbar(currentId) {
  const ch = CHAPTERS.find(c => c.id === currentId);
  const part = PARTS.find(p => p.ch.some(c => c.id === currentId));
  const total = CHAPTERS.length;
  const done = CHAPTERS.filter(c => isDone(c.id)).length;

  const bar = document.createElement("header");
  bar.className = "topbar";
  bar.innerHTML =
    `<button class="sb-toggle" type="button" aria-label="Chapters">${ICON_MENU}</button>` +
    `<a class="home" href="index.html">Transformers <i>From Scratch</i></a>` +
    (ch ? `<div class="crumb">${part ? part.n + " &middot; " : ""}Ch ${ch.id}</div>` : "") +
    `<div class="spacer"></div>` +
    (ch ? `<button class="focus-toggle" type="button" aria-pressed="false">Focus</button>` : "") +
    `<div class="prog" title="Chapters marked complete">${done}/${total}<span> chapters</span></div>` +
    `<button class="theme-toggle" type="button">${SUN}${MOON}</button>`;
  document.body.prepend(bar);
  bar.querySelector(".theme-toggle").onclick = Theme.toggle;
  bar.querySelector(".sb-toggle").onclick = toggleSidebar;
  Theme.apply();
  if (!ch) buildHomeStudy();
}

/* ----------------------------------------------------------- prev/next */

function buildChapterNav(currentId) {
  const i = CHAPTERS.findIndex(c => c.id === currentId);
  if (i < 0) return;
  // Skip over chapters that are numbered but not written yet, so prev/next
  // always lands on a page that exists.
  const live = d => { let j = i + d; while (CHAPTERS[j] && CHAPTERS[j].wip) j += d; return CHAPTERS[j]; };
  const prev = live(-1), next = live(1);
  const main = document.querySelector("main");

  const row = document.createElement("div");
  row.className = "done-row";
  row.innerHTML =
    `<div class="lbl">${isDone(currentId)
      ? "Marked complete."
      : "Finished this chapter? Mark it so the roadmap tracks your progress."}</div>
     <button class="btn ${isDone(currentId) ? "" : "primary"}" id="mark" type="button">
       ${isDone(currentId) ? "Undo" : "Mark complete"}</button>`;
  main.appendChild(row);
  row.querySelector("#mark").onclick = () => {
    Store.set(doneKey(currentId), !isDone(currentId));
    const nowDone = isDone(currentId);
    row.querySelector(".lbl").textContent = nowDone
      ? "Marked complete."
      : "Finished this chapter? Mark it so the roadmap tracks your progress.";
    const mark = row.querySelector("#mark");
    mark.textContent = nowDone ? "Undo" : "Mark complete";
    mark.classList.toggle("primary", !nowDone);
    document.querySelector(`.sb-ch[href="${CHAPTERS[i].f}"]`)?.classList.toggle("done", nowDone);
    const done = CHAPTERS.filter(c => isDone(c.id)).length;
    const prog = document.querySelector(".topbar .prog");
    if (prog) prog.innerHTML = `${done}/${CHAPTERS.length}<span> chapters</span>`;
  };

  const nav = document.createElement("nav");
  nav.className = "chapnav";
  nav.innerHTML =
    (prev ? `<a href="${prev.f}"><div class="d">&larr; Previous</div><div class="t">${prev.id}. ${prev.t}</div></a>`
          : `<a href="index.html"><div class="d">&larr;</div><div class="t">Course home</div></a>`) +
    (next ? `<a class="next" href="${next.f}"><div class="d">Next &rarr;</div><div class="t">${next.id}. ${next.t}</div></a>`
          : `<a class="next" href="index.html"><div class="d">&rarr;</div><div class="t">Back to course home</div></a>`);
  main.appendChild(nav);
}

/* -------------------------------------------- Python syntax highlighting */

const PY_KW = new Set(("False None True and as assert async await break class continue def del elif else except " +
  "finally for from global if import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
const PY_BI = new Set(("abs all any bool dict enumerate float format int isinstance len list map max min print range " +
  "repr reversed round set sorted str sum super tuple type zip torch nn F np self").split(" "));

function highlightPython(src) {
  const out = [];
  let i = 0;
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  while (i < src.length) {
    const c = src[i];
    if (c === "#") {
      let j = src.indexOf("\n", i); if (j < 0) j = src.length;
      out.push('<span class="cm">' +
        esc(src.slice(i, j)).replace(/(\([^)]*?\b(?:B|T|C|H|D|N|S)\b[^)]*?\))/g, '<span class="sh">$1</span>') +
        "</span>");
      i = j; continue;
    }
    if (src.startsWith('"""', i) || src.startsWith("'''", i)) {
      const q = src.slice(i, i + 3);
      let j = src.indexOf(q, i + 3); j = j < 0 ? src.length : j + 3;
      out.push('<span class="st">' + esc(src.slice(i, j)) + "</span>"); i = j; continue;
    }
    if (c === '"' || c === "'") {
      let j = i + 1;
      while (j < src.length && src[j] !== c) { if (src[j] === "\\") j++; j++; }
      j = Math.min(j + 1, src.length);
      out.push('<span class="st">' + esc(src.slice(i, j)) + "</span>"); i = j; continue;
    }
    if (c === "@" && /[A-Za-z_]/.test(src[i + 1] || "")) {
      let j = i + 1; while (j < src.length && /[\w.]/.test(src[j])) j++;
      out.push('<span class="dc">' + esc(src.slice(i, j)) + "</span>"); i = j; continue;
    }
    if (/[0-9]/.test(c) && !/[\w.]/.test(src[i - 1] || " ")) {
      let j = i; while (j < src.length && /[\w.]/.test(src[j])) j++;
      out.push('<span class="nu">' + esc(src.slice(i, j)) + "</span>"); i = j; continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i; while (j < src.length && /\w/.test(src[j])) j++;
      const w = src.slice(i, j);
      const isCall = src[j] === "(";
      if (PY_KW.has(w))      out.push('<span class="kw">' + w + "</span>");
      else if (PY_BI.has(w)) out.push('<span class="bi">' + w + "</span>");
      else if (isCall)       out.push('<span class="fn">' + w + "</span>");
      else                   out.push(esc(w));
      i = j; continue;
    }
    out.push(esc(c)); i++;
  }
  return out.join("");
}

function decorateCode() {
  document.querySelectorAll("pre > code").forEach(code => {
    if (code.dataset.done) return;
    code.dataset.done = "1";
    const raw = code.textContent;
    if (!code.classList.contains("nohl")) code.innerHTML = highlightPython(raw);
    const btn = document.createElement("button");
    btn.className = "copy-btn"; btn.type = "button"; btn.textContent = "copy";
    btn.onclick = () => {
      navigator.clipboard?.writeText(raw);
      btn.textContent = "copied"; setTimeout(() => btn.textContent = "copy", 1200);
    };
    code.parentElement.appendChild(btn);
  });
}

/* ---------------------------------------------------------- quiz engine */

function initQuizzes(currentId = "page") {
  document.querySelectorAll(".quiz").forEach((quiz, quizIndex) => {
    if (quiz.dataset.built) return;
    quiz.dataset.built = "1";
    const h = document.createElement("div");
    h.className = "qt";
    h.textContent = quiz.dataset.title || "Check yourself";
    quiz.prepend(h);

    const items = [...quiz.querySelectorAll(".qitem")];
    const status = document.createElement("div");
    status.className = "quiz-status";
    status.setAttribute("aria-live", "polite");
    h.insertAdjacentElement("afterend", status);

    const reset = document.createElement("button");
    reset.className = "quiz-reset";
    reset.type = "button";
    reset.textContent = "Retry";
    reset.hidden = true;
    h.appendChild(reset);

    const chosen = Array(items.length).fill(null);
    const refreshStatus = () => {
      const answered = chosen.filter(v => v !== null).length;
      const correct = chosen.reduce((n, v, i) =>
        n + (v !== null && v === parseInt(items[i].dataset.answer, 10) ? 1 : 0), 0);
      status.textContent = answered === items.length
        ? `${correct}/${items.length} correct · explanations unlocked`
        : `${answered}/${items.length} answered`;
      reset.hidden = answered === 0;
    };

    items.forEach((item, itemIndex) => {
      const ans = parseInt(item.dataset.answer, 10);
      const opts = [...item.querySelectorAll(".opt")];
      const expl = item.querySelector(".expl");
      const group = item.querySelector(".opts");
      const key = `quiz:${currentId}:${quizIndex}:${itemIndex}`;
      const prompt = item.querySelector(".qq")?.textContent.trim() || `Question ${itemIndex + 1}`;
      if (group) {
        group.setAttribute("role", "radiogroup");
        group.setAttribute("aria-label", prompt);
      }

      const announce = document.createElement("span");
      announce.className = "sr-only";
      announce.setAttribute("aria-live", "polite");
      item.appendChild(announce);

      const lockChoice = (idx, persist = true) => {
        if (item.dataset.answered) return;
        item.dataset.answered = "1";
        chosen[itemIndex] = idx;
        opts.forEach((oo, jj) => {
          oo.classList.add("disabled");
          oo.setAttribute("aria-disabled", "true");
          oo.setAttribute("aria-checked", String(jj === idx));
          oo.setAttribute("tabindex", jj === idx ? "0" : "-1");
          if (jj === ans) oo.classList.add("correct");
        });
        if (idx !== ans) opts[idx]?.classList.add("wrong");
        if (expl) expl.classList.add("show");
        announce.textContent = idx === ans
          ? "Correct. Explanation shown."
          : "Not quite. The correct answer and explanation are shown.";
        if (persist) Store.set(key, idx);
        refreshStatus();
      };

      opts.forEach((o, idx) => {
        const mk = document.createElement("div");
        mk.className = "mk"; mk.textContent = "ABCDEF"[idx];
        o.prepend(mk);
        o.setAttribute("role", "radio");
        o.setAttribute("aria-checked", "false");
        o.setAttribute("tabindex", idx === 0 ? "0" : "-1");
        o.onclick = () => lockChoice(idx);
        o.onkeydown = e => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault(); lockChoice(idx); return;
          }
          if (item.dataset.answered || !["ArrowDown", "ArrowRight", "ArrowUp", "ArrowLeft"].includes(e.key)) return;
          e.preventDefault();
          const delta = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : -1;
          const next = (idx + delta + opts.length) % opts.length;
          opts.forEach((oo, jj) => oo.setAttribute("tabindex", jj === next ? "0" : "-1"));
          opts[next].focus();
        };
      });

      const saved = Store.get(key);
      if (Number.isInteger(saved) && saved >= 0 && saved < opts.length) lockChoice(saved, false);
    });

    reset.onclick = () => {
      items.forEach((_, itemIndex) => Store.remove(`quiz:${currentId}:${quizIndex}:${itemIndex}`));
      location.reload();
    };
    refreshStatus();
  });
}

/* ------------------------------------------------------------- sidebar */

const ICON_MENU = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>';

/* One control, two behaviours:
   wide  -> collapse/expand the rail, handing its width back to the content
   narrow-> open/close the off-canvas drawer
   The wide-screen preference persists; the drawer never does. */
const WIDE_MQ = "(min-width: 82.001rem)";
const isWide = () => !window.matchMedia || window.matchMedia(WIDE_MQ).matches;

function applySidebarPref() {
  let collapsed = false;
  try { collapsed = localStorage.getItem("tfs-side") === "collapsed"; } catch (e) { /* blocked */ }
  document.body.classList.toggle("side-collapsed", collapsed && isWide());
  if (!isWide()) document.body.classList.remove("side-collapsed");
  const btn = document.querySelector(".sb-toggle");
  if (btn) {
    const open = isWide() ? !collapsed : document.body.classList.contains("side-open");
    btn.setAttribute("aria-expanded", String(open));
    btn.setAttribute("aria-label", open ? "Hide chapter list" : "Show chapter list");
  }
  if (window.layoutMarginNotes) window.layoutMarginNotes();
}

function toggleSidebar() {
  if (isWide()) {
    const collapsed = !document.body.classList.contains("side-collapsed");
    try { localStorage.setItem("tfs-side", collapsed ? "collapsed" : "open"); } catch (e) { /* blocked */ }
  } else {
    document.body.classList.toggle("side-open");
  }
  applySidebarPref();
  // the content column just changed width; re-flow the margin after paint
  requestAnimationFrame(() => window.layoutMarginNotes && window.layoutMarginNotes());
}

function slug(s) {
  return s.toLowerCase().replace(/[^\w\s·.-]/g, "").trim()
          .replace(/[\s·.]+/g, "-").replace(/-+/g, "-").slice(0, 60);
}

/* Give every h2 a stable id so the section list can link to it. */
function tagSections() {
  const out = [];
  document.querySelectorAll("main > h2").forEach(h => {
    if (!h.id) h.id = "s-" + slug(h.textContent);
    // strip the leading "5.3 · " so the sidebar stays narrow
    const label = h.textContent.replace(/^\s*\d+(\.\d+)?\s*[·.]\s*/, "");
    out.push({ id: h.id, label, el: h });
  });
  return out;
}

/* ------------------------------------------------------ short sessions */
/* The course stays intact. This layer lets a learner work one existing h2
   section at a time, save that exact place, and stop after a short sprint. */

let Study = null;

const sectionDoneKey = (chapterId, sectionId) => `section:${chapterId}:${sectionId}`;
const exerciseDoneKey = (chapterId, index) => `exercise:${chapterId}:${index}`;

function sectionGroups(sections) {
  return sections.map((section, index) => {
    const stop = sections[index + 1]?.el || null;
    const nodes = [];
    let node = section.el;
    while (node && node !== stop) {
      nodes.push(node);
      node = node.nextElementSibling;
    }
    const text = nodes.map(n => n.textContent || "").join(" ").trim();
    const words = text ? text.split(/\s+/).length : 0;
    const code = nodes.reduce((n, el) => n + el.querySelectorAll("pre").length + (el.matches("pre") ? 1 : 0), 0);
    const questions = nodes.reduce((n, el) => n + el.querySelectorAll(".qitem").length + (el.matches(".qitem") ? 1 : 0), 0);
    const exercises = nodes.reduce((n, el) => n + el.querySelectorAll(".exercise").length + (el.matches(".exercise") ? 1 : 0), 0);
    const widgets = nodes.reduce((n, el) => n + el.querySelectorAll("[data-viz]").length + (el.matches("[data-viz]") ? 1 : 0), 0);
    const project = nodes.some(el => [...el.querySelectorAll(".exercise .lvl")]
      .some(x => /\bhrs?\b/i.test(x.textContent)));
    const minutes = Math.max(2, Math.min(20,
      Math.ceil(words / 220 + code * .35 + questions * .7 + exercises * 1.5 + widgets * 1.5)));
    return { ...section, nodes, minutes, timeLabel: project ? "project block" : `~${minutes} min` };
  });
}

function saveResume(chapterId, section) {
  const chapter = CHAPTERS.find(c => c.id === chapterId);
  if (!chapter || !section) return;
  Store.set("resume", {
    chapterId,
    sectionId: section.id,
    chapter: chapter.t,
    section: section.label,
    file: chapter.f,
    updated: Date.now(),
  });
}

function initExerciseProgress(currentId) {
  document.querySelectorAll(".exercise").forEach((exercise, index) => {
    const head = exercise.querySelector(":scope > .eh");
    if (!head || head.querySelector(".ex-check")) return;
    const key = exerciseDoneKey(currentId, index);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ex-check";
    const refresh = () => {
      const done = !!Store.get(key);
      exercise.classList.toggle("exercise-done", done);
      button.classList.toggle("is-done", done);
      button.setAttribute("aria-pressed", String(done));
      button.textContent = done ? "Done ✓" : "Mark done";
    };
    button.onclick = () => { Store.set(key, !Store.get(key)); refresh(); };
    head.appendChild(button);
    refresh();
  });
}

function buildStudyConsole(currentId, sections) {
  if (!currentId || !sections.length) return null;
  const main = document.querySelector("main");
  const groups = sectionGroups(sections);
  Store.set(`sections-total:${currentId}`, groups.length);

  // The orientation note used to sit in the live margin. A wide study console
  // occupies that same column, so long notes could visually cover—and intercept
  // clicks on—the focus controls. Fold only this introductory note into the
  // reading column when the short-session interface is present.
  main.querySelector(":scope > .mn.orient")?.classList.add("study-orient");

  const consoleEl = document.createElement("section");
  consoleEl.className = "study-console";
  consoleEl.setAttribute("aria-labelledby", "study-title");
  consoleEl.innerHTML = `
    <div class="study-kicker"><span>Short-session mode</span><span class="study-count"></span></div>
    <div class="study-main">
      <div class="study-copy">
        <strong id="study-title">Do one bite, then stop guilt-free.</strong>
        <span class="study-next"></span>
      </div>
      <div class="study-actions">
        <label>Session
          <select class="study-duration" aria-label="Focus-session length">
            <option value="5">5 min</option><option value="10">10 min</option>
            <option value="15">15 min</option><option value="20">20 min</option>
          </select>
        </label>
        <button class="btn primary study-start" type="button">Focus suggested bite</button>
        <button class="btn study-exit" type="button" hidden>Show full chapter</button>
        <span class="study-timer" aria-live="polite"></span>
      </div>
    </div>
    <div class="study-message" role="status" aria-live="polite">Your exact place is saved automatically.</div>
    <nav class="study-nav" aria-label="Bite navigation">
      <button class="study-nav-first" type="button">First bite</button>
      <button class="study-nav-prev" type="button">&larr; Previous</button>
      <span class="study-position" aria-live="polite"></span>
      <button class="study-nav-next" type="button">Next &rarr;</button>
      <button class="study-nav-last" type="button">Last bite</button>
    </nav>
    <details class="study-map">
      <summary>Chapter map <span></span></summary>
      <div class="study-steps"></div>
    </details>`;
  main.insertBefore(consoleEl, sections[0].el);

  const stepList = consoleEl.querySelector(".study-steps");
  const footers = [];
  groups.forEach((group, index) => {
    const tail = group.nodes[group.nodes.length - 1];
    const step = document.createElement("button");
    step.type = "button";
    step.className = "study-step";
    step.dataset.index = index;
    step.innerHTML = `<span class="study-step-mark">${index + 1}</span>` +
      `<span class="study-step-label">${group.label}</span><span class="study-step-time">${group.timeLabel}</span>`;
    stepList.appendChild(step);

    const toolsEl = document.createElement("div");
    toolsEl.className = "section-tools";
    toolsEl.dataset.section = group.id;
    toolsEl.innerHTML = `<span>${group.timeLabel}</span>` +
      `<button class="section-focus" type="button">Focus here</button>` +
      `<button class="section-done" type="button" aria-pressed="false">Mark section done</button>`;
    group.el.insertAdjacentElement("afterend", toolsEl);
    group.nodes.splice(1, 0, toolsEl);

    const footer = document.createElement("nav");
    footer.className = "bite-footer";
    footer.dataset.index = index;
    footer.setAttribute("aria-label", `End of bite ${index + 1}`);
    footer.innerHTML = `
      <span>Bite ${index + 1} of ${groups.length}</span>
      <button class="bite-prev" type="button">&larr; Previous bite</button>
      <button class="bite-next" type="button">Next bite &rarr;</button>
      <button class="bite-last" type="button">Last bite</button>
      <button class="bite-exit" type="button">Show full chapter</button>`;
    tail.insertAdjacentElement("afterend", footer);
    group.nodes.push(footer);
    footers.push(footer);
  });

  const count = consoleEl.querySelector(".study-count");
  const nextLabel = consoleEl.querySelector(".study-next");
  const mapSummary = consoleEl.querySelector(".study-map summary span");
  const start = consoleEl.querySelector(".study-start");
  const exitButton = consoleEl.querySelector(".study-exit");
  const duration = consoleEl.querySelector(".study-duration");
  const timerEl = consoleEl.querySelector(".study-timer");
  const message = consoleEl.querySelector(".study-message");
  const position = consoleEl.querySelector(".study-position");
  const firstButton = consoleEl.querySelector(".study-nav-first");
  const prevButton = consoleEl.querySelector(".study-nav-prev");
  const nextButton = consoleEl.querySelector(".study-nav-next");
  const lastButton = consoleEl.querySelector(".study-nav-last");
  const topToggle = document.querySelector(".focus-toggle");
  const savedDuration = parseInt(Store.get("study:minutes"), 10);
  duration.value = [5, 10, 15, 20].includes(savedDuration) ? String(savedDuration) : "10";

  let activeIndex = -1;
  let timer = null;
  let secondsLeft = 0;

  const isComplete = index => !!Store.get(sectionDoneKey(currentId, groups[index].id));
  const nextIndex = () => {
    const resume = Store.get("resume");
    if (resume?.chapterId === currentId) {
      const saved = groups.findIndex(g => g.id === resume.sectionId);
      if (saved >= 0 && !isComplete(saved)) return saved;
    }
    const first = groups.findIndex((_, i) => !isComplete(i));
    return first >= 0 ? first : groups.length - 1;
  };

  const refresh = () => {
    const completed = groups.filter((_, i) => isComplete(i)).length;
    count.textContent = `${completed}/${groups.length} sections`;
    mapSummary.textContent = `· ${completed}/${groups.length} done`;
    const next = groups[nextIndex()];
    nextLabel.textContent = completed === groups.length
      ? "Chapter map complete — revisit any section or mark the chapter complete below."
      : `Next: ${next.label} · ${next.timeLabel}`;
    groups.forEach((group, index) => {
      const done = isComplete(index);
      const step = stepList.querySelector(`[data-index="${index}"]`);
      step?.classList.toggle("is-done", done);
      step?.setAttribute("aria-label", `${group.label}, ${group.timeLabel}${done ? ", complete" : ""}`);
      const toolsEl = document.querySelector(`.section-tools[data-section="${group.id}"]`);
      const doneButton = toolsEl?.querySelector(".section-done");
      if (doneButton) {
        doneButton.classList.toggle("is-done", done);
        doneButton.setAttribute("aria-pressed", String(done));
        doneButton.textContent = done ? "Section done ✓" : "Mark section done";
      }
    });
  };

  const showTimer = () => {
    const mins = Math.floor(secondsLeft / 60);
    const secs = String(secondsLeft % 60).padStart(2, "0");
    timerEl.textContent = `${mins}:${secs}`;
  };

  const stopTimer = () => {
    clearInterval(timer);
    timer = null;
  };

  /* requestAnimationFrame does not fire while a tab is hidden or backgrounded,
     and these callbacks are what actually perform the scroll. Without a
     fallback, opening a chapter in a background tab and switching to it later
     leaves the reader parked at the top of the document with no idea why.
     First-wins, so a normal foreground load still scrolls after real layout. */
  const afterLayout = fn => {
    let done = false;
    const run = () => { if (done) return; done = true; fn(); };
    requestAnimationFrame(() => requestAnimationFrame(run));
    setTimeout(run, 80);
  };

  const scrollAfterLayout = element => {
    afterLayout(() => {
      const topbar = document.querySelector(".topbar");
      const offset = (topbar?.offsetHeight || 0) + 12;
      const top = element.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top: Math.max(0, top), behavior: "auto" });
    });
  };

  /* Entering a bite anchored on the console, which is right on a wide screen
     (the console is short, so the bite's heading sits just under it) and wrong
     on a phone. In focus mode the console is position:relative and runs
     369-431px tall at narrow widths, so the heading landed 660-726px down the
     page and the reader arrived looking at controls with the actual content
     below the fold. It read as "cut off" or "skipped".

     So: anchor on the console only when doing so still leaves the heading
     comfortably on screen. Otherwise anchor on the heading itself — seeing
     the content you asked for matters more than keeping the controls in view,
     and they are one scroll up. */
  const scrollIntoBite = (consoleElement, headingElement) => {
    if (!headingElement) return scrollAfterLayout(consoleElement);
    afterLayout(() => {
      const topbar = document.querySelector(".topbar");
      const offset = (topbar?.offsetHeight || 0) + 12;
      const viewport = document.documentElement.clientHeight;
      const consoleTop = consoleElement.getBoundingClientRect().top + window.scrollY - offset;
      const headingTop = headingElement.getBoundingClientRect().top + window.scrollY - offset;
      const headingAfterConsoleScroll = (headingTop - consoleTop) + offset;
      const consoleAnchorKeepsHeadingVisible = headingAfterConsoleScroll < viewport - 140;
      const top = consoleAnchorKeepsHeadingVisible ? consoleTop : headingTop;
      window.scrollTo({ top: Math.max(0, top), behavior: "auto" });
    });
  };

  const updateNavigation = index => {
    const focused = activeIndex >= 0;
    const group = groups[index];
    position.textContent = `${focused ? "Bite" : "Suggested bite"} ${index + 1} of ${groups.length} · ${group.label}`;
    firstButton.disabled = index === 0;
    prevButton.disabled = index === 0;
    nextButton.disabled = index === groups.length - 1;
    lastButton.disabled = index === groups.length - 1;
    groups.forEach((_, i) => stepList.querySelector(`[data-index="${i}"]`)?.classList.toggle("is-active", focused && i === index));
    footers.forEach((footer, i) => {
      footer.querySelector(".bite-prev").disabled = i === 0;
      footer.querySelector(".bite-next").disabled = i === groups.length - 1;
      footer.querySelector(".bite-last").disabled = i === groups.length - 1;
    });
  };

  const exitFocus = () => {
    const restoreIndex = activeIndex >= 0 ? activeIndex : nextIndex();
    stopTimer();
    document.body.classList.remove("focus-mode");
    main.querySelectorAll(":scope > .focus-hidden").forEach(el => el.classList.remove("focus-hidden"));
    activeIndex = -1;
    timerEl.textContent = "";
    exitButton.hidden = true;
    start.hidden = false;
    start.textContent = "Focus suggested bite";
    if (topToggle) { topToggle.setAttribute("aria-pressed", "false"); topToggle.textContent = "Focus"; }
    message.textContent = "Full chapter restored. Your exact place is saved.";
    layoutMarginNotes();
    updateNavigation(nextIndex());
    scrollAfterLayout(groups[restoreIndex].el);
  };

  const beginFocus = (index = nextIndex()) => {
    if (!groups[index]) return;
    const entering = activeIndex < 0;
    activeIndex = index;
    const visible = new Set([main.querySelector(":scope > h1"), consoleEl, ...groups[index].nodes]);
    [...main.children].forEach(el => el.classList.toggle("focus-hidden", !visible.has(el)));
    document.body.classList.add("focus-mode");
    // Margin notes may carry large inline offsets from the full chapter's
    // collision layout. Re-running after focus-mode makes them static and
    // clears those offsets before we measure or scroll the bite.
    layoutMarginNotes();
    exitButton.hidden = false;
    start.hidden = true;
    if (topToggle) { topToggle.setAttribute("aria-pressed", "true"); topToggle.textContent = "Exit focus"; }
    saveResume(currentId, groups[index]);
    if (entering) {
      stopTimer();
      consoleEl.classList.remove("sprint-finished");
      secondsLeft = parseInt(duration.value, 10) * 60;
      showTimer();
      timer = setInterval(() => {
        secondsLeft--;
        showTimer();
        if (secondsLeft <= 0) {
          stopTimer();
          consoleEl.classList.add("sprint-finished");
          message.textContent = "Sprint complete. Stop guilt-free, or finish the current thought.";
        }
      }, 1000);
    }
    message.textContent = `Focused on “${groups[index].label}”. Everything else is temporarily tucked away.`;
    refresh();
    updateNavigation(index);
    scrollIntoBite(consoleEl, groups[index] && groups[index].el);
  };

  const navigate = delta => {
    const base = activeIndex >= 0 ? activeIndex : nextIndex();
    beginFocus(Math.max(0, Math.min(groups.length - 1, base + delta)));
  };

  duration.onchange = () => Store.set("study:minutes", parseInt(duration.value, 10));
  start.onclick = () => beginFocus(nextIndex());
  exitButton.onclick = exitFocus;
  if (topToggle) topToggle.onclick = () => activeIndex >= 0 ? exitFocus() : beginFocus(nextIndex());
  firstButton.onclick = () => beginFocus(0);
  prevButton.onclick = () => navigate(-1);
  nextButton.onclick = () => navigate(1);
  lastButton.onclick = () => beginFocus(groups.length - 1);

  stepList.querySelectorAll(".study-step").forEach(step => {
    step.onclick = () => beginFocus(parseInt(step.dataset.index, 10));
  });
  groups.forEach((group, index) => {
    const toolsEl = document.querySelector(`.section-tools[data-section="${group.id}"]`);
    toolsEl.querySelector(".section-focus").onclick = () => beginFocus(index);
    toolsEl.querySelector(".section-done").onclick = () => {
      const key = sectionDoneKey(currentId, group.id);
      Store.set(key, !Store.get(key));
      saveResume(currentId, group);
      message.textContent = Store.get(key)
        ? "Bite complete. That is enough for this session."
        : "Section reopened.";
      refresh();
    };
    const footer = footers[index];
    footer.querySelector(".bite-prev").onclick = () => navigate(-1);
    footer.querySelector(".bite-next").onclick = () => navigate(1);
    footer.querySelector(".bite-last").onclick = () => beginFocus(groups.length - 1);
    footer.querySelector(".bite-exit").onclick = exitFocus;
  });

  refresh();
  saveResume(currentId, groups[nextIndex()]);
  updateNavigation(nextIndex());
  const api = {
    beginFocus,
    exitFocus,
    goFirst: () => beginFocus(0),
    goLast: () => beginFocus(groups.length - 1),
    isFocused: () => activeIndex >= 0,
    activeIndex: () => activeIndex,
    groups,
  };
  Study = api;

  const params = new URLSearchParams(location.search);
  if (params.get("focus") === "1") {
    const hash = location.hash.slice(1);
    const index = groups.findIndex(g => g.id === hash);
    afterLayout(() => beginFocus(index >= 0 ? index : nextIndex()));
  }
  return api;
}

function buildHomeStudy() {
  const hero = document.querySelector(".hero");
  if (!hero || document.querySelector(".home-study")) return;
  const resume = Store.get("resume");
  const chapter = CHAPTERS.find(c => c.id === resume?.chapterId) || CHAPTERS.find(c => !c.wip);
  if (!chapter) return;
  const sectionDone = Object.entries(Store.all()).filter(([k, v]) => k.startsWith("section:") && v).length;
  const exerciseDone = Object.entries(Store.all()).filter(([k, v]) => k.startsWith("exercise:") && v).length;
  const quizDone = Object.entries(Store.all()).filter(([k, v]) => k.startsWith("quiz:") && Number.isInteger(v)).length;
  const hasResume = resume && chapter.id === resume.chapterId;
  const href = hasResume
    ? `${chapter.f}?focus=1#${resume.sectionId}`
    : `${chapter.f}?focus=1`;

  const card = document.createElement("section");
  card.className = "home-study";
  card.setAttribute("aria-labelledby", "home-study-title");
  card.innerHTML = `
    <div class="home-study-mark">${hasResume ? "Resume" : "Start small"}</div>
    <div class="home-study-copy">
      <strong id="home-study-title">${hasResume ? chapter.id + ". " + chapter.t : "One focused bite is enough."}</strong>
      <span>${hasResume ? resume.section : "Open Chapter 1 with everything except the first section tucked away."}</span>
      <small>${sectionDone} sections · ${exerciseDone} exercises · ${quizDone} checks saved</small>
    </div>
    <a class="btn primary" href="${href}">${hasResume ? "Continue this bite" : "Start a 10-min bite"} &rarr;</a>`;
  hero.insertAdjacentElement("afterend", card);
}

function buildSidebar(currentId) {
  const sections = currentId ? tagSections() : [];

  const aside = document.createElement("aside");
  aside.className = "sidebar";
  aside.setAttribute("aria-label", "Course navigation");

  let html = `<div class="sb-search">
      <input type="search" id="sbq" placeholder="Filter chapters…" aria-label="Filter chapters">
    </div>`;

  for (const p of PARTS) {
    html += `<div class="sb-part" data-part="1"><div class="h">${p.n} &middot; ${p.t}</div>`;
    for (const c of p.ch) {
      const here = c.id === currentId;
      const dt = `data-t="${(c.id + " " + c.t + " " + c.s).toLowerCase()}"`;
      // A chapter marked `wip` is numbered and listed but not written yet.
      // Render it as inert text, never a link, so the outline stays honest and
      // the sidebar can never serve a 404. Remove `wip` to switch it on.
      html += c.wip
        ? `<span class="sb-ch wip" ${dt} title="Not written yet">
             <span class="n">${c.id}</span><span>${c.t}</span></span>`
        : `<a class="sb-ch ${isDone(c.id) ? "done" : ""} ${here ? "here" : ""}"
                  href="${c.f}" ${dt}>
                 <span class="n">${c.id}</span><span>${c.t}</span></a>`;
      if (here && sections.length) {
        html += `<nav class="sb-sections">` +
          sections.map(s => `<a class="sb-sec" href="#${s.id}">${s.label}</a>`).join("") +
          `</nav>`;
      }
    }
    html += `</div>`;
  }
  aside.innerHTML = html;
  document.body.appendChild(aside);
  document.body.classList.add("has-side");

  const scrim = document.createElement("div");
  scrim.className = "scrim";
  scrim.onclick = () => document.body.classList.remove("side-open");
  document.body.appendChild(scrim);

  // keep the current chapter visible in a long list
  const here = aside.querySelector(".sb-ch.here");
  if (here && typeof here.scrollIntoView === "function") {
    try { here.scrollIntoView({ block: "center" }); } catch (e) { /* non-fatal */ }
  }

  // type-to-filter
  const q = aside.querySelector("#sbq");
  q.addEventListener("input", () => {
    const term = q.value.trim().toLowerCase();
    aside.querySelectorAll(".sb-ch").forEach(a => {
      a.style.display = !term || a.dataset.t.includes(term) ? "" : "none";
    });
    aside.querySelectorAll(".sb-part").forEach(p => {
      const any = [...p.querySelectorAll(".sb-ch")].some(a => a.style.display !== "none");
      p.style.display = any ? "" : "none";
    });
    if (term) aside.querySelectorAll(".sb-sections").forEach(n => n.style.display = "none");
    else aside.querySelectorAll(".sb-sections").forEach(n => n.style.display = "");
  });

  return sections;
}

/* Highlight the section you are actually reading. */
function initScrollspy(sections, currentId) {
  if (!sections.length || typeof IntersectionObserver === "undefined") return;
  const links = new Map();
  document.querySelectorAll(".sb-sec").forEach(a =>
    links.set(a.getAttribute("href").slice(1), a));

  let active = null;
  const setActive = id => {
    if (id === active) return;
    active = id;
    links.forEach(a => a.classList.remove("active"));
    const a = links.get(id);
    if (a) a.classList.add("active");
    const section = sections.find(s => s.id === id);
    if (currentId && section) saveResume(currentId, section);
  };

  const io = new IntersectionObserver(entries => {
    // pick the heading nearest the top of the viewport that has passed it
    const visible = sections
      .map(s => ({ id: s.id, top: s.el.getBoundingClientRect().top }))
      .filter(s => document.getElementById(s.id)?.getClientRects().length && s.top < window.innerHeight * 0.4);
    if (visible.length) setActive(visible[visible.length - 1].id);
    else setActive(sections[0].id);
  }, { rootMargin: "-10% 0px -60% 0px", threshold: [0, 1] });

  sections.forEach(s => io.observe(s.el));
  window.addEventListener("scroll", () => {
    const visible = sections
      .map(s => ({ id: s.id, top: s.el.getBoundingClientRect().top }))
      .filter(s => document.getElementById(s.id)?.getClientRects().length && s.top < window.innerHeight * 0.4);
    setActive(visible.length ? visible[visible.length - 1].id : sections[0].id);
  }, { passive: true });
}

/* Margin notes are position:absolute with no `top`, so they keep their static
   position — but two notes placed close together resolve to overlapping spots.
   Push any collision down until the column is clean. */
function layoutMarginNotes() {
  const allNotes = [...document.querySelectorAll(".mn")];
  if (!allNotes.length) return;
  allNotes.forEach(n => (n.style.marginTop = ""));
  // Some notes (such as the chapter orientation) deliberately stay in the
  // prose flow. Collision-layout only the notes that actually occupy the
  // absolute margin column; one inline note must not disable the rest.
  const notes = allNotes.filter(n => getComputedStyle(n).position === "absolute");
  if (!notes.length) return;

  const main = document.querySelector("main");
  if (!main) return;

  // Elements that break out of the prose measure now reach INTO the margin
  // column, so a note must clear them vertically as well as clearing each other.
  // main is now breakout-wide, so compare against the PROSE measure instead
  const measurePx = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--measure")) * 16;
  const wide = [...main.children].filter(el => {
    if (el.classList.contains("mn")) return false;
    return el.offsetWidth > measurePx + 4;
  }).map(el => ({ top: el.offsetTop, bottom: el.offsetTop + el.offsetHeight }));

  const GAP = 18;
  let prevBottom = -Infinity;

  for (const n of notes) {
    const natural = n.offsetTop;
    let top = Math.max(natural, prevBottom + GAP);
    // walk down past any breakout block this note would sit across
    for (let i = 0; i < wide.length; i++) {
      const w = wide[i];
      if (top < w.bottom + GAP && top + n.offsetHeight > w.top - GAP) {
        top = w.bottom + GAP;
        i = -1;                       // re-check from the start after moving
      }
    }
    n.style.marginTop = top > natural ? (top - natural) + "px" : "";
    prevBottom = top + n.offsetHeight;
  }
}
window.layoutMarginNotes = layoutMarginNotes;

/* How far through this page you are. */
function initReadingBar() {
  const bar = document.createElement("div");
  bar.className = "readbar";
  document.body.appendChild(bar);
  const update = () => {
    const h = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + "%";
  };
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

/* Keyboard: ← / → move chapters, t toggles theme, / focuses the filter. */
function initKeyboard(currentId) {
  const i = CHAPTERS.findIndex(c => c.id === currentId);
  document.addEventListener("keydown", e => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === "/" && !typing) {
      e.preventDefault();
      const q = document.getElementById("sbq");
      if (q) { document.body.classList.add("side-open"); q.focus(); q.select(); }
      return;
    }
    if (e.key === "Escape") {
      if (Study?.isFocused()) { Study.exitFocus(); return; }
      document.body.classList.remove("side-open");
      if (typing) e.target.blur();
      return;
    }
    if (typing) return;
    if (e.key === "t") { Theme.toggle(); return; }
    if (i < 0) return;
    if (e.key === "ArrowLeft"  && CHAPTERS[i - 1]) location.href = CHAPTERS[i - 1].f;
    if (e.key === "ArrowRight" && CHAPTERS[i + 1]) location.href = CHAPTERS[i + 1].f;
  });
}

/* ------------------------------------------------------------------ boot */

function initChapter(id) {
  // Each step is independent. A failure in one (an unsupported API in an old
  // browser, say) must not silently take the quizzes and navigation with it.
  const step = (name, fn) => {
    try { return fn(); }
    catch (e) { console.error(`[course] ${name} failed:`, e); return undefined; }
  };

  step("layout",   () => buildLayout());
  step("topbar",   () => buildTopbar(id));
  const sections = step("sidebar", () => buildSidebar(id)) || [];
  step("code",     () => decorateCode());
  step("quizzes",  () => initQuizzes(id));
  step("exercises",() => initExerciseProgress(id));
  step("study",    () => buildStudyConsole(id, sections));
  step("chapnav",  () => buildChapterNav(id));
  step("scrollspy",() => initScrollspy(sections, id));
  step("readbar",  () => initReadingBar());
  step("keyboard", () => initKeyboard(id));
  step("viz",      () => window.initViz && window.initViz());
  step("sidebar-pref", () => applySidebarPref());
  step("margins",  () => layoutMarginNotes());

  // widgets and fonts change heights after first paint; re-flow the margin
  window.addEventListener("load", layoutMarginNotes);
  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { applySidebarPref(); layoutMarginNotes(); }, 100);
  });
}
