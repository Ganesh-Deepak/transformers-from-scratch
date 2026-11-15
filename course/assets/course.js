/* ============================================================================
   course.js — theme, progress, layout scaffold, quiz engine, code highlighting.
   No dependencies.
   ========================================================================= */

/* -------------------------------------------------------------- course */
/* This file is shared verbatim by every course in the library (docs/PLATFORM.md).
   What differs per course lives in its own assets/chapters.js, as COURSE:
     slug   the registry id ("transformers")       store  its progress key
     title  shown in the masthead, HTML allowed     home   the course's index page
   Without a COURSE (an older chapters.js) this falls back to Transformers. */
const COURSE_CFG = Object.assign(
  { slug: "transformers", title: "Transformers <i>From Scratch</i>", store: "tfs-progress-v1" },
  typeof COURSE === "object" && COURSE ? COURSE : {});

/* ------------------------------------------------------------- storage */

/* Progress lives in ONE localStorage key holding a JSON object. The original
   version read it once at page load and then wrote its whole in-memory copy
   back on every change — so with two tabs open (a chapter and the home page is
   the normal case) the second tab to save silently erased everything the first
   had recorded since it loaded. Every write now re-reads the stored object,
   applies only its own change, and writes that back; and a `storage` event
   from another tab refreshes the in-memory copy. */
const Store = (() => {
  const KEY = COURSE_CFG.store;
  let ok = true;
  const read = () => {
    try { const v = JSON.parse(localStorage.getItem(KEY) || "{}"); return v && typeof v === "object" ? v : {}; }
    catch (e) { ok = false; return null; }
  };
  let mem = read() || {};
  const write = mutate => {
    if (ok) { const fresh = read(); if (fresh) mem = fresh; }
    mutate(mem);
    if (ok) { try { localStorage.setItem(KEY, JSON.stringify(mem)); } catch (e) { ok = false; } }
  };
  window.addEventListener("storage", e => {
    if (e.key === KEY || e.key === null) { const fresh = read(); if (fresh) mem = fresh; }
  });
  return {
    get: k => mem[k],
    set: (k, v) => write(m => { m[k] = v; }),
    remove: k => write(m => { delete m[k]; }),
    all: () => mem,
    reset: () => write(m => { Object.keys(m).forEach(k => delete m[k]); }),
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
    `<a class="home" href="index.html">${COURSE_CFG.title}</a>` +
    `<button class="lib-toggle" type="button" aria-expanded="false" aria-controls="library" ` +
      `aria-label="Library: every course">${ICON_LIB}<span>Library</span></button>` +
    (ch ? `<div class="crumb">${part ? part.n + " &middot; " : ""}Ch ${ch.id}</div>` : "") +
    `<div class="spacer"></div>` +
    (ch ? `<button class="focus-toggle" type="button" aria-pressed="false">Focus</button>` : "") +
    `<div class="prog" title="Chapters marked complete">${done}/${total}<span> chapters</span></div>` +
    `<button class="theme-toggle" type="button">${SUN}${MOON}</button>`;
  document.body.prepend(bar);
  bar.querySelector(".theme-toggle").onclick = Theme.toggle;
  bar.querySelector(".sb-toggle").onclick = toggleSidebar;
  Library.attach(bar.querySelector(".lib-toggle"));
  Theme.apply();
  if (!ch) buildHomeStudy();
}

/* ------------------------------------------------------------- library */
/* The course switcher (docs/PLATFORM.md section 4). Every course vendors the
   registry as assets/courses.js (AIE_COURSES, AIE_TRACKS). All courses live on
   one origin, so each course's progress key is readable from any of them. */

const ICON_LIB = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h3.5v16H4zM9.5 4H13v16H9.5z"/><path d="m15.2 5.1 3.3-.9 3.6 15.4-3.3.9z"/></svg>';

const Library = (() => {
  const SITE = "https://ganesh-deepak.github.io";
  /* Root-relative URLs work on the live site and on a local server that mirrors
     it (platform/serve.py). From file:// they would point at the disk root. */
  const abs = u => (location.protocol === "file:" ? SITE : "") + u;
  let btn = null, panel = null, loading = null;

  function load() {
    if (typeof AIE_COURSES !== "undefined") return Promise.resolve(true);
    if (loading) return loading;
    const me = [...document.scripts].find(s => /assets\/course\.js/.test(s.src));
    const src = me ? me.src.replace(/course\.js(\?.*)?$/, "courses.js$1") : "assets/courses.js";
    loading = new Promise(res => {
      const s = document.createElement("script");
      s.src = src; s.onload = () => res(true); s.onerror = () => res(false);
      document.head.appendChild(s);
    });
    return loading;
  }

  function progressOf(c) {
    if (c.slug === COURSE_CFG.slug) {
      const done = Object.entries(Store.all()).filter(([k, v]) => k.startsWith("done:") && v).length;
      return done;
    }
    try {
      const o = JSON.parse(localStorage.getItem(c.store) || "{}") || {};
      return Object.entries(o).filter(([k, v]) => k.startsWith("done:") && v).length;
    } catch (e) { return 0; }
  }

  const STATUS = { live: "", preview: "Outline", planned: "Planned" };

  function render() {
    const here = COURSE_CFG.slug;
    const row = c => {
      const n = progressOf(c), cur = c.slug === here;
      const go = c.status !== "planned";
      const meta = c.status === "live"
        ? `${n}/${c.chapters}<span> ch</span>`
        : `<span class="lib-status">${STATUS[c.status] || c.status}</span>`;
      const inner =
        `<span class="lib-n" style="--pc:var(--part-${c.ink})">${String(c.no).padStart(2, "0")}</span>` +
        `<span class="lib-t">${c.title}<small>${c.short}</small></span>` +
        `<span class="lib-m">${meta}</span>` +
        (c.status === "live" && c.chapters ? `<span class="lib-bar" style="--pc:var(--part-${c.ink})"><i style="width:${Math.min(100, n / c.chapters * 100)}%"></i></span>` : "");
      return go
        ? `<a class="lib-c${cur ? " here" : ""}" href="${abs(c.url)}"${cur ? ' aria-current="page"' : ""}>${inner}</a>`
        : `<div class="lib-c off" aria-disabled="true">${inner}</div>`;
    };
    panel.innerHTML =
      `<div class="lib-h"><span>The library</span><a href="${abs("/")}">All courses &amp; tracks &rarr;</a></div>` +
      `<div class="lib-list">${AIE_COURSES.map(row).join("")}</div>` +
      `<div class="lib-f">Progress is shared by every course in this browser.</div>`;
  }

  function open() {
    load().then(ok => {
      if (!panel) {
        panel = document.createElement("div");
        panel.id = "library"; panel.className = "library";
        panel.setAttribute("role", "dialog"); panel.setAttribute("aria-label", "Library");
        document.body.appendChild(panel);
      }
      if (ok && typeof AIE_COURSES !== "undefined") render();
      else panel.innerHTML = `<div class="lib-h"><span>The library</span></div>` +
        `<div class="lib-f">The course list could not be loaded. <a href="${SITE}/">Open the library &rarr;</a></div>`;
      panel.hidden = false;
      const r = btn.getBoundingClientRect(), w = panel.offsetWidth;
      panel.style.left = Math.max(8, Math.min(r.left, innerWidth - w - 8)) + "px";
      panel.style.top = (r.bottom + 10) + "px";
      btn.setAttribute("aria-expanded", "true");
      (panel.querySelector(".lib-c.here") || panel.querySelector("a"))?.focus({ preventScroll: true });
    });
  }
  function close(refocus) {
    if (!panel || panel.hidden) return;
    panel.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    if (refocus) btn.focus();
  }
  const isOpen = () => panel && !panel.hidden;

  function attach(b) {
    if (!b) return;
    btn = b;
    btn.onclick = e => { e.stopPropagation(); isOpen() ? close(true) : open(); };
    document.addEventListener("click", e => {
      if (isOpen() && !panel.contains(e.target) && e.target !== btn) close(false);
    });
    document.addEventListener("keydown", e => { if (e.key === "Escape" && isOpen()) close(true); });
    load().then(markXrefs);
  }

  /* Cross-course links: <a class="xref" data-course="slug" data-ch="15">.
     Mark the ones whose target chapter this reader has finished. */
  function markXrefs() {
    if (typeof AIE_COURSES === "undefined") return;
    document.querySelectorAll("a.xref[data-course][data-ch]").forEach(a => {
      const c = AIE_COURSES.find(x => x.slug === a.dataset.course);
      if (!c) return;
      if (location.protocol === "file:" && a.getAttribute("href").startsWith("/")) a.href = SITE + a.getAttribute("href");
      let o = {};
      try { o = c.slug === COURSE_CFG.slug ? Store.all() : JSON.parse(localStorage.getItem(c.store) || "{}") || {}; } catch (e) { /* blocked */ }
      if (o["done:" + a.dataset.ch]) { a.classList.add("done"); a.title = "You finished this chapter"; }
      if (!a.title) a.title = c.title;
    });
  }

  return { attach, open, close, load, progressOf };
})();

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
    const unlockers = [];
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

      /* the inverse of lockChoice: back to the unanswered state */
      unlockers[itemIndex] = () => {
        delete item.dataset.answered;
        chosen[itemIndex] = null;
        opts.forEach((oo, jj) => {
          oo.classList.remove("disabled", "correct", "wrong");
          oo.removeAttribute("aria-disabled");
          oo.setAttribute("aria-checked", "false");
          oo.setAttribute("tabindex", jj === 0 ? "0" : "-1");
        });
        if (expl) expl.classList.remove("show");
        announce.textContent = "";
        Store.remove(key);
      };

      const saved = Store.get(key);
      if (Number.isInteger(saved) && saved >= 0 && saved < opts.length) lockChoice(saved, false);
    });

    /* Retry used to call location.reload(). In focus mode that threw the
       reader out of their bite, restarted the sprint, reset every widget and
       scrolled to the top. Reset the quiz in place instead. */
    reset.onclick = () => {
      unlockers.forEach(u => u && u());
      refreshStatus();
      const first = quiz.querySelector(".opt");
      if (first) first.focus();
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

/* the narrow-screen drawer slides shut when dismissed, like it slid open */
function closeDrawer() {
  if (!document.body.classList.contains("side-open")) return;
  document.body.classList.add("side-anim");
  clearTimeout(toggleSidebar.t);
  toggleSidebar.t = setTimeout(() => document.body.classList.remove("side-anim"), 400);
  document.body.classList.remove("side-open");
}

function toggleSidebar() {
  // animate this one change only (see .wrap in style.css)
  document.body.classList.add("side-anim");
  clearTimeout(toggleSidebar.t);
  toggleSidebar.t = setTimeout(() => document.body.classList.remove("side-anim"), 400);
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
  /* The countdown is a DEADLINE, not a count of setInterval ticks. Browsers
     throttle intervals in background tabs (Chrome eventually to once a
     minute), so counting ticks made a 10-minute sprint in a background tab
     last far longer than 10 minutes. The interval now only repaints. */
  let deadline = 0;
  const remaining = () => Math.max(0, Math.ceil((deadline - Date.now()) / 1000));

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

  /* One place that owns the countdown, so entering a bite, changing the
     duration mid-sprint, and restarting an expired sprint cannot drift apart.
     Previously this lived inline in beginFocus and ran only on first entry,
     which is why the duration select saved a new value that nothing acted on
     until you left focus mode and came back. */
  const tick = () => {
    secondsLeft = remaining();
    showTimer();
    if (secondsLeft <= 0) {
      stopTimer();
      consoleEl.classList.add("sprint-finished");
      message.textContent = "Sprint complete. Stop guilt-free, or finish the current thought.";
    }
  };
  const startSprint = () => {
    stopTimer();
    consoleEl.classList.remove("sprint-finished");
    deadline = Date.now() + parseInt(duration.value, 10) * 60 * 1000;
    secondsLeft = remaining();
    showTimer();
    timer = setInterval(tick, 1000);
  };
  // returning to a backgrounded tab: repaint at once rather than at the next tick
  document.addEventListener("visibilitychange", () => { if (timer && !document.hidden) tick(); });

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
    consoleEl.classList.remove("sprint-finished");
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
    /* A sprint is a budget across bites, not per bite, so moving to the next
       bite keeps counting. Two cases restart it: entering focus mode, and
       continuing after the budget has already run out — if you have decided
       to keep going, a fresh sprint is more honest than a stuck 0:00 and a
       "sprint complete" banner that no longer describes anything. */
    if (entering || remaining() <= 0) startSprint();
    message.textContent = `Focused on “${groups[index].label}”. Everything else is temporarily tucked away.`;
    refresh();
    updateNavigation(index);
    scrollIntoBite(consoleEl, groups[index] && groups[index].el);
  };

  const navigate = delta => {
    const base = activeIndex >= 0 ? activeIndex : nextIndex();
    beginFocus(Math.max(0, Math.min(groups.length - 1, base + delta)));
  };

  duration.onchange = () => {
    Store.set("study:minutes", parseInt(duration.value, 10));
    // Mid-sprint the reader is telling us how long they now have. Honour it
    // immediately rather than at the next entry into focus mode.
    if (activeIndex >= 0) startSprint();
  };
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
  /* Not saving a resume point here on purpose: merely opening a chapter for a
     quick look used to move the home page's "continue" away from the bite you
     were actually working in. Resume is saved on engagement — focusing a
     bite, marking a section, or scrolling through one (initScrollspy). */
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
      <span>${hasResume ? resume.section : "Open the first chapter with everything except its first section tucked away."}</span>
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

  /* Each part is a collapsible group with its own ink colour (--pc, see
     .sb-part[data-pi] in style.css): a solid header band you click to fold
     the part away, chapters striped in that colour underneath. The current
     chapter is a raised card whose section list folds with its own button.
     By default only the part you are reading is open (on the index, all of
     them), so the list stays short. Whatever you open or fold is remembered
     (tfs-side-parts / tfs-side-secs); the part you are reading always opens.
     Part folding is per course (part 2 means something different in each);
     Transformers keeps its original key so existing readers lose nothing. */
  const PART_KEY = COURSE_CFG.slug === "transformers" ? "tfs-side-parts" : COURSE_CFG.slug + "-side-parts";
  const store = (k, v) => { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } };
  let partState = {};
  try { partState = JSON.parse(store(PART_KEY) || "{}"); } catch (e) { partState = {}; }
  if (!partState || Array.isArray(partState) || typeof partState !== "object") partState = {};
  const secsFolded = store("tfs-side-secs") === "closed";
  const CARET = `<svg class="caret" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5l3 3 3-3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

  PARTS.forEach((p, pi) => {
    const hasHere = p.ch.some(c => c.id === currentId);
    const open = hasHere || (partState[pi] ? partState[pi] === "open" : !currentId);
    const done = p.ch.filter(c => isDone(c.id)).length;
    const num = p.n.replace(/^Part\s*/i, "");
    html += `<section class="sb-part${hasHere ? " has-here" : ""}" data-part="1" data-pi="${pi}">
      <button class="sb-part-h" type="button" aria-expanded="${open}" aria-controls="sbp-${pi}">
        <span class="pn">${num}</span><span class="pt">${p.t}</span>
        <span class="pc" title="${done} of ${p.ch.length} done">${done}/${p.ch.length}</span>${CARET}
      </button>
      <div class="sb-part-body" id="sbp-${pi}"${open ? "" : " hidden"}>`;
    for (const c of p.ch) {
      const here = c.id === currentId;
      const dt = `data-t="${(c.id + " " + c.t + " " + c.s).toLowerCase()}"`;
      // A chapter marked `wip` is numbered and listed but not written yet.
      // Render it as inert text, never a link, so the outline stays honest and
      // the sidebar can never serve a 404. Remove `wip` to switch it on.
      const link = c.wip
        ? `<span class="sb-ch wip" ${dt} title="Not written yet">
             <span class="n">${c.id}</span><span>${c.t}</span></span>`
        : `<a class="sb-ch ${isDone(c.id) ? "done" : ""} ${here ? "here" : ""}"
                  href="${c.f}" ${dt}${here ? ' aria-current="page"' : ""}>
                 <span class="n">${c.id}</span><span>${c.t}</span></a>`;
      if (here && sections.length) {
        html += `<div class="sb-here">
          <div class="sb-here-row">${link}
            <button class="sb-sec-toggle" type="button" aria-expanded="${!secsFolded}" aria-controls="sbs"
              title="Show or hide this chapter's sections">${CARET}<span class="vh">Sections</span></button></div>
          <nav class="sb-sections" id="sbs"${secsFolded ? " hidden" : ""}>` +
          sections.map(s => `<a class="sb-sec" href="#${s.id}">${s.label}</a>`).join("") +
          `</nav></div>`;
      } else {
        html += link;
      }
    }
    html += `</div></section>`;
  });
  aside.innerHTML = html;

  aside.querySelectorAll(".sb-part-h").forEach(btn => {
    btn.addEventListener("click", () => {
      const body = document.getElementById(btn.getAttribute("aria-controls"));
      const open = btn.getAttribute("aria-expanded") !== "true";
      btn.setAttribute("aria-expanded", String(open));
      body.hidden = !open;
      const pi = +btn.closest(".sb-part").dataset.pi;
      partState[pi] = open ? "open" : "closed";
      store(PART_KEY, JSON.stringify(partState));
    });
  });
  const secBtn = aside.querySelector(".sb-sec-toggle");
  if (secBtn) secBtn.addEventListener("click", () => {
    const open = secBtn.getAttribute("aria-expanded") !== "true";
    secBtn.setAttribute("aria-expanded", String(open));
    aside.querySelector("#sbs").hidden = !open;
    store("tfs-side-secs", open ? "open" : "closed");
  });
  document.body.appendChild(aside);
  document.body.classList.add("has-side");

  const scrim = document.createElement("div");
  scrim.className = "scrim";
  scrim.onclick = closeDrawer;
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
      // a match inside a folded part must be visible; clearing restores the fold
      const body = p.querySelector(".sb-part-body"), btn = p.querySelector(".sb-part-h");
      body.hidden = term ? !any : btn.getAttribute("aria-expanded") !== "true";
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
  /* Only a person scrolling counts as reading. The observer also fires on
     load, and programmatic scrolls (anchors, entering a bite) fire scroll
     events, so neither may move the resume pointer by itself. */
  let engaged = false;
  ["wheel", "touchmove", "keydown", "pointerdown"].forEach(t =>
    window.addEventListener(t, () => { engaged = true; }, { passive: true, once: true }));
  const setActive = id => {
    if (id === active) return;
    active = id;
    links.forEach(a => a.classList.remove("active"));
    const a = links.get(id);
    if (a) a.classList.add("active");
    const section = sections.find(s => s.id === id);
    if (currentId && section && engaged) saveResume(currentId, section);
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
  allNotes.forEach(n => { n.style.marginTop = ""; n.style.top = ""; });
  // Some notes (such as the chapter orientation) deliberately stay in the
  // prose flow. Collision-layout only the notes that actually occupy the
  // absolute margin column; one inline note must not disable the rest.
  const notes = allNotes.filter(n => getComputedStyle(n).position === "absolute");
  if (!notes.length) return;
  const isNote = el => notes.includes(el);

  const main = document.querySelector("main");
  if (!main) return;

  // Elements that break out of the prose measure now reach INTO the margin
  // column, so a note must clear them vertically as well as clearing each other.
  // main is now breakout-wide, so compare against the PROSE measure instead
  // --measure is a clamp() now, so measure it instead of parsing it
  const prose = main.querySelector(":scope > p");
  const measurePx = prose ? prose.offsetWidth : 592;
  const wide = [...main.children].filter(el => {
    if (el.classList.contains("mn")) return false;
    return el.offsetWidth > measurePx + 4;
  }).map(el => ({ top: el.offsetTop, bottom: el.offsetTop + el.offsetHeight }));

  const GAP = 18;
  let prevBottom = -Infinity;

  for (const n of notes) {
    // Where the note would sit in the flow: just below the nearest in-flow
    // block before it. Derived from that block, NOT from n.offsetTop: an
    // absolutely positioned box's static position is not reliably updated
    // when its own margin changes, so reading it back after a previous pass
    // returned stale numbers and notes landed on top of code blocks.
    let prev = n.previousElementSibling;
    while (prev && isNote(prev)) prev = prev.previousElementSibling;
    const ownMargin = parseFloat(getComputedStyle(n).marginTop) || 0;
    const natural = (prev ? prev.offsetTop + prev.offsetHeight + (parseFloat(getComputedStyle(prev).marginBottom) || 0) : 0) + ownMargin;
    let top = Math.max(natural, prevBottom + GAP);
    // walk down past any breakout block this note would sit across
    for (let i = 0; i < wide.length; i++) {
      const w = wide[i];
      if (top < w.bottom + GAP && top + n.offsetHeight > w.top - GAP) {
        top = w.bottom + GAP;
        i = -1;                       // re-check from the start after moving
      }
    }
    n.style.top = (top - ownMargin) + "px";      // explicit, so nothing is read back
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
    /* Arrow keys are also how you move between quiz options, between the axes
       of a shape badge, and between the bits of the number explorer. Those
       handlers call preventDefault, but this listener used to ignore that —
       so pressing Right inside a quiz ALSO jumped to the next chapter. A key
       something else has already handled, or any key pressed while focus is
       inside an interactive component, is not a page shortcut. */
    if (e.defaultPrevented && e.key !== "Escape") return;
    const inWidget = e.target.closest &&
      e.target.closest('[role="radio"], [role="radiogroup"], .shp, [data-viz], .study-console, .bite-footer');
    if (inWidget && e.key !== "Escape" && e.key !== "/") return;

    if (e.key === "/" && !typing) {
      e.preventDefault();
      const q = document.getElementById("sbq");
      if (q) { document.body.classList.add("side-open"); q.focus(); q.select(); }
      return;
    }
    if (e.key === "Escape") {
      if (Study?.isFocused()) { Study.exitFocus(); return; }
      closeDrawer();
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
  // `load` is not the end: web fonts swap in after it, widgets redraw, and
  // <details> open. Any of those moves the blocks a note must clear, and a
  // note laid out against stale positions sits on top of a code block
  // (tests/shoot.js found 31 such overlaps). So re-lay whenever main changes
  // size. Notes are absolutely positioned, so moving them never resizes main:
  // no feedback loop.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(layoutMarginNotes);
  const mainEl = document.querySelector("main");
  if (mainEl && "ResizeObserver" in window) {
    let raf = 0;
    new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(layoutMarginNotes);
    }).observe(mainEl);
  }
  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { applySidebarPref(); layoutMarginNotes(); }, 100);
  });
}
