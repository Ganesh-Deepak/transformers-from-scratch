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
    `<div class="prog" title="Chapters marked complete">${done}/${total}</div>` +
    `<button class="theme-toggle" type="button">${SUN}${MOON}</button>`;
  document.body.prepend(bar);
  bar.querySelector(".theme-toggle").onclick = Theme.toggle;
  bar.querySelector(".sb-toggle").onclick = toggleSidebar;
  Theme.apply();
}

/* ----------------------------------------------------------- prev/next */

function buildChapterNav(currentId) {
  const i = CHAPTERS.findIndex(c => c.id === currentId);
  if (i < 0) return;
  const prev = CHAPTERS[i - 1], next = CHAPTERS[i + 1];
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
    location.reload();
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

function initQuizzes() {
  document.querySelectorAll(".quiz").forEach(quiz => {
    if (quiz.dataset.built) return;
    quiz.dataset.built = "1";
    const h = document.createElement("div");
    h.className = "qt";
    h.textContent = quiz.dataset.title || "Check yourself";
    quiz.prepend(h);

    quiz.querySelectorAll(".qitem").forEach(item => {
      const ans = parseInt(item.dataset.answer, 10);
      const opts = [...item.querySelectorAll(".opt")];
      const expl = item.querySelector(".expl");
      opts.forEach((o, idx) => {
        const mk = document.createElement("div");
        mk.className = "mk"; mk.textContent = "ABCDEF"[idx];
        o.prepend(mk);
        o.setAttribute("role", "button");
        o.setAttribute("tabindex", "0");
        const choose = () => {
          if (item.dataset.answered) return;
          item.dataset.answered = "1";
          opts.forEach((oo, jj) => {
            oo.classList.add("disabled");
            oo.setAttribute("tabindex", "-1");
            if (jj === ans) oo.classList.add("correct");
          });
          if (idx !== ans) o.classList.add("wrong");
          if (expl) expl.classList.add("show");
        };
        o.onclick = choose;
        o.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } };
      });
    });
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
      html += `<a class="sb-ch ${isDone(c.id) ? "done" : ""} ${here ? "here" : ""}"
                  href="${c.f}" data-t="${(c.id + " " + c.t + " " + c.s).toLowerCase()}">
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
function initScrollspy(sections) {
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
  };

  const io = new IntersectionObserver(entries => {
    // pick the heading nearest the top of the viewport that has passed it
    const visible = sections
      .map(s => ({ id: s.id, top: s.el.getBoundingClientRect().top }))
      .filter(s => s.top < window.innerHeight * 0.4);
    if (visible.length) setActive(visible[visible.length - 1].id);
    else setActive(sections[0].id);
  }, { rootMargin: "-10% 0px -60% 0px", threshold: [0, 1] });

  sections.forEach(s => io.observe(s.el));
  window.addEventListener("scroll", () => {
    const visible = sections
      .map(s => ({ id: s.id, top: s.el.getBoundingClientRect().top }))
      .filter(s => s.top < window.innerHeight * 0.4);
    setActive(visible.length ? visible[visible.length - 1].id : sections[0].id);
  }, { passive: true });
}

/* Margin notes are position:absolute with no `top`, so they keep their static
   position — but two notes placed close together resolve to overlapping spots.
   Push any collision down until the column is clean. */
function layoutMarginNotes() {
  const notes = [...document.querySelectorAll(".mn")];
  if (!notes.length) return;
  notes.forEach(n => (n.style.marginTop = ""));
  // only applies when the margin is actually a column (not folded inline)
  if (getComputedStyle(notes[0]).position !== "absolute") return;

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
  step("quizzes",  () => initQuizzes());
  step("chapnav",  () => buildChapterNav(id));
  step("scrollspy",() => initScrollspy(sections));
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
