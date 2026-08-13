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
    `<a class="home" href="index.html">Transformers <i>From Scratch</i></a>` +
    (ch ? `<div class="crumb">${part ? part.n + " &middot; " : ""}Ch ${ch.id}</div>` : "") +
    `<div class="spacer"></div>` +
    `<div class="prog">${done}/${total}</div>` +
    `<button class="theme-toggle" type="button">${SUN}${MOON}</button>`;
  document.body.prepend(bar);
  bar.querySelector(".theme-toggle").onclick = Theme.toggle;
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

/* ------------------------------------------------------------------ boot */

function initChapter(id) {
  buildLayout();
  buildTopbar(id);
  decorateCode();
  initQuizzes();
  buildChapterNav(id);
  if (window.initViz) window.initViz();
}
