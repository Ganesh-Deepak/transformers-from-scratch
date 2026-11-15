/* Headless UI regression test — contrast, theming, and page structure.

     cd tests && npm install jsdom && node check_ui.js

   Checks every colour pair against WCAG in BOTH themes, and confirms each page
   builds its layout, topbar, theme toggle, copy buttons, nav and quiz markers. */
const fs = require("fs");
const path = require("path");

let JSDOM, VirtualConsole;
try {
  ({ JSDOM, VirtualConsole } = require("jsdom"));
} catch (e) {
  console.error(
    "\n  This test needs jsdom, which is not part of requirements.txt:\n\n" +
    "      cd tests\n      npm install jsdom\n      node check_ui.js\n"
  );
  process.exit(2);
}

const ROOT = path.join(__dirname, "..", "course");

function load(file, theme) {
  const html = fs.readFileSync(path.join(ROOT, file), "utf8");
  const vc = new VirtualConsole();
  const errors = [];
  vc.on("jsdomError", e => errors.push(e.message.split("\n")[0]));
  vc.on("error", (...a) => errors.push("console.error: " + a.join(" ")));

  const dom = new JSDOM(html, {
    runScripts: "dangerously",
    resources: "usable",
    url: "file:///" + path.join(ROOT, file).replace(/\\/g, "/"),
    pretendToBeVisual: true,
    virtualConsole: vc,
    beforeParse(win) {
      // jsdom has no matchMedia; supply one so the theme code runs.
      win.matchMedia = q => ({
        matches: theme === "dark" && /dark/.test(q),
        media: q, addEventListener() {}, removeEventListener() {},
        addListener() {}, removeListener() {},
      });
      const store = {};
      Object.defineProperty(win, "localStorage", {
        value: {
          getItem: k => (k === "tfs-theme" ? theme : store[k] ?? null),
          setItem: (k, v) => { store[k] = v; },
          removeItem: k => { delete store[k]; },
        },
      });
    },
  });
  // External <script src> load asynchronously in jsdom — wait for them.
  const ready = new Promise(res => {
    if (dom.window.document.readyState === "complete") return res();
    dom.window.addEventListener("load", () => res());
    setTimeout(res, 4000);            // safety net
  });
  return { dom, errors, ready };
}

// jsdom's cssstyle does not resolve var(); read the token values from the
// stylesheet text for the requested theme instead.
const css = fs.readFileSync(path.join(ROOT, "assets", "style.css"), "utf8");
function tokensFor(theme) {
  const nocomment = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const sel = theme === "dark" ? ':root[data-theme="dark"]' : ':root[data-theme="light"]';
  const i = nocomment.indexOf(sel);
  const body = nocomment.slice(nocomment.indexOf("{", i) + 1, nocomment.indexOf("}", i));
  const out = {};
  for (const m of body.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/g)) out[m[1]] = m[2].trim();
  return out;
}

function relLum(hex) {
  const c = hex.replace("#", "");
  const v = [0, 2, 4].map(i => {
    let x = parseInt(c.substr(i, 2), 16) / 255;
    return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
}
const contrast = (a, b) => {
  const [x, y] = [relLum(a), relLum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};

let fail = 0;
const ok = (cond, label, extra = "") => {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${label}${extra ? "  " + extra : ""}`);
  if (!cond) fail++;
};

for (const theme of ["light", "dark"]) {
  console.log(`\n=== ${theme.toUpperCase()} =================================`);
  const t = tokensFor(theme);

  console.log(" contrast ratios (WCAG: 4.5 body, 3.0 large/UI)");
  const pairs = [
    ["ink on paper", t["--ink"], t["--paper"], 4.5],
    ["ink-2 on paper", t["--ink-2"], t["--paper"], 4.5],
    // ink-3 carries real reading text (margin notes, captions, labels), so
    // it is held to the body-text bar, not the large-text one. The 3.0 bar it
    // used to have let 10px labels ship at 4.7:1 on a dotted ground.
    ["ink-3 on paper", t["--ink-3"], t["--paper"], 4.5],
    ["ink-4 on paper", t["--ink-4"], t["--paper"], 3.0],
    // the reading sheet sits under every line of chapter text
    ["ink on sheet", t["--ink"], t["--sheet"], 4.5],
    ["ink-3 on sheet", t["--ink-3"], t["--sheet"], 4.5],
    ["accent on sheet", t["--accent"], t["--sheet"], 4.5],
    ["ink-3 on surface", t["--ink-3"], t["--surface"], 4.5],
    ...["1","2","3","4","5"].map(n => ["dim-" + n + " on sheet", t["--dim-" + n], t["--sheet"], 3.0]),
    ["accent on paper", t["--accent"], t["--paper"], 4.5],
    ["accent on surface", t["--accent"], t["--surface"], 4.5],
    ["dim-1 on paper", t["--dim-1"], t["--paper"], 3.0],
    ["dim-2 on paper", t["--dim-2"], t["--paper"], 3.0],
    ["dim-3 on paper", t["--dim-3"], t["--paper"], 3.0],
    ["dim-4 on paper", t["--dim-4"], t["--paper"], 3.0],
    ["dim-5 on paper", t["--dim-5"], t["--paper"], 3.0],
    ["dim-5 on surface", t["--dim-5"], t["--surface"], 3.0],
    ["ok on ok-bg", t["--ok"], t["--ok-bg"], 4.5],
    ["warn on warn-bg", t["--warn"], t["--warn-bg"], 4.5],
    ["stop on stop-bg", t["--stop"], t["--stop-bg"], 4.5],
  ];
  for (const [label, fg, bg, min] of pairs) {
    const r = contrast(fg, bg);
    ok(r >= min, label.padEnd(20), `${r.toFixed(2)}:1 (need ${min})`);
  }

  /* The axis hues must be mutually distinguishable — for EVERY reader. This
     used to check HSV hue angle >= 30deg, and that passed a palette in which
     feature and head-dim were the same grey to a deuteranope (OKLab dE 1.7).
     Hue angle is not perceptual and knows nothing about colour blindness.
     These are the dataviz validator's criteria instead: OKLab distance x100,
     under the Machado-2009 protan and deutan simulations, over ALL pairs
     (all five axes appear together inside one shape). */
  const dims = ["--dim-1", "--dim-2", "--dim-3", "--dim-4", "--dim-5"].map(k => t[k]);
  const s2l = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  const linRGB = hex => [1, 3, 5].map(i => s2l(parseInt(hex.slice(i, i + 2), 16) / 255));
  const MACHADO = {
    protan: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    deutan: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
  };
  const oklab = ([r, g, b]) => {
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const q = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * q,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * q,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * q];
  };
  const sim = (hex, kind) => { const v = linRGB(hex); if (!kind) return v;
    return MACHADO[kind].map(r => Math.max(0, Math.min(1, r[0] * v[0] + r[1] * v[1] + r[2] * v[2]))); };
  const dE = (a, b, kind) => { const x = oklab(sim(a, kind)), y = oklab(sim(b, kind));
    return 100 * Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]); };
  let cvd = [99, ""], normal = [99, ""];
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) {
    const tag = `dim-${i + 1}/dim-${j + 1}`;
    for (const kind of ["protan", "deutan"]) { const d = dE(dims[i], dims[j], kind); if (d < cvd[0]) cvd = [d, `${tag} ${kind}`]; }
    const d = dE(dims[i], dims[j]); if (d < normal[0]) normal = [d, tag];
  }
  ok(cvd[0] >= 8, "axis hues distinct to colour-blind readers (all pairs, dE >= 8)",
     `worst ${cvd[1]} ${cvd[0].toFixed(1)}`);
  ok(normal[0] >= 15, "axis hues distinct with normal vision (all pairs, dE >= 15)",
     `worst ${normal[1]} ${normal[0].toFixed(1)}`);
  const chroma = hex => { const [, a, b] = oklab(linRGB(hex)); return Math.hypot(a, b); };
  const grey = dims.filter(h => chroma(h) < 0.10);
  ok(!grey.length, "no axis hue reads as grey (OKLCH chroma >= 0.10)", grey.join(" "));
  const near = dims.map((h, i) => [i + 1, Math.min(dE(h, t["--accent"]), dE(h, t["--stamp"]))]).filter(x => x[1] < 15);
  ok(!near.length, "axis hues stay clear of the accent and the stamp (dE >= 15)",
     near.map(x => `dim-${x[0]} ${x[1].toFixed(1)}`).join(" "));
  let minDelta = 99;
  for (let i = 0; i < dims.length; i++)
    for (let j = i + 1; j < dims.length; j++)
      minDelta = Math.min(minDelta, Math.abs(relLum(dims[i]) - relLum(dims[j])));
  console.log(` dimension palette: ${dims.length} hues, min luminance delta ${minDelta.toFixed(3)}`);
  ok(t["--accent"] !== undefined && !dims.includes(t["--accent"]),
     "accent excluded from dimension hues");
}

/* --------------------------------------------------------------------------
   Manifest integrity.

   These exist because the previous version only compared the sidebar's chapter
   COUNT against the manifest — it read its expectation from the same file it
   was testing, so it happily passed while three manifest entries pointed at
   files that did not exist, and while `id` and filename had drifted apart
   during a renumber. A test whose expected value comes from the thing under
   test proves nothing.
   -------------------------------------------------------------------------- */
console.log("\n=== AXIS MODEL ==================================");
{
  /* Colour means WHICH AXIS. The widgets used to colour by tensor identity
     (Q orange, K blue, V magenta, O green) through colour keys named q/k/v/o,
     so "orange" meant batch in the prose and queries in the diagrams. Those
     keys were removed: a widget must name the axis it draws, or use a non-axis
     colour. This keeps them from quietly coming back. */
  const viz = fs.readFileSync(path.join(ROOT, "assets", "viz.js"), "utf8");
  const old = viz.match(/\bC\.[qkvo]\b/g) || [];
  ok(old.length === 0, "no widget uses the old Q/K/V/O colour keys", old.length ? old.length + " uses" : "");
  ok(!/function hues\(/.test(viz), "no index palette built from the axis colours");
}

console.log("\n=== MANIFEST INTEGRITY ==========================");
{
  const src = fs.readFileSync(path.join(ROOT, "assets", "chapters.js"), "utf8");
  const entries = [...src.matchAll(/id:\s*"(\d+)",\s*f:\s*"([^"]+)"(,\s*wip:\s*true)?/g)]
    .map(m => ({ id: m[1], file: m[2], wip: !!m[3] }));

  ok(entries.length > 0, "manifest parses", `${entries.length} entries`);

  // Chapter 00 (number formats) sits before 01, so the sequence may start at
  // either 00 or 01 -- but must then run without gaps or repeats.
  const start = entries.length && entries[0].id === "00" ? 0 : 1;
  const seq = entries.every((e, i) => e.id === String(i + start).padStart(2, "0"));
  ok(seq, "ids run " + String(start).padStart(2, "0") + "..NN with no gaps or repeats",
     entries.map(e => e.id).join(","));

  const mismatched = entries.filter(e => e.file.slice(2, 4) !== e.id);
  ok(mismatched.length === 0, "every id matches its filename",
     mismatched.map(e => `${e.id}->${e.file}`).join(" "));

  const missing = entries.filter(e => !e.wip && !fs.existsSync(path.join(ROOT, e.file)));
  ok(missing.length === 0, "every linked chapter file exists",
     missing.map(e => e.file).join(" "));

  const wip = entries.filter(e => e.wip);
  if (wip.length) {
    console.log(`  NOTE  ${wip.length} chapter(s) marked wip (listed, not linked): ` +
                wip.map(e => e.id).join(", "));
    const stale = wip.filter(e => fs.existsSync(path.join(ROOT, e.file)));
    ok(stale.length === 0, "no wip flag left on a chapter that now exists",
       stale.map(e => e.file).join(" "));
  }

  // every local href across the whole course must resolve
  const dead = [];
  for (const f of fs.readdirSync(ROOT).filter(f => f.endsWith(".html"))) {
    const html = fs.readFileSync(path.join(ROOT, f), "utf8");
    // root-relative hrefs ("/llm-evaluation/...") are other courses on the same origin:
    // the hub's tools/check_platform.py resolves those against every course
    for (const m of html.matchAll(/href="(?!\/)([^"#:]+\.html)(?:#[^"]*)?"/g)) {
      if (!fs.existsSync(path.join(ROOT, m[1]))) dead.push(`${f} -> ${m[1]}`);
    }
  }
  ok(dead.length === 0, "every chapter-to-chapter link resolves", dead.join(" "));
}

(async () => {
console.log("\n=== PAGE STRUCTURE ==============================");
const manifestFiles = [...fs.readFileSync(path.join(ROOT, "assets", "chapters.js"), "utf8")
  .matchAll(/id:\s*"\d+",\s*f:\s*"([^"]+)"/g)].map(m => m[1]);
for (const file of ["index.html", ...manifestFiles]) {
  const { dom, errors, ready } = load(file, "dark");
  await ready;
  const d = dom.window.document;
  console.log(` ${file}`);
  ok(errors.length === 0, "no script errors", errors.slice(0, 2).join(" | "));
  ok(!!d.querySelector(".wrap > main"), "main wrapped for margin layout");
  ok(!!d.querySelector(".topbar"), "topbar built");
  ok(!!d.querySelector(".theme-toggle"), "theme toggle present");
  ok(d.documentElement.getAttribute("data-theme") === "dark", "data-theme applied");
  const pres = d.querySelectorAll("pre > code");
  ok(pres.length === 0 || d.querySelectorAll(".copy-btn").length === pres.length,
     "copy buttons on every code block", `${d.querySelectorAll(".copy-btn").length}/${pres.length}`);
  // navigation
  ok(!!d.querySelector(".sidebar"), "sidebar built");
  // read the expected count from the manifest so this can never go stale
  const manifest = fs.readFileSync(path.join(ROOT, "assets", "chapters.js"), "utf8");
  const nChapters = (manifest.match(/id:\s*"\d+"/g) || []).length;
  ok(d.querySelectorAll(".sb-ch").length === nChapters,
     `all ${nChapters} chapters in sidebar`, `${d.querySelectorAll(".sb-ch").length}`);
  // a wip chapter must be listed but must NOT be clickable
  const wipLinks = [...d.querySelectorAll("a.sb-ch.wip")];
  ok(wipLinks.length === 0, "no wip chapter is rendered as a link",
     wipLinks.map(a => a.getAttribute("href")).join(" "));
  ok(!!d.querySelector("#sbq"), "chapter filter present");
  ok(!!d.querySelector(".readbar"), "reading-progress bar present");
  ok(!!d.querySelector(".sb-toggle"), "drawer toggle present (narrow screens)");

  if (file === "index.html") {
    ok(!!d.querySelector(".home-study"), "short-session resume card built");
  }

  if (file !== "index.html") {
    ok(!!d.querySelector(".chapnav"), "chapter nav built");
    ok(d.querySelectorAll(".opt .mk").length > 0, "quiz markers built");
    ok(!!d.querySelector(".sb-ch.here"), "current chapter marked in sidebar");
    const secs = d.querySelectorAll(".sb-sec");
    ok(secs.length > 0, "in-chapter section list built", `${secs.length} sections`);
    // every section link must resolve to a real heading on the page
    const bad = [...secs].filter(a => !d.getElementById(a.getAttribute("href").slice(1)));
    ok(bad.length === 0, "every section link resolves",
       bad.map(a => a.getAttribute("href")).join(" "));
    ok(!!d.querySelector(".study-console"), "short-session console built");
    ok(!!d.querySelector(".focus-toggle"), "focus-mode toggle present");
    ok(d.querySelectorAll(".study-step").length === secs.length,
       "every section is a resumable bite", `${d.querySelectorAll(".study-step").length}/${secs.length}`);
    ok(d.querySelectorAll(".section-tools").length === secs.length,
       "every section has completion controls", `${d.querySelectorAll(".section-tools").length}/${secs.length}`);
    ok(d.querySelectorAll(".bite-footer").length === secs.length,
       "every bite has bottom navigation", `${d.querySelectorAll(".bite-footer").length}/${secs.length}`);
    ok(!!d.querySelector(".study-nav-prev") && !!d.querySelector(".study-nav-next") &&
       !!d.querySelector(".study-nav-last") && !!d.querySelector(".study-nav-first"),
       "first, previous, next, and last bite controls present");
    ok(d.querySelectorAll(".exercise .ex-check").length === d.querySelectorAll(".exercise").length,
       "every exercise is independently trackable",
       `${d.querySelectorAll(".exercise .ex-check").length}/${d.querySelectorAll(".exercise").length}`);
    ok([...d.querySelectorAll(".opts")].every(x => x.getAttribute("role") === "radiogroup"),
       "quiz option groups expose radio semantics");
    ok([...d.querySelectorAll(".opt")].every(x => x.getAttribute("role") === "radio"),
       "quiz choices expose radio semantics");
  }
  dom.window.close();
}

/* ---- CONTENT INTEGRITY: the book must not promise what does not exist ----
   13 chapters once pointed at companion notebooks that were never written, and
   Ch 10 "built on" Ch 11. A reference to a repo file must resolve, unless it is
   explicitly marked as planned (<code class="planned">). Links between pages
   must resolve. "Builds on" may only point backwards. */
{
  console.log("\n=== CONTENT INTEGRITY ==============================");
  const COURSE = ROOT, REPO = path.join(ROOT, "..");
  const names = new Set(fs.readdirSync(COURSE).filter(f => f.endsWith(".html")));
  const missing = [], broken = [], forward = [];
  for (const f of [...names].sort()) {
    const s = fs.readFileSync(path.join(COURSE, f), "utf8");
    const unplanned = s.replace(/<code class="planned">[^<]*<\/code>/g, "");
    for (const m of unplanned.matchAll(/(notebooks\/[\w\-.]+\.ipynb|tfs\/[\w\-.]+\.py|tests\/[\w\-.]+\.py)/g))
      if (!fs.existsSync(path.join(REPO, m[1]))) missing.push(f + " -> " + m[1]);
    for (const m of s.matchAll(/href="(?!\/)([^"#:]+\.html)/g))      // cross-course: check_platform.py
      if (!names.has(m[1])) broken.push(f + " -> " + m[1]);
    const me = /^ch(\d\d)/.exec(f), b = /<span class="k">Builds on<\/span>([\s\S]*?)<div class="sep">/.exec(s);
    if (me && b) for (const m of b[1].matchAll(/ch(\d\d)-/g))
      if (+m[1] >= +me[1]) forward.push(f + " builds on ch" + m[1]);
  }
  ok(!missing.length, "every referenced repo file exists (or is marked planned)", missing.slice(0, 4).join(", "));
  ok(!broken.length, "every link between pages resolves", broken.slice(0, 4).join(", "));
  ok(!forward.length, "'Builds on' only points to earlier chapters", forward.join(", "));
  // There is no LaTeX renderer: `d_{head}` or `\sqrt{x}` outside <pre> shows up literally on the page.
  const latex = [];
  for (const f of [...names].sort()) {
    const body = fs.readFileSync(path.join(COURSE, f), "utf8").replace(/<pre[\s\S]*?<\/pre>/g, "");
    for (const m of body.matchAll(/\\(?:sqrt|times|text|frac|cdot)\b|[A-Za-z]+_\{[^}]*\}/g)) latex.push(f + ": " + m[0]);
  }
  ok(!latex.length, "no raw LaTeX in page text (there is no renderer)", latex.slice(0, 4).join(", "));
  const planned = [...names].filter(f => /class="planned"/.test(fs.readFileSync(path.join(COURSE, f), "utf8")));
  console.log(`  note  ${planned.length} chapters still have a planned (unwritten) companion notebook`);
}

console.log(`\n${fail === 0 ? "ALL CHECKS PASSED" : fail + " CHECK(S) FAILED"}`);
process.exit(fail ? 1 : 0);
})();
