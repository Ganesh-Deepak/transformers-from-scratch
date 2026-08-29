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
    ["ink-3 on paper", t["--ink-3"], t["--paper"], 3.0],
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

  /* The dimension hues must be mutually distinguishable. Since colour here
     encodes WHICH TENSOR AXIS, two axes that look alike is the exact failure
     the palette exists to prevent — so separation is checked by HUE ANGLE,
     not luminance. Two colours can share a luminance and still be obviously
     different (blue vs orange); they cannot share a hue and be told apart. */
  const dims = ["--dim-1", "--dim-2", "--dim-3", "--dim-4", "--dim-5"].map(k => t[k]);
  const hueOf = hex => {
    const c = hex.replace("#", "");
    const [r, g, b] = [0, 2, 4].map(i => parseInt(c.substr(i, 2), 16) / 255);
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    if (!d) return 0;
    const h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
    return (Math.round(h * 60) + 360) % 360;
  };
  let minHue = 360, worst = "";
  for (let i = 0; i < dims.length; i++)
    for (let j = i + 1; j < dims.length; j++) {
      let d = Math.abs(hueOf(dims[i]) - hueOf(dims[j]));
      d = Math.min(d, 360 - d);
      if (d < minHue) { minHue = d; worst = `dim-${i + 1}/dim-${j + 1}`; }
    }
  ok(minHue >= 30, "axis hues at least 30deg apart", `closest ${worst} at ${minHue}deg`);
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
console.log("\n=== MANIFEST INTEGRITY ==========================");
{
  const src = fs.readFileSync(path.join(ROOT, "assets", "chapters.js"), "utf8");
  const entries = [...src.matchAll(/id:\s*"(\d+)",\s*f:\s*"([^"]+)"(,\s*wip:\s*true)?/g)]
    .map(m => ({ id: m[1], file: m[2], wip: !!m[3] }));

  ok(entries.length > 0, "manifest parses", `${entries.length} entries`);

  const seq = entries.every((e, i) => e.id === String(i + 1).padStart(2, "0"));
  ok(seq, "ids run 01..NN with no gaps or repeats",
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
    for (const m of html.matchAll(/href="([^"#:]+\.html)(?:#[^"]*)?"/g)) {
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

console.log(`\n${fail === 0 ? "ALL CHECKS PASSED" : fail + " CHECK(S) FAILED"}`);
process.exit(fail ? 1 : 0);
})();
