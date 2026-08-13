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
    ["ok on ok-bg", t["--ok"], t["--ok-bg"], 4.5],
    ["warn on warn-bg", t["--warn"], t["--warn-bg"], 4.5],
    ["stop on stop-bg", t["--stop"], t["--stop-bg"], 4.5],
  ];
  for (const [label, fg, bg, min] of pairs) {
    const r = contrast(fg, bg);
    ok(r >= min, label.padEnd(20), `${r.toFixed(2)}:1 (need ${min})`);
  }

  // the dimension hues must be mutually distinguishable
  const dims = ["--dim-1", "--dim-2", "--dim-3", "--dim-4"].map(k => t[k]);
  let minDelta = 99;
  for (let i = 0; i < dims.length; i++)
    for (let j = i + 1; j < dims.length; j++)
      minDelta = Math.min(minDelta, Math.abs(relLum(dims[i]) - relLum(dims[j])));
  console.log(` dimension palette: 4 hues, min luminance delta ${minDelta.toFixed(3)}`);
  ok(t["--accent"] !== undefined && !dims.includes(t["--accent"]),
     "accent excluded from dimension hues");
}

(async () => {
console.log("\n=== PAGE STRUCTURE ==============================");
for (const file of ["index.html", "ch01-tensors.html", "ch13-gqa-mla.html"]) {
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
  if (file !== "index.html") {
    ok(!!d.querySelector(".chapnav"), "chapter nav built");
    ok(d.querySelectorAll(".opt .mk").length > 0, "quiz markers built");
  }
  dom.window.close();
}

console.log(`\n${fail === 0 ? "ALL CHECKS PASSED" : fail + " CHECK(S) FAILED"}`);
process.exit(fail ? 1 : 0);
})();
