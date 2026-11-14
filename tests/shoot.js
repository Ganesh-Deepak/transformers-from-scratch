/* shoot.js — screenshot every page at every viewport in both themes, and
   AUDIT the layout so problems are found by a check, not by eye.

   Drives real headless Chrome over the DevTools protocol (Node's built-in
   WebSocket; no npm deps). For each page x viewport x theme it:
     - emulates the viewport, colour scheme and sidebar state,
     - runs layout audits in the page (see AUDIT below),
     - collects console errors and uncaught exceptions,
     - saves an above-the-fold JPEG.
   With --full it also renders whole pages at reduced scale and lays each one
   out as a contact sheet (the long page cut into side-by-side columns), so a
   27-chapter course can be looked at in ~30 images instead of ~800.

   Usage:
     node shoot.js [--out DIR] [--full] [--only ch06,index] [--vp desktop,phone]
   Exit code 1 if any audit finds a problem.                                 */

"use strict";
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
const OUT = path.resolve(opt("--out", path.join(os.tmpdir(), "tfs-shots")));
const FULL = args.includes("--full");
const ONLY = opt("--only", "");
const VPONLY = opt("--vp", "");
const EVAL = opt("--eval", "");          // debugging: print this expression per load

const CHROME = [
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find(p => fs.existsSync(p));
if (!CHROME) { console.error("Chrome not found"); process.exit(2); }

const COURSE = path.join(__dirname, "..", "course");
let PAGES = fs.readdirSync(COURSE).filter(f => /^(index|ch\d\d-.*)\.html$/.test(f)).sort();
if (ONLY) PAGES = PAGES.filter(p => ONLY.split(",").some(o => p.startsWith(o)));

/* side: what the reader chose for the sidebar (only matters >= 82rem) */
let VIEWPORTS = [
  { name: "phone",      w: 390,  h: 844,  mobile: true },
  { name: "tablet",     w: 820,  h: 1180, mobile: true },
  { name: "laptop",     w: 1280, h: 800 },
  { name: "desktop",    w: 1440, h: 900,  side: "open" },
  { name: "wide",       w: 1920, h: 1080, side: "open" },
  { name: "wide-closed",w: 1920, h: 1080, side: "collapsed" },
  { name: "ultra",      w: 2560, h: 1440, side: "collapsed" },
];
if (VPONLY) VIEWPORTS = VIEWPORTS.filter(v => VPONLY.split(",").includes(v.name));
const THEMES = ["light", "dark"];

/* ------------------------------------------------------------ the audit */
/* Runs in the page. Every check reports the OUTERMOST offender only, with a
   short selector, so one broken widget is one line, not two hundred. */
const AUDIT = String.raw`(() => {
  const issues = [];
  const add = (kind, el, detail) => issues.push({ kind, at: el ? sel(el) : "", detail });
  function sel(el) {
    const parts = [];
    for (let e = el; e && e !== document.body && parts.length < 4; e = e.parentElement) {
      let s = e.tagName.toLowerCase();
      if (e.id) { s += "#" + e.id; parts.unshift(s); break; }
      const c = [...e.classList].filter(x => !/^(is-|ax-)/.test(x)).slice(0, 2);
      if (c.length) s += "." + c.join(".");
      parts.unshift(s);
    }
    return parts.join(" > ");
  }
  const scrolls = el => { const o = getComputedStyle(el).overflowX; return o === "auto" || o === "scroll"; };
  const insideScroller = (el, stop) => { for (let a = el.parentElement; a && a !== stop; a = a.parentElement) if (scrolls(a)) return true; return false; };
  const visible = el => { const r = el.getBoundingClientRect(); if (!r.width || !r.height) return false;
    const cs = getComputedStyle(el); return cs.visibility !== "hidden" && cs.display !== "none"; };

  const de = document.documentElement, vw = de.clientWidth;
  if (de.scrollWidth > vw + 1) add("page-scrolls-sideways", null, (de.scrollWidth - vw) + "px");

  const main = document.querySelector("main");
  const facts = {};
  if (main) {
    const mr = main.getBoundingClientRect();
    facts.mainW = Math.round(mr.width);
    facts.ext = Math.round(parseFloat(getComputedStyle(main).paddingLeft));

    /* 1. content outside the sheet (main's box) — not inside a scroller */
    const out = [];
    for (const el of main.querySelectorAll("*")) {
      if (!visible(el)) continue;
      const r = el.getBoundingClientRect();
      if ((r.right > mr.right + 2 || r.left < mr.left - 2) && !insideScroller(el, main)) {
        if (out.some(o => o.contains(el))) continue;
        if (el.closest("svg") && el.tagName.toLowerCase() !== "svg") continue;   // handled by svg check
        out.push(el);
        add("outside-sheet", el, "L" + Math.round(r.left - mr.left) + " R" + Math.round(r.right - mr.right));
      }
    }

    /* 2. prose balance inside the sheet: free space left vs right of the
          text column. Reported as a fact; flagged when lopsided and the
          sheet is wide enough to centre without losing the margin notes. */
    const p = [...main.querySelectorAll(":scope > p")].find(visible);
    if (p) {
      const r = p.getBoundingClientRect();
      facts.left = Math.round(r.left - mr.left);
      facts.right = Math.round(mr.right - r.right);
    }
    if (p && Math.abs(facts.left - facts.right) > 4) add("prose-off-centre", p, "L" + facts.left + " R" + facts.right);
    const h1 = main.querySelector("h1");
    if (h1 && p) {
      const a = h1.getBoundingClientRect(), b = p.getBoundingClientRect();
      if (Math.abs(a.left - b.left) > 2) add("heading-not-aligned-with-text", h1, Math.round(a.left - b.left) + "px");
    }

    /* 3. margin notes colliding with each other or with wide content */
    const notes = [...main.querySelectorAll(":scope > .mn")].filter(n => visible(n) && getComputedStyle(n).position === "absolute");
    const blocks = [...main.children].filter(c => !c.classList.contains("mn") && visible(c));
    for (let i = 0; i < notes.length; i++) {
      const a = notes[i].getBoundingClientRect();
      for (let j = i + 1; j < notes.length; j++) {
        const b = notes[j].getBoundingClientRect();
        if (a.bottom > b.top + 1 && b.bottom > a.top + 1) add("margin-notes-overlap", notes[j], Math.round(a.bottom - b.top) + "px");
      }
      for (const blk of blocks) {
        const b = blk.getBoundingClientRect();
        if (b.right > a.left + 1 && a.right > b.left + 1 && a.bottom > b.top + 1 && b.bottom > a.top + 1) {
          add("margin-note-covers-content", notes[i], "over " + sel(blk)); break;
        }
      }
    }
    if (notes.length) {
      const last = notes[notes.length - 1].getBoundingClientRect();
      if (last.bottom > mr.bottom + 40) add("margin-note-past-sheet", notes[notes.length - 1], Math.round(last.bottom - mr.bottom) + "px");
    }
  }

  /* 4. clipped text: a non-scrolling box whose content is wider than it */
  for (const el of document.querySelectorAll("body *")) {
    const cs = getComputedStyle(el);
    if (!/(hidden|clip)/.test(cs.overflowX) || cs.textOverflow === "ellipsis") continue;
    if (!visible(el) || el.closest(".sidebar")) continue;
    if (el.scrollWidth > el.clientWidth + 2 && el.textContent.trim()) add("text-clipped", el, (el.scrollWidth - el.clientWidth) + "px");
  }

  /* 5. svg labels drawn outside their svg (an svg clips its overflow) */
  for (const svg of document.querySelectorAll("main svg")) {
    if (!visible(svg)) continue;
    const s = svg.getBoundingClientRect();
    for (const t of svg.querySelectorAll("text")) {
      const r = t.getBoundingClientRect();
      if (!r.width) continue;
      if (r.left < s.left - 1 || r.right > s.right + 1 || r.top < s.top - 1 || r.bottom > s.bottom + 1) {
        add("svg-label-clipped", t, JSON.stringify(t.textContent.slice(0, 24))); break;
      }
    }
  }

  /* 6. type too small to read (computed px, text-bearing elements only) */
  const tiny = new Map();
  for (const el of document.querySelectorAll("body *")) {
    if (el.closest("svg")) continue;
    if (el.closest("sub, sup")) continue;   // indices sit at ~75% by design
    const own = [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    if (!own || !visible(el)) continue;
    const fs = parseFloat(getComputedStyle(el).fontSize);
    if (fs < 11) { const k = sel(el).split(" > ").slice(-2).join(" > ") + " @" + fs.toFixed(1); tiny.set(k, (tiny.get(k) || 0) + 1); }
  }
  for (const [k, n] of tiny) add("tiny-text", null, k + " x" + n);

  /* 7. touch targets on touch viewports */
  if (matchMedia("(pointer: coarse)").matches || innerWidth < 700) {
    const small = new Map();
    for (const el of document.querySelectorAll("button, select, input, .topbar a, .bite-footer a")) {
      if (!visible(el) || el.closest("svg")) continue;
      // exempt: the bit explorer has to fit 32 bit cells across a phone; the
      // number field next to it is the full-size way to set the same value
      if (el.classList.contains("vf-bit") || el.type === "checkbox" || el.type === "radio") continue;
      const r = el.getBoundingClientRect();
      if (el.type === "range") continue;     // the thumb is the target, not the 3px track
      if (r.height < 30 || r.width < 30) { const k = sel(el).split(" > ").slice(-1)[0] + " " + Math.round(r.width) + "x" + Math.round(r.height); small.set(k, (small.get(k) || 0) + 1); }
    }
    for (const [k, n] of small) add("small-touch-target", null, k + " x" + n);
  }

  /* 8. fonts that failed or never loaded */
  for (const f of document.fonts) if (f.status === "error") add("font-failed", null, f.family);

  return { issues, facts, h: de.scrollHeight };
})()`;

/* ---------------------------------------------------------------- CDP */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function launch() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tfs-cdp-"));
  const proc = spawn(CHROME, ["--headless=new", "--disable-gpu", "--hide-scrollbars",
    "--remote-debugging-port=0", "--user-data-dir=" + dir, "--no-first-run",
    "--allow-file-access-from-files", "about:blank"], { stdio: "ignore" });
  const portFile = path.join(dir, "DevToolsActivePort");
  for (let i = 0; i < 100 && !fs.existsSync(portFile); i++) await sleep(100);
  const [port, wsPath] = fs.readFileSync(portFile, "utf8").trim().split("\n");
  const ws = new WebSocket(`ws://127.0.0.1:${port}${wsPath}`);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map(); const listeners = [];
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(m.error.message)) : res(m.result); }
    else listeners.forEach(l => l(m));
  };
  const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
    const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params, sessionId }));
  });
  return { proc, dir, send, on: f => listeners.push(f), close: () => { try { ws.close(); } catch (e) {} proc.kill(); } };
}

async function newPage(b) {
  const { targetId } = await b.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await b.send("Target.attachToTarget", { targetId, flatten: true });
  const s = (m, p) => b.send(m, p, sessionId);
  await s("Page.enable"); await s("Runtime.enable"); await s("Log.enable");
  const errors = [];
  b.on(m => {
    if (m.sessionId !== sessionId) return;
    if (m.method === "Runtime.exceptionThrown") errors.push("exception: " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).split("\n")[0]);
    if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") errors.push("console.error: " + m.params.args.map(a => a.value ?? a.description).join(" ").slice(0, 160));
    if (m.method === "Log.entryAdded" && m.params.entry.level === "error") errors.push("log: " + m.params.entry.text.slice(0, 160));
  });
  let loaded = null;
  b.on(m => { if (m.sessionId === sessionId && m.method === "Page.loadEventFired" && loaded) { loaded(); loaded = null; } });
  return {
    s, errors,
    async go(url) { const p = new Promise(r => (loaded = r)); await s("Page.navigate", { url }); await Promise.race([p, sleep(15000)]); },
    async eval(expr) { const r = await s("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); return r.result.value; },
    async shot(file, extra = {}) { const r = await s("Page.captureScreenshot", { format: file.endsWith(".png") ? "png" : "jpeg", quality: 72, ...extra }); fs.writeFileSync(file, Buffer.from(r.data, "base64")); },
  };
}

/* ------------------------------------------------------------------ run */
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const b = await launch();
  const pg = await newPage(b);
  const report = [];
  let bad = 0;
  const fileUrl = f => "file:///" + path.join(COURSE, f).replace(/\\/g, "/");

  for (const vp of VIEWPORTS) for (const theme of THEMES) {
    await pg.s("Emulation.setDeviceMetricsOverride", { width: vp.w, height: vp.h, deviceScaleFactor: 1, mobile: !!vp.mobile });
    await pg.s("Emulation.setTouchEmulationEnabled", { enabled: !!vp.mobile });
    await pg.s("Emulation.setEmulatedMedia", { features: [
      { name: "prefers-color-scheme", value: theme }, { name: "prefers-reduced-motion", value: "reduce" }] });
    for (const page of PAGES) {
      // fresh state per load: no stored theme (system decides), chosen sidebar state
      await pg.go("about:blank");
      const seed = await pg.s("Page.addScriptToEvaluateOnNewDocument", { source:
        `try{localStorage.clear();${vp.side ? `localStorage.setItem("tfs-side","${vp.side}");` : ""}}catch(e){}` });
      pg.errors.length = 0;
      await pg.go(fileUrl(page));
      await pg.eval("document.fonts.ready.then(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))))");
      await sleep(150);
      await pg.s("Page.removeScriptToEvaluateOnNewDocument", { identifier: seed.identifier });
      if (EVAL) console.log(page, vp.name, theme, JSON.stringify(await pg.eval(EVAL)));
      const res = await pg.eval(AUDIT);
      const tag = `${page.replace(".html", "")}__${vp.name}__${theme}`;
      await pg.shot(path.join(OUT, tag + ".jpg"));
      const issues = res.issues.concat(pg.errors.map(e => ({ kind: "js-error", at: "", detail: e })));
      report.push({ page, vp: vp.name, theme, facts: res.facts, issues });
      if (issues.length) bad++;
      const f = res.facts;
      console.log(`${issues.length ? "FAIL" : "ok  "}  ${tag.padEnd(46)} main ${f.mainW ?? "-"} ext ${f.ext ?? "-"} text L${f.left ?? "-"} R${f.right ?? "-"}`);
      for (const i of issues) console.log(`        ${i.kind.padEnd(28)} ${i.at}  ${i.detail}`);

      if (FULL && theme === "light" || FULL && vp.name === "wide-closed") {
        // whole page at reduced scale, then cut into columns on one sheet
        const scale = vp.w > 1500 ? 0.22 : vp.w > 1000 ? 0.3 : 0.5;
        const H = Math.min(res.h, 60000);
        const png = path.join(OUT, tag + "__full.png");
        await pg.shot(png, { format: "png", captureBeyondViewport: true, clip: { x: 0, y: 0, width: vp.w, height: H, scale } });
        const colW = Math.round(vp.w * scale);
        const colH = Math.max(Math.round(vp.h * scale), Math.ceil(H * scale / (vp.w < 700 ? 8 : 5)));   // ~5 columns, phones 8
        const cols = Math.ceil(H * scale / colH);
        const html = `<body style="margin:0;background:#333;display:flex;gap:8px;padding:8px;width:max-content">` +
          Array.from({ length: cols }, (_, i) => `<div style="width:${colW}px;height:${colH}px;background:url('file:///${png.replace(/\\/g, "/")}') 0 -${i * colH}px no-repeat;outline:1px solid #000"></div>`).join("") + `</body>`;
        const sheetHtml = path.join(OUT, tag + "__sheet.html");
        fs.writeFileSync(sheetHtml, html);
        const sp = await newPage(b);
        const sw = cols * (colW + 8) + 8, sh = colH + 16;
        await sp.s("Emulation.setDeviceMetricsOverride", { width: sw, height: sh, deviceScaleFactor: 1, mobile: false });
        await sp.go("file:///" + sheetHtml.replace(/\\/g, "/"));
        await sleep(200);
        await sp.shot(path.join(OUT, tag + "__sheet.jpg"));
        await b.send("Target.closeTarget", { targetId: (await sp.s("Target.getTargetInfo")).targetInfo.targetId });
        fs.unlinkSync(sheetHtml);
      }
    }
  }
  fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 1));
  const kinds = {};
  for (const r of report) for (const i of r.issues) kinds[i.kind] = (kinds[i.kind] || 0) + 1;
  console.log(`\n${report.length} loads, ${bad} with problems. Screenshots: ${OUT}`);
  for (const [k, n] of Object.entries(kinds).sort((a, b) => b[1] - a[1])) console.log(`  ${String(n).padStart(4)}  ${k}`);
  b.close();
  process.exit(bad ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
