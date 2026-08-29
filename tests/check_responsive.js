/* Responsive regression test — real layout, real widths.

     cd tests && node check_responsive.js

   jsdom has no layout engine, so this drives headless Chrome instead and asks
   the page directly whether anything overflows the viewport.

   The page is measured inside a fixed-width IFRAME rather than by sizing the
   browser window. --window-size is clamped by the OS to roughly 500px, so the
   previous approach silently rendered 320, 390 and 430 all at the same ~504px
   and reported three passes for widths it never drew. That gap hid four real
   bugs, each of which only overflows below ~500px:
     - the masthead (.home does not shrink, four controls beside it)
     - .shp { white-space: nowrap } on a shape badge near the right edge
     - tables not wrapped in .tw
     - prose inside .opt, a display:flex container, where every text run and
       inline span is blockified into a non-wrapping flex item
   An iframe gets a genuine narrow viewport, so these rows now mean what they
   say. */
const { execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME_CANDIDATES = [
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
];
const CHROME = CHROME_CANDIDATES.find(p => fs.existsSync(p));
if (!CHROME) {
  console.error("\n  Chrome not found — this test drives headless Chrome.\n");
  process.exit(2);
}

const COURSE = path.join(__dirname, "..", "course");
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "tfs-resp-"));

const WIDTHS = [320, 375, 390, 430, 768, 1024, 1440, 1920];
const PAGES = ["index.html", "ch01-tensors.html", "ch05-attention.html", "ch06-mha.html",
               "ch11-training.html", "ch16-gqa-mla.html", "ch17-flashattention.html",
               "ch19-distributed.html", "ch23-sft-rlhf.html"];

/* Runs INSIDE the harness page, against the iframe's window/document. */
const PROBE = `
function probe(W, D) {
  var d = D.documentElement, b = D.body;
  var vw = W.innerWidth;
  var wide = [];
  var all = D.querySelectorAll("body *");
  for (var i = 0; i < all.length; i++) {
    var el = all[i];
    var r = el.getBoundingClientRect();
    if (r.width > 0 && r.right > vw + 1) {
      // An ancestor that scrolls or clips owns its overflow, and that is fine.
      // Walk real computed styles: the old check matched tag names and inline
      // style attributes, which missed .tw wrappers and any new container.
      var a = el.parentElement, contained = false;
      while (a && a !== D.body) {
        var ox = W.getComputedStyle(a).overflowX;
        if (ox === "auto" || ox === "scroll" || ox === "hidden" || ox === "clip") { contained = true; break; }
        a = a.parentElement;
      }
      if (!contained) {
        wide.push(el.tagName.toLowerCase() + "." + (el.className || "").toString().split(" ")[0]
                  + " right=" + Math.round(r.right));
      }
    }
  }
  return JSON.stringify({
    vw: vw,
    docScrollW: d.scrollWidth,
    clientW: d.clientWidth,
    bodyScrollW: b.scrollWidth,
    overflows: d.scrollWidth > d.clientWidth + 1,
    offenders: wide.slice(0, 4)
  });
}
`;

function measure(page, width) {
  const pageUrl = "file:///" + path.join(COURSE, page).split(path.sep).join("/");
  const wrapper = [
    '<!DOCTYPE html><html><head><meta charset="utf-8"><style>',
    'html,body{margin:0;padding:0;overflow:hidden}',
    'iframe{width:' + width + 'px;height:900px;border:0;display:block}',
    '</style></head><body>',
    '<iframe id="f" src="' + pageUrl + '"></iframe><pre id="__probe"></pre>',
    '<script>', PROBE,
    'document.getElementById("f").addEventListener("load", function(){',
    '  var f = document.getElementById("f");',
    '  setTimeout(function(){',
    '    var out;',
    '    try { out = probe(f.contentWindow, f.contentDocument); }',
    '    catch (e) { out = JSON.stringify({ error: String(e) }); }',
    '    document.getElementById("__probe").textContent = out;',
    '  }, 500);',
    '});',
    '<\/script></body></html>',
  ].join("\n");

  const f = path.join(TMP, width + "-" + page);
  fs.writeFileSync(f, wrapper);
  const out = execFileSync(CHROME, [
    "--headless", "--disable-gpu", "--no-sandbox",
    "--window-size=1400,1000",
    "--allow-file-access-from-files",
    "--user-data-dir=" + path.join(TMP, "prof" + width),
    "--virtual-time-budget=6000", "--dump-dom",
    "file:///" + f.split(path.sep).join("/"),
  ], { encoding: "utf8", maxBuffer: 40 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"] });
  const m = out.match(/<pre id="__probe">([\s\S]*?)<\/pre>/);
  if (!m || !m[1].trim()) return null;
  return JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&")
                        .replace(/&lt;/g, "<").replace(/&gt;/g, ">"));
}

let fail = 0;
console.log("Horizontal overflow check — document must never exceed the viewport");
console.log("(measured in a fixed-width iframe, so narrow widths are real)\n");
for (const page of PAGES) {
  console.log(` ${page}`);
  for (const w of WIDTHS) {
    const r = measure(page, w);
    if (!r || r.error) { console.log(`   ${String(w).padStart(4)}px   probe failed ${r ? r.error : ""}`); fail++; continue; }
    const bad = r.overflows;
    if (bad) fail++;
    console.log(`   ${String(w).padStart(4)}px   ${bad ? "FAIL" : "ok  "}  ` +
      `scrollW ${String(r.docScrollW).padStart(5)} vs client ${String(r.clientW).padStart(5)}` +
      (bad && r.offenders.length ? `   ${r.offenders.join(" | ")}` : ""));
  }
}
fs.rmSync(TMP, { recursive: true, force: true });
console.log(`\n${fail === 0 ? "NO HORIZONTAL OVERFLOW AT ANY WIDTH (320px up)" : fail + " WIDTH(S) OVERFLOW"}`);
process.exit(fail ? 1 : 0);
