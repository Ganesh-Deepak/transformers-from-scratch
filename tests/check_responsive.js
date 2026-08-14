/* Responsive regression test — real layout, real widths.

     cd tests && node check_responsive.js

   jsdom has no layout engine, so this drives headless Chrome instead and asks
   the page directly whether anything overflows the viewport. Catches the class
   of bug where a fixed `37rem` measure silently exceeds a 390px phone.
*/
const { execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME_CANDIDATES = [
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
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

// Widths that matter: phone, large phone, tablet, laptop, desktop.
// CAVEAT: headless Chrome clamps --window-size to a minimum around 500px, so
// the three narrowest entries all actually render at ~504px. They still catch
// the bug class we care about (content wider than its viewport); they do NOT
// prove true 320px behaviour. Verify a real 320px device separately.
const WIDTHS = [320, 390, 430, 768, 1024, 1440, 1920];
const PAGES = ["index.html", "ch01-tensors.html", "ch05-attention.html", "ch16-gqa-mla.html"];

const PROBE = `
(function(){
  var d = document.documentElement, b = document.body;
  var vw = window.innerWidth;
  var wide = [];
  var all = document.querySelectorAll("main *");
  for (var i = 0; i < all.length; i++) {
    var r = all[i].getBoundingClientRect();
    // an element is a problem only if it extends PAST the viewport edge
    if (r.width > 0 && r.right > vw + 1) {
      var el = all[i];
      // a scroll container handles its own overflow — that is fine
      var oc = getComputedStyle(el).overflowX;
      var parentScrolls = el.closest("[style*='overflow'], pre, .tw, figure") !== null;
      if (oc !== "auto" && oc !== "scroll" && !parentScrolls) {
        wide.push(el.tagName.toLowerCase() + "." + (el.className || "").toString().split(" ")[0]
                  + " right=" + Math.round(r.right));
      }
    }
  }
  return JSON.stringify({
    vw: vw,
    docScrollW: d.scrollWidth,
    bodyScrollW: b.scrollWidth,
    overflows: d.scrollWidth > vw + 1,
    offenders: wide.slice(0, 4)
  });
})()
`;

function measure(page, width) {
  // pin the theme and stub storage so nothing depends on prior state
  let src = fs.readFileSync(path.join(COURSE, page), "utf8");
  src = src.replace(/href="assets\//g, `href="file:///${COURSE.replace(/\\/g, "/")}/assets/`)
           .replace(/src="assets\//g, `src="file:///${COURSE.replace(/\\/g, "/")}/assets/`);
  const probeHtml = src.replace("</body>",
    `<script>window.addEventListener("load",function(){
       setTimeout(function(){
         document.title = "RESULT:" + ${JSON.stringify(PROBE)}.length; // keep simple
         var el = document.createElement("pre"); el.id="__probe";
         el.textContent = (function(){ ${PROBE.trim().replace(/^\(function\(\)\{/, "").replace(/\}\)\(\)$/, "")} })();
         document.body.appendChild(el);
       }, 250);
     });</script></body>`);
  const f = path.join(TMP, `${width}-${page}`);
  fs.writeFileSync(f, probeHtml);
  const out = execFileSync(CHROME, [
    "--headless", "--disable-gpu", "--no-sandbox",
    `--window-size=${width},900`,
    `--user-data-dir=${path.join(TMP, "prof" + width)}`,
    "--virtual-time-budget=4000", "--dump-dom",
    "file:///" + f.replace(/\\/g, "/"),
  ], { encoding: "utf8", maxBuffer: 40 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"] });
  const m = out.match(/<pre id="__probe">([\s\S]*?)<\/pre>/);
  if (!m) return null;
  return JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&")
                        .replace(/&lt;/g, "<").replace(/&gt;/g, ">"));
}

let fail = 0;
console.log("Horizontal overflow check — document must never exceed the viewport\n");
for (const page of PAGES) {
  console.log(` ${page}`);
  for (const w of WIDTHS) {
    const r = measure(page, w);
    if (!r) { console.log(`   ${String(w).padStart(4)}px   probe failed`); fail++; continue; }
    const bad = r.docScrollW > r.vw + 1;
    if (bad) fail++;
    console.log(`   ${String(w).padStart(4)}px   ${bad ? "FAIL" : "ok  "}  ` +
      `scrollW ${String(r.docScrollW).padStart(5)} vs vw ${String(r.vw).padStart(5)}` +
      (bad && r.offenders.length ? `   ${r.offenders.join(" | ")}` : ""));
  }
}
fs.rmSync(TMP, { recursive: true, force: true });
console.log(`\n${fail === 0 ? "NO HORIZONTAL OVERFLOW AT ANY WIDTH" : fail + " WIDTH(S) OVERFLOW"}`);
process.exit(fail ? 1 : 0);
