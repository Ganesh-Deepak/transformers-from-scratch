/* List every tensor shape axes.js annotates at runtime, per chapter.

     cd tests && node audit_annotations.js ch05-attention.html ...

   axes.js wraps shape expressions inside <pre> code blocks after the page's
   syntax highlighter runs. Its whitelist was first audited on Ch 6 only, where
   a naive version had dressed `def forward(self, x)` up as a tensor shape.
   Run this whenever axes.js is loaded on a new chapter and read the list for
   anything that is not really a shape. */
const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs");
const path = require("path");
const COURSE = path.join(__dirname, "..", "course");

(async () => {
  const files = process.argv.slice(2);
  for (const f of files) {
    const html = fs.readFileSync(path.join(COURSE, f), "utf8");
    const vc = new VirtualConsole(); const errs = [];
    vc.on("jsdomError", e => errs.push(e.message.split("\n")[0]));
    const dom = new JSDOM(html, {
      runScripts: "dangerously", resources: "usable", pretendToBeVisual: true, virtualConsole: vc,
      url: "file:///" + path.join(COURSE, f).split(path.sep).join("/"),
      beforeParse(w) {
        w.matchMedia = q => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {},
                               addListener() {}, removeListener() {} });
      },
    });
    await new Promise(r => { dom.window.addEventListener("load", () => setTimeout(r, 300)); setTimeout(r, 6000); });
    const D = dom.window.document;
    const inCode = [...D.querySelectorAll("pre .shp")].map(s => s.textContent);
    const tally = {};
    inCode.forEach(t => { tally[t] = (tally[t] || 0) + 1; });
    console.log(`\n${f}: ${inCode.length} code-block shapes, ` +
      `${D.querySelectorAll("[data-viz] [data-axis]").length} widget axis marks, script errors: ${errs.length}`);
    Object.entries(tally).sort((a, b) => b[1] - a[1]).forEach(([t, n]) => console.log(`   ${String(n).padStart(2)}x  ${t}`));
    errs.slice(0, 3).forEach(e => console.log("   ERR", e));
  }
})();
