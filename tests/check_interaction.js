/* Interaction regression test — real Chrome, real layout, deterministic clock.

     cd tests && node check_interaction.js [chapter.html]

   WHY THIS EXISTS
   check_ui.js proves the controls are PRESENT (jsdom: no layout, no timers).
   check_responsive.js proves nothing OVERFLOWS. Neither one ever presses a
   button, so a whole class of bug shipped unseen — a duration select that
   saved its value but never restarted the running sprint, a sprint that
   expired and left its "finished" state stuck on the console afterwards.

   Those need a real browser. They also need TIME, and waiting fifteen real
   minutes for a countdown is not a test. So a fake clock is injected BEFORE
   course.js loads, replacing setInterval/setTimeout with a manually advanced
   queue. __advance(60000) moves a sprint one minute in microseconds, exactly
   and repeatably.

   Each scenario runs in its own page load, drives real clicks, and returns
   observable DOM state as JSON, which is asserted here. */
const { execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME = [
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find(p => fs.existsSync(p));
if (!CHROME) { console.error("\n  Chrome not found — this test drives headless Chrome.\n"); process.exit(2); }

const COURSE = path.join(__dirname, "..", "course");
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "tfs-int-"));
const ALL = process.argv.includes("--all");
const PAGE = process.argv.filter(a => a.endsWith(".html"))[0] || "ch06-mha.html";

/* Every chapter, read from the manifest, so a new chapter is covered
   automatically instead of having to be remembered. */
function allChapters() {
  const src = fs.readFileSync(path.join(COURSE, "assets", "chapters.js"), "utf8");
  return [...src.matchAll(/id:\s*"\d+",\s*f:\s*"([^"]+)"(,\s*wip:\s*true)?/g)]
    .filter(m => !m[2]).map(m => m[1]);
}

/* Installed before course.js so the study console's timer is the fake one.
   A real setTimeout is kept aside so the harness itself can still yield. */
const CLOCK_PARTS = [
  "<scr" + "ipt>",
  "(function () {",
  "  var now = 0, seq = 1, tasks = [];",
  "  window.__realTimeout = window.setTimeout.bind(window);",
  "  window.setInterval = function (fn, ms) {",
  "    var t = { id: seq++, fn: fn, ms: Math.max(1, ms || 1), repeat: true };",
  "    t.next = now + t.ms; tasks.push(t); return t.id;",
  "  };",
  "  window.setTimeout = function (fn, ms) {",
  "    var t = { id: seq++, fn: fn, ms: Math.max(0, ms || 0), repeat: false };",
  "    t.next = now + t.ms; tasks.push(t); return t.id;",
  "  };",
  "  window.clearInterval = window.clearTimeout = function (id) {",
  "    for (var i = 0; i < tasks.length; i++) if (tasks[i].id === id) { tasks.splice(i, 1); return; }",
  "  };",
  "  window.__advance = function (ms) {",
  "    var end = now + ms, guard = 0;",
  "    while (guard++ < 200000) {",
  "      var due = null;",
  "      for (var i = 0; i < tasks.length; i++)",
  "        if (tasks[i].next <= end && (!due || tasks[i].next < due.next)) due = tasks[i];",
  "      if (!due) break;",
  "      now = due.next;",
  "      if (due.repeat) due.next = now + due.ms;",
  "      else tasks.splice(tasks.indexOf(due), 1);",
  "      try { due.fn(); } catch (e) {}",
  "    }",
  "    now = end;",
  "  };",
  "})();",
  "</scr" + "ipt>",
].join("\n");

/* helpers every scenario can use */
const HELPERS = [
  "var $ = function (s) { return document.querySelector(s); };",
  "var timer = function () { var e = $('.study-timer'); return e ? e.textContent : ''; };",
  "var position = function () { var e = $('.study-position'); return e ? e.textContent : ''; };",
  "var setDuration = function (v) {",
  "  var d = $('.study-duration'); d.value = String(v);",
  "  d.dispatchEvent(new Event('change', { bubbles: true }));",
  "};",
  "var goBite = function (i) { $('.study-step[data-index=\"' + i + '\"]').click(); };",
  "var visibleFooterIndex = function () {",
  "  var f = [].slice.call(document.querySelectorAll('.bite-footer'))",
  "    .filter(function (x) { return getComputedStyle(x).display !== 'none'; });",
  "  return f.length === 1 ? +f[0].dataset.index : f.map(function (x) { return +x.dataset.index; });",
  "};",
  "var visibleHeading = function () {",
  "  var h = [].slice.call(document.querySelector('main').children)",
  "    .filter(function (e) { return e.tagName === 'H2' && !e.classList.contains('focus-hidden'); });",
  "  return h.length === 1 ? h[0].textContent.trim().slice(0, 40) : h.length + ' headings';",
  "};",
  "var finished = function () { return $('.study-console').classList.contains('sprint-finished'); };",
  // There is one .bite-footer per bite and only the active one is shown, so a
  // bare querySelector('.bite-prev') grabs footer 0's button -- which is
  // correctly DISABLED. Drive the visible footer, the way a reader would.
  "var footBtn = function (which) {",
  "  var f = [].slice.call(document.querySelectorAll('.bite-footer'))",
  "    .filter(function (x) { return getComputedStyle(x).display !== 'none'; })[0];",
  "  if (!f) throw new Error('no visible bite footer');",
  "  var b = f.querySelector('.bite-' + which);",
  "  if (b.disabled) throw new Error('bite-' + which + ' is disabled on the visible footer');",
  "  b.click();",
  "};",
].join("\n");

function run(scenarioSrc, page) {
  let src = fs.readFileSync(path.join(COURSE, page || PAGE), "utf8");
  const base = "file:///" + COURSE.split(path.sep).join("/");
  src = src.replace(/href="assets\//g, 'href="' + base + '/assets/')
           .replace(/src="assets\//g, 'src="' + base + '/assets/');
  src = src.replace("</head>", CLOCK_PARTS + "\n</head>");

  const runner = [
    '<pre id="__result"></pre>',
    "<scr" + "ipt>",
    'window.addEventListener("load", function () {',
    "  window.__realTimeout(function () {",
    "    var out;",
    "    try { out = (function () {", HELPERS, scenarioSrc, "})(); }",
    '    catch (e) { out = { error: String((e && e.stack) || e) }; }',
    '    document.getElementById("__result").textContent = JSON.stringify(out);',
    "  }, 150);",
    "});",
    "</scr" + "ipt></body>",
  ].join("\n");
  src = src.replace("</body>", runner);

  const f = path.join(TMP, "run-" + (page || PAGE));
  fs.writeFileSync(f, src);
  const dom = execFileSync(CHROME, [
    "--headless", "--disable-gpu", "--no-sandbox",
    "--window-size=1280,900", "--allow-file-access-from-files",
    "--user-data-dir=" + path.join(TMP, "prof"),
    "--virtual-time-budget=8000", "--dump-dom",
    "file:///" + f.split(path.sep).join("/"),
  ], { encoding: "utf8", maxBuffer: 40 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"] });
  const m = dom.match(/<pre id="__result">([\s\S]*?)<\/pre>/);
  if (!m || !m[1].trim()) return { error: "scenario produced no output" };
  return JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&")
                        .replace(/&lt;/g, "<").replace(/&gt;/g, ">"));
}

let fail = 0;
const ok = (cond, label, extra) => {
  console.log("  " + (cond ? "PASS" : "FAIL") + "  " + label + (extra ? "   " + extra : ""));
  if (!cond) fail++;
};

console.log("Interaction check — " + PAGE + ", real Chrome, deterministic clock\n");

/* ---------------------------------------------------- 1. duration select -- */
console.log(" sprint duration");
const d = run([
  "var r = {};",
  "goBite(0);",
  "r.defaultTimer = timer();",
  "setDuration(15);",
  "r.afterChangeWhileFocused = timer();",
  "window.__advance(60000);",
  "r.afterOneMinute = timer();",
  "setDuration(5);",
  "r.afterSecondChange = timer();",
  "return r;",
].join("\n"));
if (d.error) ok(false, "scenario ran", d.error);
else {
  ok(/^\d+:\d\d$/.test(d.defaultTimer), "timer starts on entering a bite", d.defaultTimer);
  ok(d.afterChangeWhileFocused === "15:00",
     "changing duration mid-sprint resets the clock", "got " + d.afterChangeWhileFocused + ", want 15:00");
  ok(d.afterOneMinute === "14:00",
     "counts down from the NEW duration", "got " + d.afterOneMinute + ", want 14:00");
  ok(d.afterSecondChange === "5:00",
     "changing again resets again", "got " + d.afterSecondChange + ", want 5:00");
}

/* --------------------------------------------------- 2. bite navigation --- */
console.log("\n bite navigation");
const n = run([
  "var r = {};",
  "goBite(0);",
  "r.pos0 = position(); r.foot0 = visibleFooterIndex(); r.head0 = visibleHeading();",
  "footBtn('next');",
  "r.pos1 = position(); r.foot1 = visibleFooterIndex(); r.head1 = visibleHeading();",
  "footBtn('next');",
  "r.pos2 = position(); r.foot2 = visibleFooterIndex(); r.head2 = visibleHeading();",
  "footBtn('prev');",
  "r.posBack = position(); r.footBack = visibleFooterIndex();",
  "footBtn('last');",
  "r.posLast = position(); r.footLast = visibleFooterIndex();",
  "r.total = document.querySelectorAll('.bite-footer').length;",
  "return r;",
].join("\n"));
if (n.error) ok(false, "scenario ran", n.error);
else {
  ok(/\b1 of\b/.test(n.pos0), "counter reads bite 1 on entry", n.pos0);
  ok(/\b2 of\b/.test(n.pos1), "counter advances to 2 on next", n.pos1);
  ok(/\b3 of\b/.test(n.pos2), "counter advances to 3 on next", n.pos2);
  ok(/\b2 of\b/.test(n.posBack), "counter returns to 2 on prev", n.posBack);
  ok(n.posLast.indexOf(n.total + " of " + n.total) !== -1, "counter reads last on last-bite", n.posLast);
  ok(n.foot0 === 0 && n.foot1 === 1 && n.foot2 === 2 && n.footBack === 1 && n.footLast === n.total - 1,
     "visible footer tracks the bite",
     [n.foot0, n.foot1, n.foot2, n.footBack, n.footLast].join(","));
  ok(n.head0 !== n.head1 && n.head1 !== n.head2, "content changes per bite",
     [n.head0, n.head1, n.head2].join(" | "));
}

/* --------------------------------------------------- 3. timer lifecycle --- */
console.log("\n timer lifecycle");
const t = run([
  "var r = {};",
  "setDuration(5);",
  "goBite(0);",
  "r.start = timer();",
  "window.__advance(120000);",
  "r.after2min = timer();",
  "footBtn('next');",
  "r.afterNavigate = timer();",
  "window.__advance(180000);",
  "r.afterExpiry = timer();",
  "r.finishedFlag = finished();",
  "footBtn('next');",
  "r.afterNavPastExpiry = timer();",
  "r.finishedAfterNav = finished();",
  "footBtn('exit');",
  "r.afterExit = timer();",
  "window.__advance(60000);",
  "r.afterExitPlusMinute = timer();",
  "return r;",
].join("\n"));
if (t.error) ok(false, "scenario ran", t.error);
else {
  ok(t.start === "5:00", "sprint starts at the chosen length", t.start);
  ok(t.after2min === "3:00", "counts down", t.after2min);
  ok(t.afterNavigate === "3:00", "sprint continues across bites (by design)", t.afterNavigate);
  ok(t.afterExpiry === "0:00", "reaches zero", t.afterExpiry);
  ok(t.finishedFlag === true, "marks the sprint finished");
  ok(t.finishedAfterNav === false,
     "finished state clears when a new bite starts", "still finished: " + t.finishedAfterNav);
  ok(t.afterExit === "", "timer clears on leaving focus mode", JSON.stringify(t.afterExit));
  ok(t.afterExitPlusMinute === "",
     "timer does not keep running after exit", JSON.stringify(t.afterExitPlusMinute));
}

/* -------------------------------- 4. axis badges inside quiz options ------ */
console.log("\n axis badges inside quiz options");
const a = run([
  "var r = {};",
  "var opt = [].slice.call(document.querySelectorAll('.opt'))",
  "  .filter(function (o) { return o.querySelector('[data-axis]'); })[0];",
  "if (!opt) return { skipped: true };",
  "var item = opt.closest('.qitem');",
  "opt.querySelector('[data-axis]').click();",
  "r.answeredAfterAxisClick = item.dataset.answered || 'no';",
  "r.pinned = document.documentElement.getAttribute('data-ax-pinned') || '';",
  "opt.click();",
  "r.answeredAfterOptionClick = item.dataset.answered || 'no';",
  "return r;",
].join("\n"));
if (a.skipped) console.log("  SKIP  no quiz option on this page contains a shape badge");
else if (a.error) ok(false, "scenario ran", a.error);
else {
  ok(a.answeredAfterAxisClick === "no", "clicking an axis does not answer the question");
  ok(a.pinned !== "", "clicking an axis pins it", a.pinned);
  ok(a.answeredAfterOptionClick === "1", "clicking the option still answers it");
}

/* ------------------------------- 5. every chapter, condensed -------------- */
if (ALL) {
  console.log("\n every chapter: navigation + timer lifecycle");
  const SWEEP = [
    "var r = { steps: [] };",
    "var n = document.querySelectorAll('.bite-footer').length;",
    "r.n = n;",
    "setDuration(5);",
    "goBite(0);",
    "r.timerStart = timer();",
    "for (var i = 0; i < n; i++) {",
    "  if (i) footBtn('next');",
    "  r.steps.push({ i: i, pos: position(), foot: visibleFooterIndex(), head: visibleHeading() });",
    "}",
    "goBite(0);",
    "footBtn('last');",
    "r.lastFoot = visibleFooterIndex();",
    "goBite(0);",
    "r.backToFirst = visibleFooterIndex();",
    "window.__advance(300000);",
    "r.expired = timer(); r.finished = finished();",
    "footBtn('next');",
    "r.afterExpiryNav = timer(); r.finishedAfterNav = finished();",
    "$('.bite-exit').click();",
    "r.afterExit = timer();",
    "r.hiddenAfterExit = document.querySelectorAll('main > .focus-hidden').length;",
    "return r;",
  ].join("\n");
  for (const page of allChapters()) {
    const r = run(SWEEP, page);
    const problems = [];
    if (r.error) problems.push(String(r.error).slice(0, 90));
    else {
      if (r.timerStart !== "5:00") problems.push("timer start " + r.timerStart);
      r.steps.forEach(st => {
        if (st.foot !== st.i) problems.push("b" + st.i + " footer=" + st.foot);
        if (st.pos.indexOf((st.i + 1) + " of " + r.n) === -1) problems.push("b" + st.i + " counter '" + st.pos + "'");
        if (/headings$/.test(st.head)) problems.push("b" + st.i + " " + st.head);
      });
      if (r.lastFoot !== r.n - 1) problems.push("last->" + r.lastFoot);
      if (r.backToFirst !== 0) problems.push("first->" + r.backToFirst);
      if (r.expired !== "0:00") problems.push("expiry " + r.expired);
      if (r.finished !== true) problems.push("not marked finished");
      if (r.finishedAfterNav !== false) problems.push("finished stuck after nav");
      if (r.afterExit !== "") problems.push("timer after exit '" + r.afterExit + "'");
      if (r.hiddenAfterExit !== 0) problems.push(r.hiddenAfterExit + " hidden after exit");
    }
    ok(problems.length === 0,
       page.replace(".html", "").padEnd(22) + (r.n ? r.n + " bites" : ""),
       problems.slice(0, 3).join(" | "));
  }
}

fs.rmSync(TMP, { recursive: true, force: true });
console.log("\n" + (fail === 0 ? "ALL INTERACTION CHECKS PASSED" : fail + " INTERACTION CHECK(S) FAILED"));
process.exit(fail ? 1 : 0);
