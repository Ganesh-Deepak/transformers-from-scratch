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
  // Date.now moves with the fake clock, so deadline-based timers are testable
  "  var realNow = Date.now.bind(Date), base = realNow();",
  "  Date.now = function () { return base + now; };",
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
  // a frozen background tab: wall-clock time passes but no interval fires
  "  window.__jump = function (ms) { now += ms; };",
  // every run starts from empty storage, optionally pre-seeded by the scenario
  "  try { localStorage.clear(); if (window.__SEED__) for (var k in window.__SEED__) localStorage.setItem(k, window.__SEED__[k]); } catch (e) {}",
  // record and cancel any page navigation, so 'must not navigate' is observable
  "  window.__navs = [];",
  "  if (window.navigation) window.navigation.addEventListener('navigate', function (e) {",
  "    if (e.hashChange) return;",
  "    window.__navs.push(e.destination.url.split('/').pop());",
  "    if (e.cancelable) e.preventDefault();",
  "  });",
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
  "var key = function (el, k) { el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true })); };",
  "var store = function () { try { return JSON.parse(localStorage.getItem('tfs-progress-v1') || '{}'); } catch (e) { return {}; } };",
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

function run(scenarioSrc, page, seed) {
  let src = fs.readFileSync(path.join(COURSE, page || PAGE), "utf8");
  const base = "file:///" + COURSE.split(path.sep).join("/");
  src = src.replace(/href="assets\//g, 'href="' + base + '/assets/')
           .replace(/src="assets\//g, 'src="' + base + '/assets/');
  const seedJs = "<scr" + "ipt>window.__SEED__ = " + JSON.stringify(seed || {}) + ";</scr" + "ipt>";
  // before the page's own theme snippet, which reads localStorage
  src = src.replace("<meta charset=\"utf-8\">", "<meta charset=\"utf-8\">\n" + seedJs + "\n" + CLOCK_PARTS);

  const runner = [
    '<pre id="__result"></pre>',
    "<scr" + "ipt>",
    'window.addEventListener("load", function () {',
    "  window.__realTimeout(function () {",
    "    var out;",
    '    var done = function (o) { document.getElementById("__result").textContent = JSON.stringify(o); };',
    "    try { out = (function () {", HELPERS, scenarioSrc, "})(); }",
    '    catch (e) { out = { error: String((e && e.stack) || e) }; }',
    // a scenario may return a promise when it must let observers run
    '    if (out && typeof out.then === "function") out.then(done, function (e) { done({ error: String((e && e.stack) || e) }); });',
    '    else done(out);',
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


/* ----------------------------- 6. state machines found in the audit ------ */
console.log("\n state machines");

// 6a. keyboard shortcuts must not hijack keys a component already handled
const kb = run([
  "var r = {};",
  "var opt = $('.qitem .opt');",
  "opt.focus(); key(opt, 'ArrowRight'); key(opt, 'ArrowLeft');",
  "r.navsAfterQuizArrows = window.__navs.slice();",
  "var shp = $('.shp'); shp.focus(); key(shp, 'ArrowRight'); key(shp, 'ArrowLeft');",
  "r.navsAfterBadgeArrows = window.__navs.slice();",
  "key(document.body, 'ArrowRight');",                 // control: on the page itself it SHOULD navigate
  "r.navsAfterBodyArrow = window.__navs.slice();",
  "return r;",
].join("\n"));
if (kb.error) ok(false, "keyboard scenario ran", kb.error);
else {
  ok(kb.navsAfterBodyArrow.length === 1, "control: Right on the page still goes to the next chapter",
     JSON.stringify(kb.navsAfterBodyArrow));
  ok(kb.navsAfterQuizArrows.length === 0, "arrow keys inside a quiz do not change chapter",
     JSON.stringify(kb.navsAfterQuizArrows));
  ok(kb.navsAfterBadgeArrows.length === 0, "arrow keys inside a shape badge do not change chapter",
     JSON.stringify(kb.navsAfterBadgeArrows));
}

// 6b. quiz Retry resets in place: no reload, nothing left answered
const qr = run([
  "window.__alive = 'same page';",
  "var item = $('.qitem'); var quiz = item.closest('.quiz');",
  "item.querySelectorAll('.opt')[0].click();",
  "var r = { answeredBefore: !!item.dataset.answered, storedBefore: Object.keys(store()).filter(function (k) { return k.indexOf('quiz:') === 0; }).length };",
  "quiz.querySelector('.quiz-reset').click();",
  "r.alive = window.__alive; r.navs = window.__navs.length;",
  "r.answeredAfter = !!item.dataset.answered;",
  "r.lockedOpts = item.querySelectorAll('.opt.disabled, .opt.correct, .opt.wrong').length;",
  "r.explShown = !!item.querySelector('.expl.show');",
  "r.storedAfter = Object.keys(store()).filter(function (k) { return k.indexOf('quiz:') === 0; }).length;",
  "item.querySelectorAll('.opt')[1].click();",
  "r.canAnswerAgain = !!item.dataset.answered;",
  "return r;",
].join("\n"));
if (qr.error) ok(false, "quiz retry scenario ran", qr.error);
else {
  ok(qr.answeredBefore && qr.storedBefore === 1, "an answer is recorded");
  ok(qr.alive === "same page" && qr.navs === 0, "Retry does not reload the page");
  ok(!qr.answeredAfter && qr.lockedOpts === 0 && !qr.explShown, "Retry returns the question to unanswered",
     `answered=${qr.answeredAfter} locked=${qr.lockedOpts} expl=${qr.explShown}`);
  ok(qr.storedAfter === 0, "Retry clears the stored answer");
  ok(qr.canAnswerAgain, "the question can be answered again");
}

// 6c. two tabs: one tab's write must not erase another tab's
const st = run([
  "var r = {};",
  "var mark = document.querySelector('.ex-check'); mark.click();",              // tab A writes
  "var other = store(); other['written-by-another-tab'] = true;",               // tab B writes behind its back
  "localStorage.setItem('tfs-progress-v1', JSON.stringify(other));",
  "$('.quiz .opt').click();",                                                   // tab A writes again
  "var now = store();",
  "r.otherSurvived = now['written-by-another-tab'] === true;",
  "r.ownKeys = Object.keys(now).filter(function (k) { return k.indexOf('exercise:') === 0 || k.indexOf('quiz:') === 0; }).length;",
  "return r;",
].join("\n"));
if (st.error) ok(false, "storage scenario ran", st.error);
else {
  ok(st.otherSurvived, "a write from another tab survives this tab's next save");
  ok(st.ownKeys === 2, "and this tab's own writes are kept too", String(st.ownKeys));
}

// 6d. the sprint is a wall-clock deadline: a throttled background tab cannot stretch it
const bg = run([
  "var r = {};",
  "setDuration(10); goBite(0);",
  "r.start = timer();",
  "window.__jump(7 * 60 * 1000);",                    // 7 minutes pass, no interval fires
  "document.dispatchEvent(new Event('visibilitychange'));",
  "r.afterReturn = timer();",
  "window.__jump(4 * 60 * 1000);",
  "window.__advance(1000);",                          // the next real tick
  "r.afterDeadline = timer(); r.finished = finished();",
  "footBtn('exit');",
  "r.finishedAfterExit = finished();",
  "return r;",
].join("\n"));
if (bg.error) ok(false, "background-tab scenario ran", bg.error);
else {
  ok(bg.start === "10:00", "sprint starts at 10:00", bg.start);
  ok(bg.afterReturn === "3:00", "after 7 min in a background tab it reads 3:00, not 10:00", bg.afterReturn);
  ok(bg.afterDeadline === "0:00" && bg.finished, "it expires on the wall clock", bg.afterDeadline);
  ok(bg.finishedAfterExit === false, "leaving focus mode clears the finished state");
}

// 6e. merely opening a chapter must not move the home page's resume pointer
const elsewhere = { chapterId: "12", sectionId: "s-12-3-how-fp16-kills-a-training-run", chapter: "x",
                    section: "x", file: "ch12-precision.html", updated: 1 };
const rs = run("return { resume: store().resume };", "ch05-attention.html",
               { "tfs-progress-v1": JSON.stringify({ resume: elsewhere }) });
if (rs.error) ok(false, "resume scenario ran", rs.error);
else ok(rs.resume && rs.resume.chapterId === "12", "visiting a chapter leaves the resume pointer where you were",
        rs.resume ? "now " + rs.resume.chapterId : "cleared");

// 6f. a theme change rebuilds widgets but keeps what the reader set
const th = run([
  "var r = {};",
  "var w = document.querySelector('[data-viz=\"float\"]');",
  "var sel = w.querySelector('select'); sel.value = 'bf16'; sel.dispatchEvent(new Event('change', { bubbles: true }));",
  "w.querySelector('.vf-bit[data-bit=\"0\"]').click();",
  "r.before = w.querySelector('.vf-value').textContent;",
  "r.fmtBefore = w.querySelector('select').value;",
  "document.querySelector('.theme-toggle').click();",
  "r.after = w.querySelector('.vf-value').textContent;",
  "r.fmtAfter = w.querySelector('select').value;",
  "return r;",
].join("\n"), "ch00-numbers.html");
if (th.error) ok(false, "theme-rebuild scenario ran", th.error);
else ok(th.before === th.after && th.fmtBefore === th.fmtAfter && th.fmtAfter === "bf16",
        "the bit explorer keeps its format and bit pattern across a theme toggle",
        `${th.fmtBefore}:${th.before} -> ${th.fmtAfter}:${th.after}`);

const sm = run([
  "var w = document.querySelector('[data-viz=\"softmax\"]');",
  "var sl = w.querySelector('input[type=range]'); sl.value = '25'; sl.dispatchEvent(new Event('input', { bubbles: true }));",
  "var before = w.querySelector('input[type=range]').value;",
  "document.querySelector('.theme-toggle').click();",
  "return { before: before, after: document.querySelector('[data-viz=\"softmax\"] input[type=range]').value };",
].join("\n"), "ch13-decoding.html");
if (sm.error) ok(false, "slider-rebuild scenario ran", sm.error);
else ok(sm.before === "25" && sm.after === "25", "a widget slider survives a theme toggle", `${sm.before} -> ${sm.after}`);

/* --------------------------------------------- 7. the bit explorer -------- */
console.log("\n the bit explorer (Ch 00)");
const fx = run([
  "var r = {};",
  "var w = document.querySelector('[data-viz=\"float\"]');",
  "var sel = w.querySelector('select'), num = w.querySelector('.vf-num');",
  "var val = function () { return w.querySelector('.vf-value').textContent; };",
  "var setFmt = function (f) { sel.value = f; sel.dispatchEvent(new Event('change', { bubbles: true })); };",
  "var type = function (t) { num.value = t; num.dispatchEvent(new Event('change', { bubbles: true })); };",
  "setFmt('bf16'); type('0.1'); r.bf16_01 = val();",
  "setFmt('fp16'); r.fp16_01 = val();",                 // same number, re-encoded in the new format
  "type('70000'); r.fp16_70000 = val();",
  "setFmt('e4m3'); type('500'); r.e4m3_500 = val();",
  "type('300'); r.e4m3_300 = val();",
  "[].slice.call(w.querySelectorAll('.vf-preset')).filter(function (b) { return b.textContent === 'max'; })[0].click();",
  "r.e4m3_max = val(); r.max_label = w.querySelector('.vf-max').textContent;",
  "setFmt('e2m1'); type('5'); r.fp4_5 = val();",
  "setFmt('fp32'); type('1'); r.bits1 = w.querySelectorAll('.vf-bit').length;",
  "var b = w.querySelector('.vf-bit[data-bit=\"23\"]');",  // lowest exponent bit of 1.0 (01111111) -> 0 gives 0.5
  "b.click(); r.afterFlip = val(); r.pressed = w.querySelector('.vf-bit[data-bit=\"23\"]').getAttribute('aria-pressed');",
  "r.focusKept = document.activeElement && document.activeElement.dataset.bit === '23';",
  "key(document.activeElement, 'ArrowRight'); r.navs = window.__navs.length;",
  "type('not a number'); r.invalid = num.getAttribute('aria-invalid'); r.unchanged = val();",
  "return r;",
].join("\n"), "ch00-numbers.html");
if (fx.error) ok(false, "explorer scenario ran", fx.error);
else {
  ok(fx.bf16_01 === "0.10009766", "bf16 stores 0.1 as 0.10009766", fx.bf16_01);
  ok(fx.fp16_01 === "0.099975586", "switching format re-encodes the same number", fx.fp16_01);
  ok(fx.fp16_70000 === "+inf", "fp16 overflows 70000 to infinity", fx.fp16_70000);
  ok(fx.e4m3_500 === "448" && fx.e4m3_300 === "288", "e4m3 saturates 500 to 448 and rounds 300 to 288",
     `${fx.e4m3_500} ${fx.e4m3_300}`);
  ok(fx.e4m3_max === "448" && fx.max_label === "448", "the max preset and the derived max agree", fx.e4m3_max);
  ok(fx.fp4_5 === "4", "fp4 rounds the tie 5 to 4 (even mantissa)", fx.fp4_5);
  ok(fx.bits1 === 32, "fp32 shows 32 clickable bits", String(fx.bits1));
  ok(fx.afterFlip === "0.5" && fx.pressed === "false", "clicking a bit flips it and re-decodes", fx.afterFlip);
  ok(fx.focusKept, "focus stays on the bit you clicked");
  ok(fx.navs === 0, "arrow keys on a bit do not change chapter");
  ok(fx.invalid === "true" && fx.unchanged === "0.5", "invalid input is flagged and ignored");
}


/* ---------------------------- 8. Phase 2: widgets on the axis bus -------- */
console.log("\n widgets coupled to the axis highlighter");

/* helpers: how lit is an axis inside a widget, and on the page's badges */
const W_HELP = [
  "var visibleBands = function (w) { return [].slice.call(w.querySelectorAll('rect[pointer-events=\"none\"]')).filter(function (r) { return r.getAttribute('opacity') === '1'; }).length; };",
  "var proseBadge = function (a) { return [].slice.call(document.querySelectorAll('main > p b[data-axis=\"' + a + '\"], main li b[data-axis=\"' + a + '\"], .qq b[data-axis=\"' + a + '\"]'))[0] || document.querySelector('pre b[data-axis=\"' + a + '\"]'); };",
  "var op = function (el) { return getComputedStyle(el).opacity; };",
].join("\n");

const hx = run([W_HELP,
  "var r = {};",
  "var w = document.querySelector('[data-viz=\"heads\"]');",
  "r.idleBands = visibleBands(w);",
  // widget -> page: click the widget's own T label
  "var tLabel = [].slice.call(w.querySelectorAll('text[data-axis=\"time\"]')).filter(function (t) { return /T=/.test(t.textContent); })[0];",
  "tLabel.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));",
  "r.pinnedAfterWidgetClick = document.documentElement.getAttribute('data-ax-pinned');",
  "r.bandsWhenTimePinned = visibleBands(w);",
  "var pb = proseBadge('time'); var pbOther = proseBadge('batch');",
  "r.proseTimeOpacity = op(pb); r.proseBatchOpacity = op(pbOther);",
  "r.widgetTLabelDecor = tLabel.getAttribute('text-decoration');",
  "var dmLabel = [].slice.call(w.querySelectorAll('text[data-axis=\"feat\"]'))[0];",
  "r.otherLabelOpacity = dmLabel.getAttribute('opacity');",
  // survives a redraw (n_head change) without losing the pin or leaking listeners
  "var sel = w.querySelector('select'); sel.value = '4'; sel.dispatchEvent(new Event('change', { bubbles: true }));",
  "sel.value = '6'; sel.dispatchEvent(new Event('change', { bubbles: true }));",
  "r.bandsAfterRedraw = visibleBands(w);",
  "r.expectedTimeBands = 4 + 4 + 4 * 6;",       // T rows in step 1, step 2, and per head in step 3 (nh = 6)
  // and a theme toggle
  "document.querySelector('.theme-toggle').click();",
  "w = document.querySelector('[data-viz=\"heads\"]');",
  "r.bandsAfterTheme = visibleBands(w);",
  "r.nheadAfterTheme = w.querySelector('select').value;",
  // page -> widget: clear, then hover a prose badge for head-dim
  "key(document.body, 'Escape');",
  "r.bandsAfterEscape = visibleBands(w);",
  "var hd = proseBadge('headdim'); hd.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));",
  "r.bandsOnProseHover = visibleBands(w);",
  "hd.dispatchEvent(new MouseEvent('mouseout', { bubbles: true }));",
  "r.bandsAfterUnhover = visibleBands(w);",
  // keyboard: tab stop on a widget axis label, Enter pins, arrows do not change chapter
  "var nhLabel = [].slice.call(w.querySelectorAll('text[data-axis=\"head\"][tabindex=\"0\"]'))[0];",
  "nhLabel.focus(); r.focusOk = document.activeElement === nhLabel;",
  "key(nhLabel, 'Enter'); r.pinnedByKeyboard = document.documentElement.getAttribute('data-ax-pinned');",
  "key(nhLabel, 'ArrowRight'); r.navs = window.__navs.length;",
  "return r;",
].join("\n"), "ch06-mha.html");
if (hx.error) ok(false, "heads scenario ran", hx.error);
else {
  ok(hx.idleBands === 0, "heads: no bands lit while idle", String(hx.idleBands));
  ok(hx.pinnedAfterWidgetClick === "time", "heads: clicking the widget's T pins time on the page bus", hx.pinnedAfterWidgetClick);
  ok(hx.bandsWhenTimePinned === 4 + 4 + 4 * 3, "heads: every time row lights, in all three steps",
     `${hx.bandsWhenTimePinned} bands (want ${4 + 4 + 4 * 3})`);
  ok(hx.proseTimeOpacity === "1" && +hx.proseBatchOpacity < 0.5, "heads: the prose badges respond to the widget",
     `T ${hx.proseTimeOpacity}, B ${hx.proseBatchOpacity}`);
  ok(hx.widgetTLabelDecor === "underline" && +hx.otherLabelOpacity < 0.5, "heads: active label underlines, others recede (not colour-only)",
     `${hx.widgetTLabelDecor} / ${hx.otherLabelOpacity}`);
  ok(hx.bandsAfterRedraw === hx.expectedTimeBands, "heads: the pin survives redrawing at n_head = 6",
     `${hx.bandsAfterRedraw} (want ${hx.expectedTimeBands})`);
  ok(hx.bandsAfterTheme === hx.expectedTimeBands && hx.nheadAfterTheme === "6", "heads: and survives a theme toggle",
     `${hx.bandsAfterTheme} bands, n_head ${hx.nheadAfterTheme}`);
  ok(hx.bandsAfterEscape === 0, "heads: Escape clears the widget too", String(hx.bandsAfterEscape));
  ok(hx.bandsOnProseHover > 0 && hx.bandsAfterUnhover === 0, "heads: hovering a prose hd badge lights the widget's head-dim columns",
     `${hx.bandsOnProseHover} on, ${hx.bandsAfterUnhover} after`);
  ok(hx.focusOk && hx.pinnedByKeyboard === "head", "heads: an axis label is a tab stop and Enter pins it", hx.pinnedByKeyboard);
  ok(hx.navs === 0, "heads: arrow keys on a widget label do not change chapter");
}

const tz = run([W_HELP,
  "var w = document.querySelector('[data-viz=\"tensor\"]');",
  "var labs = [].slice.call(w.querySelectorAll('text[data-axis]')).map(function (t) { return t.getAttribute('data-axis'); });",
  "window.tfsAxis.toggle('time'); var t = visibleBands(w); window.tfsAxis.clear();",
  "window.tfsAxis.toggle('batch'); var b = visibleBands(w);",
  "return { labels: labs, timeBands: t, batchBands: b };",
].join("\n"), "ch01-tensors.html");
if (tz.error) ok(false, "tensor scenario ran", tz.error);
else {
  ok(tz.labels.join(",") === "batch,time,feat", "tensor: labels resolve batch / time / channel -> feat (axis identity, not Q/K/V)", tz.labels.join(","));
  ok(tz.timeBands === 4 && tz.batchBands === 2, "tensor: time lights 4 rows, batch lights 2 slabs", `${tz.timeBands} / ${tz.batchBands}`);
}

const ze = run("var w = document.querySelector('[data-viz=\"tensor\"]'); return { labels: [].slice.call(w.querySelectorAll('text[data-axis]')).map(function (t) { return t.getAttribute('data-axis'); }) };",
               "ch04-embeddings.html");
if (ze.error) ok(false, "ch04 tensor scenario ran", ze.error);
else ok(ze.labels.join(",") === "batch,time,feat", "tensor (Ch 4): token -> time, embedding -> feat", ze.labels.join(","));

const at = run([W_HELP,
  "var w = document.querySelector('[data-viz=\"attention\"]');",
  "var marks = [].slice.call(w.querySelectorAll('[data-axis]')).map(function (e) { return e.getAttribute('data-axis'); });",
  "var tok = w.querySelector('text[data-axis=\"time\"]'); tok.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));",
  "var active = document.documentElement.getAttribute('data-ax');",
  "var box = w.querySelector('input[type=checkbox]'); box.click();",          // redraw with the hover still active
  "var stillUnderlined = w.querySelectorAll('text[data-axis=\"time\"][text-decoration=\"underline\"]').length;",
  "return { kinds: Array.from(new Set(marks)), active: active, underlined: stillUnderlined, total: w.querySelectorAll('text[data-axis]').length };",
].join("\n"), "ch05-attention.html");
if (at.error) ok(false, "attention scenario ran", at.error);
else {
  ok(at.kinds.join(",") === "time", "attention: both query and key positions are the time axis, and nothing else", at.kinds.join(","));
  ok(at.active === "time", "attention: hovering a token label drives the page bus", at.active);
  ok(at.underlined === at.total && at.total > 0, "attention: still applied after a redraw", `${at.underlined}/${at.total}`);
}

const gq = run([W_HELP,
  "var w = document.querySelector('[data-viz=\"gqa\"]');",
  "window.tfsAxis.toggle('head');",
  "var before = visibleBands(w);",
  "var sl = w.querySelector('input[type=range]'); sl.value = '0'; sl.dispatchEvent(new Event('input', { bubbles: true }));",   // MQA: 1 kv head
  "return { before: before, after: visibleBands(w) };",
].join("\n"), "ch16-gqa-mla.html");
if (gq.error) ok(false, "gqa scenario ran", gq.error);
else ok(gq.before === 8 + 4 && gq.after === 8 + 1, "gqa: head lights every query and KV head, through the slider",
        `${gq.before} -> ${gq.after} (want 12 -> 9)`);

const tc = run([
  "var w = document.querySelector('[data-viz=\"trace\"]');",
  "var nx = w.querySelector('.btn.primary'); for (var k = 0; k < 7; k++) nx.click();",
  "var shape = [].slice.call(w.querySelectorAll('.shp b[data-axis]')).map(function (b) { return b.textContent + ':' + b.getAttribute('data-axis'); });",
  "window.tfsAxis.toggle('time');",
  "var t = [].slice.call(w.querySelectorAll('b[data-axis=\"time\"]')).map(function (b) { return getComputedStyle(b).opacity; });",
  "nx.click();",
  "var next = [].slice.call(w.querySelectorAll('b[data-axis=\"time\"]')).map(function (b) { return getComputedStyle(b).opacity; });",
  "var focusable = w.querySelectorAll('.shp[tabindex=\"0\"]').length;",
  "return { shape: shape, timeOpacity: t, nextStepTime: next, focusable: focusable };",
].join("\n"), "ch09-architectures.html");
if (tc.error) ok(false, "trace scenario ran", tc.error);
else {
  ok(tc.shape.slice(0, 4).join(" ") === "2:batch 8:head 6:time 64:headdim", "trace: (2, 8, 6, 64) resolves through the page's declared config",
     tc.shape.slice(0, 4).join(" "));
  ok(tc.timeOpacity.length > 0 && tc.timeOpacity.every(o => o === "1"), "trace: pinned time lights in the shape line and the explanation",
     tc.timeOpacity.join(","));
  ok(tc.nextStepTime.length > 0 && tc.nextStepTime.every(o => o === "1"), "trace: the next step's badges arrive already lit");
  ok(tc.focusable > 0, "trace: the redrawn badges get keyboard focus (tfsEnhance)", String(tc.focusable));
}

/* ------------------------------- sidebar: parts and sections fold -------- */
console.log("\n sidebar folding");
const SIDE = [
  "var r = {};",
  "var parts = [].slice.call(document.querySelectorAll('.sb-part'));",
  "var openOf = function (p) { return !p.querySelector('.sb-part-body').hidden; };",
  "r.n = parts.length;",
  "r.openAtStart = parts.map(openOf);",
  "r.hereOpen = openOf(document.querySelector('.sb-part.has-here'));",
  "r.colours = parts.map(function (p) { return getComputedStyle(p.querySelector('.sb-part-h')).backgroundColor; });",
  "var other = parts.filter(function (p) { return !p.classList.contains('has-here'); })[0];",
  "other.querySelector('.sb-part-h').click();",
  "r.otherOpened = openOf(other) && other.querySelector('.sb-part-h').getAttribute('aria-expanded') === 'true';",
  "var here = document.querySelector('.sb-part.has-here');",
  "here.querySelector('.sb-part-h').click();",
  "r.hereFolded = !openOf(here);",
  "r.saved = localStorage.getItem('tfs-side-parts');",
  "here.querySelector('.sb-part-h').click();",
  "var t = $('.sb-sec-toggle');",
  "r.secsOpen = !$('#sbs').hidden;",
  "t.click();",
  "r.secsFolded = $('#sbs').hidden && t.getAttribute('aria-expanded') === 'false';",
  "r.secsSaved = localStorage.getItem('tfs-side-secs');",
  "var q = $('#sbq'); q.value = 'flashattention'; q.dispatchEvent(new Event('input'));",
  "var fa = [].slice.call(document.querySelectorAll('.sb-ch')).filter(function (a) { return a.style.display !== 'none'; });",
  "r.filterHit = fa.length === 1 && !fa[0].closest('.sb-part-body').hidden;",
  "q.value = ''; q.dispatchEvent(new Event('input'));",
  "r.afterClear = parts.map(openOf);",
  "return r;",
].join("\n");
const sd = run(SIDE, "ch06-mha.html");
if (sd.error) ok(false, "sidebar scenario ran", sd.error);
else {
  ok(sd.n === 8, "sidebar: one group per part", String(sd.n));
  ok(sd.hereOpen && sd.openAtStart.filter(Boolean).length === 1, "sidebar: only the part you are reading starts open",
     sd.openAtStart.map(Number).join(""));
  ok(new Set(sd.colours).size === 8, "sidebar: every part band has its own colour", String(new Set(sd.colours).size));
  ok(sd.otherOpened, "sidebar: clicking a folded part's band opens it");
  ok(sd.hereFolded, "sidebar: the current part can be folded too");
  ok(/"2":"closed"/.test(sd.saved || "") && /"0":"open"/.test(sd.saved || ""), "sidebar: open and folded parts are remembered", sd.saved);
  ok(sd.secsOpen && sd.secsFolded && sd.secsSaved === "closed", "sidebar: the chapter's section list folds and is remembered");
  ok(sd.filterHit, "sidebar: filtering opens the folded part that holds the match");
  ok(sd.afterClear.filter(Boolean).length === sd.openAtStart.filter(Boolean).length + 1, "sidebar: clearing the filter restores the folds",
     sd.afterClear.map(Number).join(""));
}
// a reload with remembered state: part 0 stays open, the sections stay folded
const sd2 = run([
  "var parts = [].slice.call(document.querySelectorAll('.sb-part'));",
  "return { open: parts.map(function (p) { return !p.querySelector('.sb-part-body').hidden; }), secs: $('#sbs').hidden };",
].join("\n"), "ch06-mha.html", { "tfs-side-parts": '{"0":"open"}', "tfs-side-secs": "closed" });
if (sd2.error) ok(false, "sidebar reload scenario ran", sd2.error);
else ok(sd2.open[0] && sd2.open[2] && sd2.open.filter(Boolean).length === 2 && sd2.secs,
        "sidebar: remembered state is restored on the next visit", sd2.open.map(Number).join("") + " secs folded=" + sd2.secs);

/* ------------------------------- Phase 3: the shape inspector ------------ */
console.log("\n shape inspector");
const INSP = [
  "var r = {};",
  "var p = document.getElementById('inspector'), btn = $('.insp-toggle');",
  "r.exists = !!p && !!btn;",
  "r.openOnWide = !p.hidden && document.body.classList.contains('insp-open') && btn.getAttribute('aria-expanded') === 'true';",
  "var main = $('main'), wrapR = getComputedStyle($('.wrap')).paddingRight;",
  "r.reservesColumn = parseFloat(wrapR) > 250;",
  "r.rows = [].slice.call(p.querySelectorAll('.insp-ax')).map(function (b) { return b.getAttribute('data-axis') + ':' + b.querySelector('.sym').textContent; }).join(' ');",
  // pin from the key, as a reader would
  "var t = p.querySelector('.insp-ax[data-axis=\"time\"]'); t.click();",
  "r.pinnedFromKey = t.getAttribute('aria-pressed') === 'true' && (document.documentElement.getAttribute('data-ax-pinned') || '').indexOf('time') !== -1;",
  "r.textLit = [].slice.call(main.querySelectorAll('b[data-axis=\"time\"]')).slice(0, 5).every(function (b) { return getComputedStyle(b).opacity === '1'; });",
  "r.clearShown = !p.querySelector('.insp-clear').hidden;",
  "t.click();",
  "r.unpinned = t.getAttribute('aria-pressed') === 'false';",
  // move to section 6.3 and read the live list
  "var h = [].slice.call(main.querySelectorAll(':scope > h2'))[2];",
  "window.scrollTo(0, h.getBoundingClientRect().top + window.scrollY - 100); tfsInspector.refresh();",
  "r.where = p.querySelector('.insp-where').textContent;",
  "r.listed = [].slice.call(p.querySelectorAll('.insp-shapes li .shp')).map(function (s) { return s.textContent.replace(/\\s+/g, ''); });",
  "r.badgesLive = p.querySelectorAll('.insp-shapes [data-axis]').length > 0 && p.querySelector('.insp-shapes .shp').getAttribute('tabindex') === '0';",
  "r.timeCount = t.querySelector('.ct').textContent;",
  // jump
  "p.querySelector('.insp-jump').click();",
  "r.flashed = main.querySelectorAll('.shp.insp-flash').length === 1;",
  // a widget redrawing its shapes in this section is picked up (debounced)
  "var hh = [].slice.call(main.querySelectorAll(':scope > h2'))[3];",
  "var add = document.createElement('p'); add.innerHTML = '<span class=\"shp\">(<b data-axis=\"batch\">B</b>, <b data-axis=\"vocab\">V</b>)</span>';",
  "main.insertBefore(add, hh);",
  // the observer runs after this script yields (real timer: setTimeout is faked),
  // then its 200ms debounce is a FAKE timer, so advance the clock past it
  "return new Promise(function (res) { window.__realTimeout(function () {",
  "  window.__advance(400);",
  "  r.picksUpRedraw = [].slice.call(p.querySelectorAll('.insp-shapes li .shp')).some(function (s) { return s.textContent === '(B, V)'; });",
  "  p.querySelector('.insp-close').click();",
  "  r.closed = p.hidden && !document.body.classList.contains('insp-open') && localStorage.getItem('tfs-inspector') === 'closed';",
  "  res(r);",
  "}, 50); });",
].join("\n");
const ins = run(INSP, "ch06-mha.html");
if (ins.error) ok(false, "inspector scenario ran", ins.error);
else {
  ok(ins.exists, "inspector: built, with a topbar toggle");
  ok(ins.openOnWide, "inspector: open by default on a wide screen");
  ok(ins.reservesColumn, "inspector: reserves its column (text reflows, is not covered)");
  ok(ins.rows === "batch:B time:T feat:C head:nh headdim:hd vocab:V", "inspector: the key lists every axis with its letter", ins.rows);
  ok(ins.pinnedFromKey && ins.textLit, "inspector: pressing an axis in the key pins it and lights it in the text");
  ok(ins.clearShown, "inspector: 'clear pinned axes' appears while something is pinned");
  ok(ins.unpinned, "inspector: pressing it again unpins");
  ok(/^6\.3/.test(ins.where), "inspector: follows the section you are reading", ins.where);
  ok(ins.listed.length > 0 && new Set(ins.listed).size === ins.listed.length, "inspector: lists that section's shapes, de-duplicated",
     ins.listed.slice(0, 4).join(" "));
  ok(ins.badgesLive, "inspector: listed shapes are live, focusable badges");
  ok(/^×\d+$/.test(ins.timeCount), "inspector: per-axis counts for the section", ins.timeCount);
  ok(ins.flashed, "inspector: jump marks the shape it scrolled to");
  ok(ins.picksUpRedraw, "inspector: picks up shapes a widget draws later");
  ok(ins.closed, "inspector: closes, and remembers that");
}
const ins2 = run([
  "var p = document.getElementById('inspector');",
  "return { hidden: p.hidden, pad: parseFloat(getComputedStyle($('.wrap')).paddingRight) };",
].join("\n"), "ch06-mha.html", { "tfs-inspector": "closed" });
if (ins2.error) ok(false, "inspector reload scenario ran", ins2.error);
else ok(ins2.hidden && ins2.pad < 100, "inspector: stays closed on the next page, and gives the column back", JSON.stringify(ins2));

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
