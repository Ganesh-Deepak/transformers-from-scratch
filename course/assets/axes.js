/* ============================================================================
   AXIS IDENTITY — the data model behind the tensor-shape highlighter.

   THE DESIGN DECISION THIS FILE ENCODES
   -------------------------------------
   Colour used to mean "which tensor is this" (.shape.q -> every <b> orange).
   It now means "which AXIS is this". The difference matters because the two
   encode different amounts of information:

     tensor identity  answers "which tensor?"  -- but you already know, the
                      variable name is sitting right next to it (q, k, v).
     axis identity    answers "which axis, and where did it go?" -- and that
                      is NOT recoverable from context, because axes are
                      positional and unnamed. (B,T,C) and (B,C,T) differ only
                      by position.

   Colour is the scarce channel, so it buys the scarce information. Tensor
   identity keeps living on .shape.q/.k/.v/.o, which now drive a non-colour
   marker instead of the fill.

   Consequence you can actually use: T is blue in Q, in K and in V, so across
   .view(B,T,nh,hd).transpose(1,2) you watch the blue block slide from
   position 1 to position 2. That is the reshape becoming visible.

   FIVE HUES IS THE CEILING
   ------------------------
   The four existing --dim-* plus a teal for head-dim sit at hues
   20 / 90 / 175 / 218 / 324 degrees, with the violet accent at 276. Every
   remaining gap is under 45 degrees or already spoken for by --warn (55) or
   --stop (5). A sixth hue measured 12-19 degrees from --dim-2, which is
   precisely the "two axes look like the same axis" failure the colour system
   exists to prevent. So `vocab` and anything unrecognised get a NON-HUE
   treatment (neutral ink, dotted underline) rather than a bad sixth colour.
   ========================================================================= */

var TFS_AXES = {
  /* key      token used for colour     what it is
     ------------------------------------------------------------------ */
  batch:   { css: "--dim-1", label: "batch",     blurb: "independent sequences in the mini-batch" },
  time:    { css: "--dim-2", label: "time",      blurb: "position in the sequence" },
  feat:    { css: "--dim-3", label: "feature",   blurb: "the residual-stream width" },
  head:    { css: "--dim-4", label: "head",      blurb: "attention heads, independent of each other" },
  headdim: { css: "--dim-5", label: "head dim",  blurb: "features belonging to one head" },
  vocab:   { css: null,      label: "vocab",     blurb: "logits over the token vocabulary" },
};

/* Every written form that means the same axis. Longest-first matching is done
   by the consumer; these are compared case-SENSITIVELY except where a lower
   case word form is listed, because `T` (time) and `t` (a loop variable) are
   not the same thing and `B` (batch) and `b` (a bias) are not either. */
var TFS_AXIS_SYMBOLS = {
  batch:   ["B", "batch"],
  time:    ["T", "T_q", "T_kv", "S", "L", "time", "token", "tokens", "seq", "n_ctx"],
  feat:    ["C", "d_model", "n_embd", "E", "feature", "features", "channel", "channels", "embedding", "embed", "d_embed"],
  head:    ["H", "nh", "n_head", "n_heads", "head", "heads", "n_kv_head", "n_kv_heads"],
  headdim: ["hd", "d_head", "head_dim", "d_k", "d_v", "hs"],
  vocab:   ["V", "n_vocab", "vocab"],
};

/* symbol -> axis key, built once */
var TFS_SYMBOL_TO_AXIS = (function () {
  var m = {};
  for (var k in TFS_AXIS_SYMBOLS) {
    for (var i = 0; i < TFS_AXIS_SYMBOLS[k].length; i++) m[TFS_AXIS_SYMBOLS[k][i]] = k;
  }
  return m;
})();

/* Resolve one part of a shape expression to an axis key, or null.
   Handles the three decorations that show up in real code:
     self.nh  -> nh   (attribute access on the module)
     3C       -> C    (the fused qkv projection is 3 x the feature width)
     C//H     -> handled by the caller, which splits on / first
   A bare number is a concrete extent, not an axis: returns null so it renders
   plain. That is deliberate -- we cannot know WHICH axis a literal 12 is. */
function tfsResolveAxis(part) {
  if (!part) return null;
  var p = String(part).trim();
  if (/^\d+$/.test(p)) return null;                 // literal extent
  p = p.replace(/^(?:self|cfg|config)\./, "");      // self.nh -> nh
  p = p.replace(/^\d+(?=[A-Za-z_])/, "");           // 3C -> C
  return TFS_SYMBOL_TO_AXIS[p] || null;
}

/* ------------------------------------------------------------------ the bus
   One active-axis set, published as `data-ax` on <html> as a space-separated
   list. CSS does all the work from there via [data-ax~="time"] -- no
   per-element class toggling, so hovering does not walk 150 nodes.

   hover is transient; pins persist (that is what makes it work on touch and
   for anyone who cannot hold a hover steady). */
(function () {
  if (typeof document === "undefined") return;      // required from node

  var pinned = [];        // axis keys, explicitly locked on
  var hovered = null;     // axis key, transient

  function publish() {
    var active = pinned.slice();
    if (hovered && active.indexOf(hovered) === -1) active.push(hovered);
    var root = document.documentElement;
    if (active.length) root.setAttribute("data-ax", active.join(" "));
    else root.removeAttribute("data-ax");
    root.setAttribute("data-ax-pinned", pinned.length ? pinned.join(" ") : "");
    document.querySelectorAll("[data-axis]").forEach(function (el) {
      var a = el.getAttribute("data-axis");
      el.setAttribute("aria-pressed", pinned.indexOf(a) !== -1 ? "true" : "false");
    });
    document.dispatchEvent(new CustomEvent("tfs:axis", { detail: { active: active, pinned: pinned.slice() } }));
  }

  window.tfsAxis = {
    hover: function (a) { hovered = a; publish(); },
    unhover: function (a) { if (hovered === a || a == null) hovered = null; publish(); },
    toggle: function (a) {
      var i = pinned.indexOf(a);
      if (i === -1) pinned.push(a); else pinned.splice(i, 1);
      publish();
    },
    clear: function () { pinned = []; hovered = null; publish(); },
    active: function () { var a = pinned.slice(); if (hovered && a.indexOf(hovered) === -1) a.push(hovered); return a; },
    pinned: function () { return pinned.slice(); },
  };

  function axisOf(node) {
    var el = node && node.closest ? node.closest("[data-axis]") : null;
    return el ? el.getAttribute("data-axis") : null;
  }

  document.addEventListener("mouseover", function (e) {
    var a = axisOf(e.target); if (a) window.tfsAxis.hover(a);
  });
  document.addEventListener("mouseout", function (e) {
    var a = axisOf(e.target); if (a) window.tfsAxis.unhover(a);
  });
  /* CAPTURE phase, and it stops propagation. Quiz options bind their own
     onclick directly on .opt (course.js), and a quiz locks on first answer.
     A bubble-phase listener here runs AFTER that handler, so tapping a shape
     badge inside an option to inspect an axis would silently lock in that
     answer. Capturing lets the axis click be handled without ever reaching
     the option. */
  document.addEventListener("click", function (e) {
    var el = e.target.closest ? e.target.closest("[data-axis]") : null;
    if (!el) return;
    e.preventDefault();
    e.stopPropagation();
    window.tfsAxis.toggle(el.getAttribute("data-axis"));
  }, true);

  /* Keyboard: the BADGE is the tab stop, not each axis -- ch06 alone has 41
     occurrences of T, and 41 extra tab stops would make the page hostile to
     anyone using a keyboard. Inside a focused badge, Left/Right move a cursor
     across its axes (the standard composite-widget pattern), Enter/Space pins,
     Escape clears everything. */
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { window.tfsAxis.clear(); return; }
    var badge = e.target.closest ? e.target.closest(".shp") : null;
    if (!badge) return;
    var parts = [].slice.call(badge.querySelectorAll("[data-axis]"));
    if (!parts.length) return;
    var cur = parts.indexOf(badge.querySelector("[data-axis].ax-cursor"));
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      var next = cur === -1 ? (e.key === "ArrowRight" ? 0 : parts.length - 1)
                            : (cur + (e.key === "ArrowRight" ? 1 : -1) + parts.length) % parts.length;
      parts.forEach(function (p) { p.classList.remove("ax-cursor"); });
      parts[next].classList.add("ax-cursor");
      window.tfsAxis.hover(parts[next].getAttribute("data-axis"));
    } else if (e.key === "Enter" || e.key === " ") {
      if (cur === -1) return;
      e.preventDefault();
      window.tfsAxis.toggle(parts[cur].getAttribute("data-axis"));
    }
  });

  /* ------------------------------------------------- runtime annotation
     Code blocks CANNOT carry baked-in markup: course.js highlights each
     <pre><code> by reading textContent and reassigning innerHTML, which
     strips any span put there at build time. So we annotate them here, after
     highlighting has run. Same whitelist as the build-time pass -- an
     expression is only wrapped if every non-numeric part is a known axis,
     which is what keeps `def forward(self, x)` and `(q, k, v)` out of it.

     Walking TEXT NODES (not innerHTML) means the syntax-highlight spans
     around them are left completely intact. */
  var SHAPE_RE = /([(\[])\s*([A-Za-z_][A-Za-z0-9_.]*(?:\s*\/\/?\s*[A-Za-z0-9_.]+)?(?:\s*,\s*[A-Za-z0-9_.*\/\-]+)+)\s*([)\]])/g;

  function resolvePart(p) {
    var direct = tfsResolveAxis(p);
    if (direct) return direct;
    var q = p.match(/^([A-Za-z_][A-Za-z0-9_.]*)\s*\/\/?\s*([A-Za-z_][A-Za-z0-9_.]*)$/);
    if (q && tfsResolveAxis(q[1]) === "feat" && tfsResolveAxis(q[2]) === "head") return "headdim";
    return null;
  }

  function annotateString(text) {
    return text.replace(SHAPE_RE, function (m, open, inner, close) {
      var bits = inner.split(/(\s*,\s*)/);
      var parts = bits.filter(function (_, i) { return i % 2 === 0; });
      if (parts.length < 2 || parts.length > 6) return m;
      var symbolic = 0, ok = true;
      var axes = parts.map(function (p) {
        var t = p.trim();
        if (/^\d+$/.test(t)) return null;
        var a = resolvePart(t);
        if (!a) ok = false; else symbolic++;
        return a;
      });
      if (!ok || !symbolic) return m;
      var out = "", pi = 0;
      for (var i = 0; i < bits.length; i++) {
        if (i % 2 === 1) { out += bits[i]; continue; }
        var a = axes[pi++];
        out += a ? '<b data-axis="' + a + '">' + bits[i] + "</b>" : bits[i];
      }
      return '<span class="shp">' + open + out + close + "</span>";
    });
  }

  function tfsAnnotate(root) {
    var blocks = (root || document).querySelectorAll("pre code");
    var n = 0;
    blocks.forEach(function (code) {
      var walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT, null);
      var nodes = [], t;
      while ((t = walker.nextNode())) nodes.push(t);
      nodes.forEach(function (node) {
        if (node.parentElement.closest(".shp")) return;
        var html = annotateString(
          node.nodeValue.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"));
        if (html === node.nodeValue.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")) return;
        var frag = document.createElement("span");
        frag.innerHTML = html;
        node.parentNode.replaceChild(frag, node);
        n += frag.querySelectorAll(".shp").length;
      });
    });
    return n;
  }
  window.tfsAnnotate = tfsAnnotate;

  /* Focus affordances are added by JS, not baked into the markup: without JS
     the badges do nothing, so making them tab stops would be a trap. This is
     the progressive-enhancement seam — colours are CSS, interaction is here. */
  function enhance() {
    tfsAnnotate(document);
    document.querySelectorAll(".shp").forEach(function (badge) {
      if (badge.hasAttribute("tabindex")) return;
      badge.setAttribute("tabindex", "0");
      badge.setAttribute("role", "group");
      var names = [].slice.call(badge.querySelectorAll("[data-axis]"))
        .map(function (b) { return (TFS_AXES[b.getAttribute("data-axis")] || {}).label || b.textContent; });
      badge.setAttribute("aria-label",
        "tensor shape " + badge.textContent.trim() +
        (names.length ? "; axes " + names.join(", ") + "; arrow keys to select, enter to pin" : ""));
    });
    document.querySelectorAll("[data-axis]").forEach(function (b) {
      b.setAttribute("role", "button");
      b.setAttribute("aria-pressed", "false");
      var a = TFS_AXES[b.getAttribute("data-axis")];
      if (a) b.setAttribute("title", a.label + " — " + a.blurb);
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", enhance);
  else enhance();
  /* Widgets redraw their own DOM (a slider move, a theme change), which
     creates fresh badges after enhance() has run. They call this to give the
     new badges the same focus and keyboard affordances. Idempotent. */
  window.tfsEnhance = enhance;

  document.addEventListener("focusout", function (e) {
    var badge = e.target.closest ? e.target.closest(".shp") : null;
    if (!badge) return;
    setTimeout(function () {
      if (badge.contains(document.activeElement)) return;
      badge.querySelectorAll(".ax-cursor").forEach(function (p) { p.classList.remove("ax-cursor"); });
      window.tfsAxis.unhover(null);
    }, 0);
  });
})();

/* ============================================================================
   PHASE 3 — THE SHAPE INSPECTOR: the axis key, always one click away.

   WHERE IT LIVES, AND WHY
   The choice was "margin column or pinned". There is no margin column any
   more: the prose takes 75% of the width and margin notes sit inline. A
   floating box would cover text. So it is a DOCKED panel, the mirror image
   of the chapter sidebar:
     - wide screens (>= 64rem): a right-hand rail that RESERVES its column,
       so the text reflows beside it instead of being covered;
     - narrow screens: a bottom sheet over the page, closed by default.
   Toggled from the topbar ("Shapes"), remembered across pages.

   WHAT IT SHOWS
     1. The key: every axis with its letter, its name, its meaning and how
        often it appears in the section you are reading. Not colour-only: the
        letter and the name carry the meaning in greyscale. Each row is a
        real [data-axis] control, so the existing bus does the work: hover
        previews, click / tap / Enter pins, Escape clears.
     2. The live shapes of the section you are reading, de-duplicated, as
        real badges (same keyboard composite as in the text) with a jump
        button. "Live": it follows your scroll and re-reads the section when
        a widget redraws its shapes.
   Without JS none of this exists, and nothing needs it: the badges in the
   text are coloured by CSS alone.
   ========================================================================= */
(function () {
  if (typeof document === "undefined") return;

  var SYM = { batch: "B", time: "T", feat: "C", head: "nh", headdim: "hd", vocab: "V" };
  var ORDER = ["batch", "time", "feat", "head", "headdim", "vocab"];
  var KEY = "tfs-inspector";
  var WIDE = "(min-width: 64rem)";

  function store(v) {
    try { if (v === undefined) return localStorage.getItem(KEY); localStorage.setItem(KEY, v); } catch (e) { return null; }
  }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  function visible(el) { return el.getClientRects().length > 0; }

  function build() {
    var main = document.querySelector("main");
    var bar = document.querySelector(".topbar");
    if (!main || !bar || document.getElementById("inspector")) return;
    if (!main.querySelector(".shp [data-axis], [data-viz] [data-axis]")) return;   // a page with no shapes needs no key

    /* ---- the topbar toggle, next to Focus ---- */
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "insp-toggle";
    btn.setAttribute("aria-controls", "inspector");
    btn.innerHTML = '<span class="insp-ico" aria-hidden="true">(B,T)</span><span class="insp-lbl">Shapes</span>';
    btn.title = "Shape inspector: the axis key and this section's shapes";
    var anchor = bar.querySelector(".focus-toggle") || bar.querySelector(".prog") || bar.querySelector(".theme-toggle");
    bar.insertBefore(btn, anchor);

    /* ---- the panel ---- */
    var panel = document.createElement("aside");
    panel.className = "inspector";
    panel.id = "inspector";
    panel.setAttribute("aria-label", "Shape inspector");
    panel.innerHTML =
      '<div class="insp-head"><div><div class="insp-k">Shape key</div><div class="insp-where"></div></div>' +
      '<button type="button" class="insp-close" aria-label="Close the shape key">&times;</button></div>' +
      '<div class="insp-axes" role="group" aria-label="Axes: press one to light it everywhere">' +
      ORDER.map(function (k) {
        var a = TFS_AXES[k];
        return '<button type="button" class="insp-ax" data-axis="' + k + '">' +
          '<span class="sym">' + SYM[k] + '</span>' +
          '<span class="nm">' + esc(a.label) + '</span>' +
          '<span class="ct" title="appearances in this section"></span>' +
          '<span class="bl">' + esc(a.blurb) + '</span></button>';
      }).join("") + "</div>" +
      '<div class="insp-sub">In this section <span class="insp-n"></span></div>' +
      '<ol class="insp-shapes"></ol>' +
      '<p class="insp-empty" hidden>No tensor shapes in this section.</p>' +
      '<div class="insp-foot"><button type="button" class="insp-clear" hidden>Clear pinned axes</button>' +
      '<p class="insp-hint">Hover, tap or Tab to any axis to light it everywhere. Enter pins it; Esc clears.</p></div>';
    document.body.appendChild(panel);

    var where = panel.querySelector(".insp-where");
    var list = panel.querySelector(".insp-shapes");
    var empty = panel.querySelector(".insp-empty");
    var countEl = panel.querySelector(".insp-n");
    var clearBtn = panel.querySelector(".insp-clear");

    /* ---- open / closed ---- */
    function setOpen(open, remember) {
      panel.hidden = !open;
      btn.setAttribute("aria-expanded", String(open));
      document.body.classList.toggle("insp-open", open);
      if (remember) store(open ? "open" : "closed");
      if (open) refresh(true);
      if (window.layoutMarginNotes) requestAnimationFrame(window.layoutMarginNotes);
    }
    var saved = store();
    setOpen(saved ? saved === "open" : matchMedia(WIDE).matches, false);
    function animate() {
      document.body.classList.add("side-anim");
      setTimeout(function () { document.body.classList.remove("side-anim"); }, 400);
    }
    btn.addEventListener("click", function () { animate(); setOpen(panel.hidden, true); });
    panel.querySelector(".insp-close").addEventListener("click", function () { animate(); setOpen(false, true); btn.focus(); });
    clearBtn.addEventListener("click", function () { window.tfsAxis.clear(); });
    document.addEventListener("keydown", function (e) {
      // Esc closes the SHEET on a phone (it covers the page); the rail stays
      if (e.key === "Escape" && !panel.hidden && !matchMedia(WIDE).matches) setOpen(false, true);
    });

    /* ---- which section are you reading? ---- */
    function sections() {
      return [].slice.call(main.children).filter(function (c) { return c.tagName === "H2" && visible(c); });
    }
    function currentIndex() {
      var hs = sections(), line = window.innerHeight * 0.35, idx = -1;
      for (var i = 0; i < hs.length; i++) if (hs[i].getBoundingClientRect().top < line) idx = i;
      return { idx: idx, h: idx >= 0 ? hs[idx] : null };
    }

    /* every visible shape in section idx (-1 = before the first heading) */
    function shapesIn(h) {
      var out = [], inside = !h;
      [].slice.call(main.children).forEach(function (c) {
        if (c.tagName === "H2" && visible(c)) { inside = c === h; return; }
        if (!inside) return;
        var found = c.matches(".shp") ? [c] : [].slice.call(c.querySelectorAll(".shp"));
        found.forEach(function (s) { if (visible(s) && s.querySelector("[data-axis]")) out.push(s); });
      });
      return out;
    }

    var lastKey = null;
    function refresh(force) {
      if (panel.hidden) return;
      var cur = currentIndex();
      var shapes = shapesIn(cur.h);
      var key = (cur.h ? cur.h.id : "intro") + "|" + shapes.map(function (s) { return s.textContent; }).join("|");
      if (!force && key === lastKey) return;
      lastKey = key;

      where.textContent = cur.h ? cur.h.textContent.replace(/\s+/g, " ").trim() : "Introduction";

      // per-axis counts for this section
      var counts = {};
      shapes.forEach(function (s) {
        s.querySelectorAll("[data-axis]").forEach(function (b) {
          var a = b.getAttribute("data-axis"); counts[a] = (counts[a] || 0) + 1;
        });
      });
      panel.querySelectorAll(".insp-ax").forEach(function (b) {
        var n = counts[b.getAttribute("data-axis")] || 0;
        b.querySelector(".ct").textContent = n ? "×" + n : "–";
        b.classList.toggle("absent", !n);
      });

      // de-duplicated shapes, first occurrence wins, with a multiplicity
      var seen = {}, uniq = [];
      shapes.forEach(function (s) {
        // spacing is not meaning: (B, T, C) and (B,T,C) are one shape
        var t = s.textContent.replace(/\s+/g, " ").trim(), k = t.replace(/\s+/g, "");
        if (seen[k]) { seen[k].n++; return; }
        seen[k] = { el: s, n: 1, text: t };
        uniq.push(seen[k]);
      });
      list.innerHTML = "";
      uniq.slice(0, 24).forEach(function (u) {
        var li = document.createElement("li");
        var clone = u.el.cloneNode(true);
        clone.querySelectorAll(".ax-cursor").forEach(function (x) { x.classList.remove("ax-cursor"); });
        clone.removeAttribute("id");
        li.appendChild(clone);
        if (u.n > 1) {
          var m = document.createElement("span");
          m.className = "insp-mult"; m.textContent = "×" + u.n;
          m.title = "appears " + u.n + " times in this section";
          li.appendChild(m);
        }
        var jump = document.createElement("button");
        jump.type = "button"; jump.className = "insp-jump";
        jump.setAttribute("aria-label", "Jump to " + u.text);
        jump.innerHTML = '<span aria-hidden="true">→</span>';
        jump.addEventListener("click", function () {
          var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
          u.el.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
          u.el.classList.add("insp-flash");
          setTimeout(function () { u.el.classList.remove("insp-flash"); }, 1600);
          if (!matchMedia(WIDE).matches) setOpen(false, false);   // the sheet would hide it
        });
        li.appendChild(jump);
        list.appendChild(li);
      });
      countEl.textContent = uniq.length ? "(" + uniq.length + (uniq.length > 24 ? ", first 24" : "") + ")" : "";
      empty.hidden = uniq.length > 0;
      list.hidden = uniq.length === 0;
    }

    /* follow the reader: section changes on scroll, shapes change when a
       widget redraws (childList only: the bus rewrites aria-pressed on every
       badge, and watching attributes would refresh on every hover) */
    var raf = 0;
    window.addEventListener("scroll", function () {
      cancelAnimationFrame(raf); raf = requestAnimationFrame(function () { refresh(false); });
    }, { passive: true });
    var mt = 0;
    new MutationObserver(function () {
      clearTimeout(mt); mt = setTimeout(function () { refresh(false); }, 200);
    }).observe(main, { childList: true, subtree: true });
    document.addEventListener("tfs:axis", function (e) {
      clearBtn.hidden = !e.detail.pinned.length;
    });
    window.addEventListener("resize", function () { refresh(true); });

    window.tfsInspector = { refresh: function () { refresh(true); }, open: function (v) { setOpen(v !== false, false); } };
    refresh(true);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = { TFS_AXES: TFS_AXES, TFS_AXIS_SYMBOLS: TFS_AXIS_SYMBOLS, tfsResolveAxis: tfsResolveAxis };
}
