/* ============================================================
   viz.js — interactive tensor / matrix / attention visualisations.
   Pure SVG + DOM, no dependencies.

   Usage in a chapter:
     <div data-viz="matmul" data-m="4" data-k="3" data-n="5"></div>
   ============================================================ */

const NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
};
/* Colours come from the stylesheet's tokens so the widgets follow the theme.
   Re-read on every render; `themechange` triggers a rebuild (see bottom). */
const C = {};
function readTheme() {
  const s = getComputedStyle(document.documentElement);
  const v = (n, fb) => (s.getPropertyValue(n) || "").trim() || fb;
  Object.assign(C, {
    // Axis colours by AXIS NAME. There are deliberately no q/k/v/o keys any
    // more: under the axis-identity model an axis colour may only ever mean
    // its axis, so a widget has to say which axis it is drawing.
    axis: {
      batch:   v("--dim-1", "#C25D35"),
      time:    v("--dim-2", "#1B64DE"),
      feat:    v("--dim-3", "#B815B0"),
      head:    v("--dim-4", "#43590F"),
      headdim: v("--dim-5", "#08897A"),
    },
    stamp:  v("--stamp", "#5A241F"),
    accent: v("--accent", "#6A2E93"),
    grid:   v("--line", "#C9B492"),
    gridHi: v("--line-2", "#8C5F4B"),
    cell:   v("--surface-2", "#DCCBA9"),
    paper:  v("--paper", "#E7D8BC"),
    fg:     v("--ink", "#2B1512"),
    dim:    v("--ink-2", "#563A31"),
    faint:  v("--ink-3", "#775446"),
    fainter:v("--ink-4", "#9A7B66"),
    danger: v("--stop", "#A82A16"),
    ok:     v("--ok", "#2F6B22"),
  });
  return C;
}
/* deterministic pseudo-random so numbers are stable across reloads (mulberry32) */
function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/* small non-zero integers in [-3,3]\{0} — zeros make the dot-product demo unreadable */
const smallInt = r => { const v = Math.floor(r() * 6) - 3; return v >= 0 ? v + 1 : v; };

function panel(el, title, sub) {
  el.classList.add("panel");
  if (title) {
    const h = document.createElement("h4");
    h.textContent = title;
    h.style.cssText = "margin:0 0 4px;color:var(--ink-2)";
    el.appendChild(h);
  }
  if (sub) {
    const p = document.createElement("div");
    p.className = "tiny"; p.style.marginBottom = "12px"; p.textContent = sub;
    el.appendChild(p);
  }
  return el;
}
function ctlRow(el) { const d = document.createElement("div"); d.className = "ctl"; el.appendChild(d); return d; }
function slider(row, label, min, max, val, oninput) {
  const l = document.createElement("label");
  l.innerHTML = `${label} <input type="range" min="${min}" max="${max}" value="${val}"><output>${val}</output>`;
  row.appendChild(l);
  const inp = l.querySelector("input"), out = l.querySelector("output");
  inp.oninput = () => { out.textContent = inp.value; oninput(+inp.value); };
  return inp;
}
function toggle(row, label, checked, onchange) {
  const l = document.createElement("label");
  l.innerHTML = `<input type="checkbox" ${checked ? "checked" : ""}> ${label}`;
  row.appendChild(l);
  l.querySelector("input").onchange = e => onchange(e.target.checked);
  return l;
}

/* ============================================================
   AXIS COUPLING (Phase 2) — widgets join the axes.js highlight bus.

   An axis colour may appear in a widget ONLY where it means that axis.
   The old Q/K/V/O colour keys (q, k, v, o on the colour table) are gone, so a
   widget cannot quietly keep the tensor-identity model: it must name the
   axis (C.axis.time) or use a non-axis colour (accent, stamp, ink).

   Three kinds of mark a widget can register:
     label  axis-coloured text the reader can hover, tap or focus to drive the
            bus; it emphasises itself when its axis is active
     guide  a bracket/rule showing an axis's extent; thickens when active
     band   an outline over the rows / columns / slices that ARE that axis;
            invisible until its axis is active
   Not colour-only, like the badges: on activation everything else recedes
   (opacity), labels underline and bold, guides double in width, and bands
   appear — each readable with the hue removed.
   ============================================================ */
const AXIS_KEYS = ["batch", "time", "feat", "head", "headdim", "vocab"];
const axisCol = a => (C.axis && C.axis[a]) || C.dim;
const axisName = a => (window.TFS_AXES && TFS_AXES[a] ? TFS_AXES[a].label : a);
const resolveAxis = name => (typeof tfsResolveAxis === "function" ? tfsResolveAxis(name) : null);

/* a shape as badge HTML: parts = [[text, axisKeyOrNull], ...] */
function shapeHTML(parts) {
  return '<span class="shp">(' + parts.map(([t, a]) =>
    a ? `<b data-axis="${a}">${t}</b>` : `${t}`).join(", ") + ")</span>";
}

/* one coupler per widget host, reused across redraws and theme rebuilds so
   the bus listener is attached exactly once */
const COUPLERS = new WeakMap();
function coupler(host) {
  let cp = COUPLERS.get(host);
  if (cp) { cp.reset(); return cp; }
  const marks = [];
  const reduce = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
  const apply = () => {
    const active = window.tfsAxis ? window.tfsAxis.active() : [];
    const any = active.length > 0;
    for (const m of marks) {
      if (!m.node.isConnected) continue;
      const on = active.indexOf(m.axis) !== -1;
      if (m.kind === "band") m.node.setAttribute("opacity", on ? 1 : 0);
      else if (m.kind === "guide") {
        m.node.setAttribute("stroke-width", on ? m.w * 2.2 : m.w);
        m.node.setAttribute("opacity", any && !on ? .28 : 1);
      } else {
        m.node.setAttribute("font-weight", on ? 800 : m.fw);
        m.node.setAttribute("text-decoration", on ? "underline" : "none");
        m.node.setAttribute("opacity", any && !on ? .32 : 1);
      }
    }
  };
  document.addEventListener("tfs:axis", apply);
  cp = {
    reset() { marks.length = 0; },
    apply() { apply(); if (window.tfsEnhance) window.tfsEnhance(); },
    band(node, axis) {
      node.setAttribute("opacity", 0);
      node.setAttribute("pointer-events", "none");
      if (!reduce) node.style.transition = "opacity .12s";
      marks.push({ node, axis, kind: "band" }); return node;
    },
    guide(node, axis) {
      marks.push({ node, axis, kind: "guide", w: +node.getAttribute("stroke-width") || 1.5 });
      return node;
    },
    /* focusable: the few labels that name an axis get a tab stop; repeated
       per-token labels stay mouse/touch only so the tab order stays short */
    label(node, axis, focusable) {
      node.setAttribute("data-axis", axis);
      node.style.cursor = "pointer";
      if (focusable) {
        node.setAttribute("tabindex", "0");
        node.setAttribute("role", "button");
        node.setAttribute("aria-label", `${axisName(axis)} axis — highlight it everywhere on the page`);
        node.addEventListener("focus", () => window.tfsAxis && window.tfsAxis.hover(axis));
        node.addEventListener("blur", () => window.tfsAxis && window.tfsAxis.unhover(axis));
        node.addEventListener("keydown", e => {
          if (e.key !== "Enter" && e.key !== " ") return;
          e.preventDefault();
          if (window.tfsAxis) window.tfsAxis.toggle(axis);
        });
      }
      marks.push({ node, axis, kind: "label", fw: node.getAttribute("font-weight") || 600 });
      return node;
    },
  };
  COUPLERS.set(host, cp);
  return cp;
}

/* SVG helpers shared by the coupled widgets */
function svgText(sv, x, y, txt, col, opt = {}) {
  const t = svgEl("text", { x, y, "text-anchor": opt.anchor || "start", "font-size": opt.size || 11,
    fill: col, "font-family": "var(--mono)", "font-weight": opt.weight || 600 });
  t.textContent = txt; sv.appendChild(t); return t;
}
/* text whose shape parts are axis tspans: parts like [["B", "batch"], ["12", null]] */
function svgShape(sv, cp, x, y, prefix, parts, suffix, opt = {}) {
  const t = svgEl("text", { x, y, "font-size": opt.size || 11, fill: C.dim, "font-family": "var(--mono)", "font-weight": 600 });
  const add = (s, a) => {
    const sp = svgEl("tspan", a ? { fill: axisCol(a) } : {});
    sp.textContent = s; t.appendChild(sp);
    if (a) cp.label(sp, a, false);
  };
  add(prefix + "(");
  parts.forEach(([s, a], i) => { add(s, a); if (i < parts.length - 1) add(", "); });
  add(")" + (suffix || ""));
  sv.appendChild(t); return t;
}
/* a bracket along one side of a block, with its axis label */
function svgBracket(sv, cp, axis, x1, y1, x2, y2, side, text, focusable) {
  const col = axisCol(axis), tick = 5;
  let path, lx, ly, anchor = "middle";
  if (side === "left") { path = `M${x1 + tick} ${y1} H${x1} V${y2} H${x1 + tick}`; lx = x1 - 6; ly = (y1 + y2) / 2 + 4; anchor = "end"; }
  else if (side === "top") { path = `M${x1} ${y1 + tick} V${y1} H${x2} V${y1 + tick}`; lx = (x1 + x2) / 2; ly = y1 - 6; }
  else { path = `M${x1} ${y1 - tick} V${y1} H${x2} V${y1 - tick}`; lx = (x1 + x2) / 2; ly = y1 + 14; }
  cp.guide(sv.appendChild(svgEl("path", { d: path, fill: "none", stroke: col, "stroke-width": 1.6 })), axis);
  cp.label(svgText(sv, lx, ly, text, col, { anchor, size: 11, weight: 700 }), axis, focusable);
}
function svgBand(sv, cp, axis, x, y, w, h) {
  const col = axisCol(axis);
  return cp.band(sv.appendChild(svgEl("rect", { x: x - 1.5, y: y - 1.5, width: w + 1, height: h + 1, rx: 4,
    fill: col, "fill-opacity": .14, stroke: col, "stroke-width": 2.2 })), axis);
}

/* ============================================================
   2. TENSOR — a rank-3 tensor as labelled slabs (coupled)
   ============================================================ */
function vizTensor(el, d) {
  const dims = (d.dims || "2,4,3").split(",").map(Number);
  const names = (d.names || "batch,time,channel").split(",");
  panel(el, d.title || `Tensor of shape (${dims.join(", ")})`,
    d.sub || "Each small square is one number. Rightmost axis is contiguous in memory. Hover or tap an axis label.");
  const cp = coupler(el);
  const wrap = document.createElement("div"); wrap.style.overflowX = "auto"; el.appendChild(wrap);
  const note = document.createElement("div");
  note.style.cssText = "margin-top:10px;font-family:var(--mono);font-size:12.5px;color:var(--ink-2);line-height:1.8";
  el.appendChild(note);
  const [B, T, Ch] = dims.length === 3 ? dims : dims.length === 2 ? [1, ...dims] : [1, 1, dims[0]];
  const ax = (dims.length === 3 ? names : ["", ...names]).map(n => resolveAxis(n));
  const [aB, aT, aC] = ax;
  const cs = 22, dx = 13, dy = -9;
  // left margin sized to the longest left-hand label, so it is never clipped;
  // the stats block sits after the slabs instead of at a fixed offset
  const leftLabel = dims.length === 3 ? `axis 1: ${names[1]} = ${T}` : `${names[0] || "rows"} = ${T}`;
  const ox = 24 + leftLabel.length * 6.8, oy = 58 + (B - 1) * Math.abs(dy);
  const statsX = ox + Ch * cs + (B - 1) * Math.abs(dx) + 34;
  const bottomLabel = dims.length === 3 ? `axis 2: ${names[2]} = ${Ch}  ← contiguous` : "";
  const W = Math.max(statsX + 130, ox + Ch * cs / 2 + bottomLabel.length * 3.5 + 20), H = T * cs + (B - 1) * Math.abs(dy) + 110;
  const sv = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

  for (let b = B - 1; b >= 0; b--) {
    const sx = ox + b * dx, sy = oy + b * dy;
    for (let t = 0; t < T; t++) for (let c = 0; c < Ch; c++) {
      sv.appendChild(svgEl("rect", {
        x: sx + c * cs, y: sy + t * cs, width: cs - 2, height: cs - 2, rx: 3,
        fill: C.cell, stroke: b === 0 ? C.gridHi : C.grid, "stroke-width": b === 0 ? 1.3 : 1, opacity: 1 - b * 0.13,
      }));
    }
    if (B > 1) svgText(sv, sx - 8, sy + 12, `[${b}]`, C.faint, { anchor: "end", size: 10, weight: 400 });
    // batch: each slab IS one batch element
    if (aB) svgBand(sv, cp, aB, sx, sy, Ch * cs - 2, T * cs - 2);
  }
  // time: rows of the front slab; feature: its columns
  for (let t = 0; t < T; t++) if (aT) svgBand(sv, cp, aT, ox, oy + t * cs, Ch * cs - 2, cs - 2);
  for (let c = 0; c < Ch; c++) if (aC) svgBand(sv, cp, aC, ox + c * cs, oy, cs - 2, T * cs - 2);

  const lab = (axis, txt, x, y, anchor) => axis
    ? cp.label(svgText(sv, x, y, txt, axisCol(axis), { anchor, weight: 700 }), axis, true)
    : svgText(sv, x, y, txt, C.dim, { anchor, weight: 600 });
  if (dims.length === 3) {
    lab(aB, `axis 0: ${names[0]} = ${B}`, ox + (B - 1) * dx / 2 + 40, oy + (B - 1) * dy - 22, "start");
    lab(aT, `axis 1: ${names[1]} = ${T}`, ox - 14, oy + T * cs / 2 + 4, "end");
    lab(aC, `axis 2: ${names[2]} = ${Ch}  ← contiguous`, ox + Ch * cs / 2, oy + T * cs + 22, "middle");
    if (aT) cp.guide(sv.appendChild(svgEl("path", { d: `M${ox - 6} ${oy} V${oy + T * cs - 2}`, stroke: axisCol(aT), "stroke-width": 1.6 })), aT);
    if (aC) cp.guide(sv.appendChild(svgEl("path", { d: `M${ox} ${oy + T * cs + 4} H${ox + Ch * cs - 2}`, stroke: axisCol(aC), "stroke-width": 1.6 })), aC);
  } else {
    lab(aT, `${names[0] || "rows"} = ${T}`, ox - 14, oy + T * cs / 2 + 4, "end");
    lab(aC, `${names[1] || "cols"} = ${Ch}`, ox + Ch * cs / 2, oy + T * cs + 22, "middle");
  }
  const n = dims.reduce((a, b) => a * b, 1);
  svgText(sv, statsX, oy + 4, `numel = ${n}`, C.dim, { weight: 600 });
  svgText(sv, statsX, oy + 22, `rank  = ${dims.length}`, C.dim, { weight: 600 });
  svgText(sv, statsX, oy + 40, `${(n * 4 / 1024).toFixed(2)} KB @ fp32`, C.faint, { size: 10, weight: 400 });
  wrap.appendChild(sv);
  note.innerHTML = `shape ${shapeHTML(names.map((nm, i) => [`${nm}=${dims[i]}`, resolveAxis(nm)]))}`;
  cp.apply();
}

/* ============================================================
   6. HEADS — the reshape dance, coloured by AXIS (coupled)
   The cells carry their flat index, so "nothing moved, it was only
   re-labelled" is visible without any colour. The axes are the brackets.
   ============================================================ */
function vizHeads(el, d) {
  const T = +d.t || 4, dmodel = +d.dmodel || 12;
  panel(el, "The reshape dance: (B, T, d_model) → (B, n_head, T, d_head)",
    "Nothing is computed. Each cell shows its position in memory; watch the same numbers get re-labelled. Hover, tap or tab to an axis.");
  const cp = coupler(el);
  const row = ctlRow(el);
  let nh = +d.nh || 3;
  const wrap = document.createElement("div"); wrap.style.overflowX = "auto"; el.appendChild(wrap);
  const code = document.createElement("div");
  code.style.cssText = "margin-top:12px;font-family:var(--mono);font-size:12.5px;color:var(--ink-2);line-height:1.9";
  el.appendChild(code);
  const opts = [1, 2, 3, 4, 6, 12].filter(x => dmodel % x === 0);
  const sel = document.createElement("label");
  sel.innerHTML = "n_head " + `<select>${opts.map(o => `<option ${o === nh ? "selected" : ""}>${o}</option>`).join("")}</select>`;
  row.appendChild(sel);
  sel.querySelector("select").onchange = e => { nh = +e.target.value; draw(); };

  function draw() {
    cp.reset();
    const dh = dmodel / nh, cs = 24, gap = 12, gap3 = 30;
    const x0 = 52, y0 = 58;
    const x2 = x0 + dmodel * cs + 92;
    const w2 = nh * (dh * cs + gap) - gap;
    const y3 = y0 + T * cs + 74;
    const w3 = nh * (dh * cs + gap3) - gap3;
    const W = Math.max(x2 + w2, x0 + w3) + 24, Hh = y3 + 44 + T * cs + 40;
    wrap.innerHTML = "";
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${Hh}`, width: W, height: Hh });

    const cell = (x, y, v) => {
      sv.appendChild(svgEl("rect", { x, y, width: cs - 2, height: cs - 2, rx: 3, fill: C.cell, stroke: C.grid }));
      svgText(sv, x + (cs - 2) / 2, y + (cs - 2) / 2 + 3.5, String(v), C.dim, { anchor: "middle", size: 9.5, weight: 500 });
    };

    // ---- step 1: (B, T, d_model): rows are time, columns are features
    svgShape(sv, cp, x0 - 40, 16, "1. after W_q   ", [["B", "batch"], ["T", "time"], ["d_model", "feat"]], ` = (1, ${T}, ${dmodel})`);
    for (let t = 0; t < T; t++) for (let c = 0; c < dmodel; c++) cell(x0 + c * cs, y0 + t * cs, t * dmodel + c);
    for (let t = 0; t < T; t++) svgBand(sv, cp, "time", x0, y0 + t * cs, dmodel * cs - 2, cs - 2);
    for (let c = 0; c < dmodel; c++) svgBand(sv, cp, "feat", x0 + c * cs, y0, cs - 2, T * cs - 2);
    svgBracket(sv, cp, "time", x0 - 8, y0, 0, y0 + T * cs - 2, "left", `T=${T}`, true);
    svgBracket(sv, cp, "feat", x0, y0 + T * cs + 4, x0 + dmodel * cs - 2, 0, "bottom", `d_model = ${dmodel}`, true);
    svgText(sv, x0 + dmodel * cs + 30, y0 + T * cs / 2 + 7, "→", C.dim, { size: 22, anchor: "middle" });
    svgText(sv, x0 + dmodel * cs + 30, y0 + T * cs / 2 + 26, ".view", C.faint, { size: 10, anchor: "middle", weight: 400 });

    // ---- step 2: (B, T, nh, hd): the feature axis is SPLIT into head x head-dim
    svgShape(sv, cp, x2, 16, "2. ", [["B", "batch"], ["T", "time"], ["nh", "head"], ["hd", "headdim"]], ` = (1, ${T}, ${nh}, ${dh})`);
    for (let h = 0; h < nh; h++) {
      const gx = x2 + h * (dh * cs + gap);
      for (let t = 0; t < T; t++) for (let i = 0; i < dh; i++) cell(gx + i * cs, y0 + t * cs, t * dmodel + h * dh + i);
      for (let i = 0; i < dh; i++) svgBand(sv, cp, "headdim", gx + i * cs, y0, cs - 2, T * cs - 2);
      svgBand(sv, cp, "head", gx, y0, dh * cs - 2, T * cs - 2);
      svgText(sv, gx + (dh * cs - 2) / 2, y0 + T * cs + 30, `h${h}`, C.faint, { anchor: "middle", size: 10, weight: 500 });
    }
    for (let t = 0; t < T; t++) svgBand(sv, cp, "time", x2, y0 + t * cs, w2, cs - 2);
    svgBracket(sv, cp, "time", x2 - 8, y0, 0, y0 + T * cs - 2, "left", `T`, false);
    svgBracket(sv, cp, "head", x2, y0 - 12, x2 + w2, 0, "top", `nh = ${nh}`, true);
    svgBracket(sv, cp, "headdim", x2, y0 + T * cs + 4, x2 + dh * cs - 2, 0, "bottom", `hd = ${dh}`, true);

    // ---- step 3: .transpose(1, 2) -> (B, nh, T, hd): heads become the outer axis
    svgShape(sv, cp, x0 - 40, y3 - 6, "3. .transpose(1, 2)   ", [["B", "batch"], ["nh", "head"], ["T", "time"], ["hd", "headdim"]],
      `   ← T moved from position 1 to position 2`);
    const yb = y3 + 30;
    for (let h = 0; h < nh; h++) {
      const bx = x0 + h * (dh * cs + gap3);
      for (let t = 0; t < T; t++) for (let i = 0; i < dh; i++) cell(bx + i * cs, yb + t * cs, t * dmodel + h * dh + i);
      for (let t = 0; t < T; t++) svgBand(sv, cp, "time", bx, yb + t * cs, dh * cs - 2, cs - 2);
      for (let i = 0; i < dh; i++) svgBand(sv, cp, "headdim", bx + i * cs, yb, cs - 2, T * cs - 2);
      svgBand(sv, cp, "head", bx, yb, dh * cs - 2, T * cs - 2);
      sv.appendChild(svgEl("rect", { x: bx - 5, y: yb - 5, width: dh * cs + 6, height: T * cs + 8, rx: 6,
        fill: "none", stroke: C.gridHi, "stroke-width": 1.2, "stroke-dasharray": "4 3" }));
      svgText(sv, bx + (dh * cs - 2) / 2, yb + T * cs + 22, `head ${h}: (T, hd)`, C.faint, { anchor: "middle", size: 10, weight: 500 });
    }
    svgBracket(sv, cp, "head", x0, yb - 14, x0 + w3, 0, "top", `nh = ${nh} independent (T, hd) matrices`, false);
    wrap.appendChild(sv);

    code.innerHTML =
      `<span style="color:${C.faint}"># B=1, T=${T}, d_model=${dmodel}, n_head=${nh}, d_head=${dmodel}/${nh}=${dh}</span><br>` +
      `q = q.view${shapeHTML([["B", "batch"], ["T", "time"], ["nh", "head"], ["hd", "headdim"]])}.transpose(1, 2)` +
      `&nbsp;&nbsp;<span style="color:${C.faint}">#</span> ${shapeHTML([["B", "batch"], ["nh", "head"], ["T", "time"], ["hd", "headdim"]])}<br>` +
      `<b style="color:${C.accent}">Key insight:</b> d_model is <i>partitioned</i>, not duplicated: the cell numbers ` +
      `0…${T * dmodel - 1} each appear exactly once in every step. Total parameters and FLOPs are the same as one big ` +
      `head — you just get ${nh} independent attention patterns for free.`;
    cp.apply();
  }
  draw();
}

/* ============================================================
   11. GQA — query heads grouped onto shared K/V heads (coupled)
   Group identity is shown by alternating fill and by the connecting
   lines, NOT by axis colours; the only axis here is "head".
   ============================================================ */
function vizGQA(el, d) {
  panel(el, "MHA → GQA → MQA is one slider", "Query heads are grouped; each group shares one K/V head.");
  const cp = coupler(el);
  const row = ctlRow(el);
  const nh = +d.nh || 8;
  let g = +d.groups || 4;
  const wrap = document.createElement("div"); el.appendChild(wrap);
  const note = document.createElement("div");
  note.style.cssText = "margin-top:12px;font-family:var(--mono);font-size:12px;color:var(--ink-2);line-height:1.85";
  el.appendChild(note);
  const divisors = [];
  for (let i = 1; i <= nh; i++) if (nh % i === 0) divisors.push(i);
  slider(row, "n_kv_heads", 0, divisors.length - 1, divisors.indexOf(g), i => { g = divisors[i]; draw(); });

  function draw() {
    cp.reset();
    const per = nh / g;
    const cw = 60, W = nh * cw + 40, Hh = 200;
    wrap.innerHTML = "";
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${Hh}`, width: "100%", height: Hh, preserveAspectRatio: "xMidYMid meet" });
    const shade = j => (j % 2 ? C.cell : C.paper);
    svgBracket(sv, cp, "head", 20, 30, 20 + nh * cw - 8, 0, "top", `query heads · n_head = ${nh}`, true);
    for (let h = 0; h < nh; h++) {
      const grp = Math.floor(h / per);
      sv.appendChild(svgEl("rect", { x: 20 + h * cw, y: 36, width: cw - 8, height: 34, rx: 6, fill: shade(grp), stroke: C.gridHi, "stroke-width": 1.4 }));
      svgBand(sv, cp, "head", 20 + h * cw, 36, cw - 8, 34);
      svgText(sv, 20 + h * cw + (cw - 8) / 2, 57, "Q" + h, C.fg, { anchor: "middle", size: 12 });
    }
    for (let j = 0; j < g; j++) {
      const x = 20 + j * per * cw, w = per * cw - 8;
      for (let h = j * per; h < (j + 1) * per; h++)
        sv.appendChild(svgEl("line", { x1: 20 + h * cw + (cw - 8) / 2, y1: 70, x2: x + w / 2, y2: 120, stroke: C.gridHi, "stroke-width": 1.2, opacity: .7 }));
      sv.appendChild(svgEl("rect", { x, y: 120, width: w, height: 34, rx: 6, fill: shade(j), stroke: C.gridHi, "stroke-width": 1.4 }));
      svgBand(sv, cp, "head", x, 120, w, 34);
      svgText(sv, x + w / 2, 141, `K${j} / V${j}`, C.fg, { anchor: "middle", size: 12 });
    }
    svgBracket(sv, cp, "head", 20, 160, 20 + nh * cw - 8, 0, "bottom", `kv heads · n_kv_heads = ${g}`, true);
    wrap.appendChild(sv);
    const name = g === nh ? "MHA (Multi-Head Attention)" : g === 1 ? "MQA (Multi-Query Attention)" : "GQA (Grouped-Query Attention)";
    note.innerHTML =
      `<b style="color:${C.accent}">${name}</b> — ${nh} query heads, ${g} KV head${g > 1 ? "s" : ""}, ${per} quer${per > 1 ? "ies" : "y"} per group.<br>` +
      `K and V: ${shapeHTML([["B", "batch"], [`n_kv_heads=${g}`, "head"], ["T", "time"], ["hd", "headdim"]])} — ` +
      `the cache is <b style="color:${C.accent}">${(nh / g).toFixed(0)}× smaller</b> than MHA. Q projection is unchanged; only W_k and W_v shrink.<br>` +
      `<span style="color:${C.faint}">In code the K/V tensors are repeat_interleave'd back up to ${nh} heads before the matmul — ` +
      `or, in a fused kernel, simply re-read. The saving is memory and bandwidth, not FLOPs.</span>`;
    cp.apply();
  }
  draw();
}

/* ============================================================
   1. MATMUL — the machine
   ============================================================ */
function vizMatmul(el, d) {
  const m = +d.m || 4, k = +d.k || 3, n = +d.n || 5;
  const cs = 30, gap = 46;
  const A = [], B = [];
  const r = rng(7);
  for (let i = 0; i < m; i++) { A.push([]); for (let j = 0; j < k; j++) A[i].push(smallInt(r)); }
  for (let i = 0; i < k; i++) { B.push([]); for (let j = 0; j < n; j++) B[i].push(smallInt(r)); }
  const Cm = A.map(row => B[0].map((_, j) => row.reduce((s, a, t) => s + a * B[t][j], 0)));

  panel(el, "Matrix multiplication, one output cell at a time",
    "Drag the sliders to pick an output cell. Watch which row and which column feed it.");
  const row = ctlRow(el);
  let hi = 0, hj = 0;
  const wrap = document.createElement("div");
  wrap.style.overflowX = "auto";
  el.appendChild(wrap);
  const expl = document.createElement("div");
  expl.style.cssText = "margin-top:14px;font-family:var(--mono);font-size:12.5px;line-height:1.9;color:var(--ink-2)";
  el.appendChild(expl);

  slider(row, "row i", 0, m - 1, 0, v => { hi = v; draw(); });
  slider(row, "col j", 0, n - 1, 0, v => { hj = v; draw(); });

  function grid(sv, x0, y0, M, rows, cols, color, hiRow, hiCol) {
    const g = svgEl("g");
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const on = (hiRow === i) || (hiCol === j);
      const rect = svgEl("rect", {
        x: x0 + j * cs, y: y0 + i * cs, width: cs - 2, height: cs - 2, rx: 4,
        fill: on ? color + "33" : C.cell,
        stroke: on ? color : C.grid, "stroke-width": on ? 1.6 : 1,
      });
      g.appendChild(rect);
      const t = svgEl("text", {
        x: x0 + j * cs + (cs - 2) / 2, y: y0 + i * cs + (cs - 2) / 2 + 4,
        "text-anchor": "middle", "font-size": 11.5, "font-family": "var(--mono)",
        fill: on ? color : C.faint, "font-weight": on ? 700 : 400,
      });
      t.textContent = M[i][j];
      g.appendChild(t);
    }
    sv.appendChild(g);
  }
  function label(sv, x, y, txt, color, size = 11, anchor = "middle", weight = 600) {
    const t = svgEl("text", { x, y, "text-anchor": anchor, "font-size": size, fill: color, "font-family": "var(--mono)", "font-weight": weight });
    t.textContent = txt; sv.appendChild(t);
  }

  function draw() {
    wrap.innerHTML = "";
    const aw = k * cs, bw = n * cs, cw = n * cs;
    const W = aw + gap + bw + gap + 26 + cw + 40;
    const H = Math.max(m, k) * cs + 76;
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    const y0 = 34;

    // A
    grid(sv, 10, y0, A, m, k, C.stamp, hi, -1);
    label(sv, 10 + aw / 2, 22, `A  (${m}, ${k})`, C.stamp, 12);
    label(sv, 10 + aw / 2, y0 + m * cs + 18, `${k} columns`, C.faint, 10);

    // ×
    label(sv, 10 + aw + gap / 2, y0 + Math.max(m, k) * cs / 2, "×", C.dim, 20);

    // B
    const bx = 10 + aw + gap;
    grid(sv, bx, y0, B, k, n, C.dim, -1, hj);
    label(sv, bx + bw / 2, 22, `B  (${k}, ${n})`, C.dim, 12);
    label(sv, bx + bw / 2, y0 + k * cs + 18, `${k} rows`, C.faint, 10);

    // =
    const ex = bx + bw + gap / 2;
    label(sv, ex, y0 + Math.max(m, k) * cs / 2, "=", C.dim, 20);

    // C
    const cx = bx + bw + gap + 26;
    const g = svgEl("g");
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
      const on = i === hi && j === hj;
      g.appendChild(svgEl("rect", {
        x: cx + j * cs, y: y0 + i * cs, width: cs - 2, height: cs - 2, rx: 4,
        fill: on ? C.accent + "44" : C.cell, stroke: on ? C.accent : C.grid, "stroke-width": on ? 2 : 1,
      }));
      const t = svgEl("text", {
        x: cx + j * cs + (cs - 2) / 2, y: y0 + i * cs + (cs - 2) / 2 + 4, "text-anchor": "middle",
        "font-size": 11, "font-family": "var(--mono)", fill: on ? C.accent : C.faint, "font-weight": on ? 800 : 400,
      });
      t.textContent = Cm[i][j]; g.appendChild(t);
    }
    sv.appendChild(g);
    label(sv, cx + cw / 2, 22, `C  (${m}, ${n})`, C.accent, 12);
    wrap.appendChild(sv);

    const terms = A[hi].map((a, t) => `${a}×${B[t][hj]}`).join(" + ");
    const prods = A[hi].map((a, t) => a * B[t][hj]).join(" + ");
    expl.innerHTML =
      `<span style="color:${C.accent}">C[${hi},${hj}]</span> = dot( <span style="color:${C.stamp}">A[${hi},:]</span> , <span style="color:${C.dim}">B[:,${hj}]</span> )<br>` +
      `&nbsp;&nbsp;= ${terms}<br>&nbsp;&nbsp;= ${prods} = <b style="color:${C.accent}">${Cm[hi][hj]}</b><br>` +
      `<span style="color:${C.faint}">The shared dimension k=${k} is summed away. It does not appear in the output shape.</span>`;
  }
  draw();
}

/* ============================================================
   3. BROADCAST — right-aligned shape rules
   ============================================================ */
function vizBroadcast(el, d) {
  panel(el, "Broadcasting: align from the RIGHT",
    "Shapes are compared right-to-left. A dimension is compatible if it matches, or if one of them is 1.");
  const cases = JSON.parse(d.cases || '[[[3,1,4],[5,4]],[[8,1,6,1],[7,1,5]],[[2,3],[4,3]]]');
  const box = document.createElement("div");
  box.style.cssText = "font-family:var(--mono);font-size:13px;line-height:2.05";
  cases.forEach(([a, b]) => {
    const L = Math.max(a.length, b.length);
    const pa = Array(L - a.length).fill(null).concat(a);
    const pb = Array(L - b.length).fill(null).concat(b);
    let okAll = true;
    const outs = [];
    for (let i = 0; i < L; i++) {
      const x = pa[i], y = pb[i];
      if (x === null) { outs.push({ v: y, ok: true }); continue; }
      if (y === null) { outs.push({ v: x, ok: true }); continue; }
      if (x === y || x === 1 || y === 1) outs.push({ v: Math.max(x, y), ok: true });
      else { outs.push({ v: "✗", ok: false }); okAll = false; }
    }
    const cell = (v, col) => `<span style="display:inline-block;min-width:34px;text-align:center;border:1px solid ${col === C.faint ? C.grid : col};padding:0 5px;margin:0 2px;color:${col}">${v === null ? "·" : v}</span>`;
    const rowHtml = (arr, col) => arr.map(v => cell(v, v === null ? C.faint : col)).join("");
    const div = document.createElement("div");
    div.style.cssText = "margin-bottom:16px;padding:11px 13px;background:var(--surface);border:1px solid " + (okAll ? C.grid : C.danger) + ";";
    div.innerHTML =
      `<div><span style="display:inline-block;width:52px;color:${C.stamp}">A</span>${rowHtml(pa, C.stamp)}  <span style="color:${C.faint}">← (${a.join(", ")})</span></div>` +
      `<div><span style="display:inline-block;width:52px;color:${C.dim}">B</span>${rowHtml(pb, C.dim)}  <span style="color:${C.faint}">← (${b.join(", ")})</span></div>` +
      `<div style="border-top:1px solid ${C.grid};margin:6px 0 4px"></div>` +
      `<div><span style="display:inline-block;width:52px;color:${okAll ? C.accent : C.danger}">${okAll ? "out" : "ERROR"}</span>` +
      outs.map(o => cell(o.v, o.ok ? C.accent : C.danger)).join("") +
      (okAll ? `  <span style="color:${C.faint}">← (${outs.map(o => o.v).join(", ")})</span>`
             : `  <span style="color:${C.danger}">incompatible: neither is 1 and they differ</span>`) + `</div>`;
    box.appendChild(div);
  });
  el.appendChild(box);
}

/* ============================================================
   4. SOFTMAX — logits to probabilities, with temperature
   ============================================================ */
function vizSoftmax(el, d) {
  const logits = (d.logits || "3.2,1.1,0.4,-0.8,2.5,-2.1,0.9,1.8").split(",").map(Number);
  const labels = (d.labels || "").split(",").filter(Boolean);
  panel(el, "Softmax, temperature, and truncation",
    "Softmax turns any real numbers into a probability distribution. Temperature reshapes it before sampling.");
  const row = ctlRow(el);
  let T = 1.0, topk = 0, topp = 1.0;
  const wrap = document.createElement("div"); el.appendChild(wrap);
  const info = document.createElement("div");
  info.style.cssText = "margin-top:10px;font-family:var(--mono);font-size:12px;color:var(--ink-2)";
  el.appendChild(info);

  slider(row, "temperature ×0.1", 1, 30, 10, v => { T = v / 10; draw(); });
  slider(row, "top-k (0=off)", 0, logits.length, 0, v => { topk = v; draw(); });
  slider(row, "top-p ×0.05", 1, 20, 20, v => { topp = v * 0.05; draw(); });

  function draw() {
    const n = logits.length;
    const scaled = logits.map(x => x / Math.max(T, 0.01));
    const mx = Math.max(...scaled);
    const ex = scaled.map(x => Math.exp(x - mx));
    const Z = ex.reduce((a, b) => a + b, 0);
    let p = ex.map(x => x / Z);

    // top-k / top-p truncation
    const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    const keep = new Set(order.map(o => o[1]));
    if (topk > 0) { keep.clear(); order.slice(0, topk).forEach(o => keep.add(o[1])); }
    if (topp < 0.999) {
      const kp = new Set(); let c = 0;
      for (const [v, i] of order) { if (!keep.has(i)) continue; kp.add(i); c += v; if (c >= topp) break; }
      keep.forEach(i => { if (!kp.has(i)) keep.delete(i); });
    }
    const masked = p.map((v, i) => keep.has(i) ? v : 0);
    const Z2 = masked.reduce((a, b) => a + b, 0) || 1;
    const pf = masked.map(v => v / Z2);

    const bw = 52, H = 190;
    const W = n * bw + 60;
    wrap.innerHTML = "";
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${H + 46}`, width: "100%", height: H + 46, preserveAspectRatio: "xMidYMid meet" });
    const maxp = Math.max(...pf, 0.01);
    for (let i = 0; i < n; i++) {
      const h = (pf[i] / maxp) * H;
      const x = 40 + i * bw;
      const alive = keep.has(i);
      sv.appendChild(svgEl("rect", {
        x, y: H - h + 8, width: bw - 12, height: Math.max(h, 1), rx: 3,
        fill: alive ? C.accent : C.gridHi, opacity: alive ? .9 : .45,
      }));
      const t1 = svgEl("text", { x: x + (bw - 12) / 2, y: H - h + 2, "text-anchor": "middle", "font-size": 10, fill: alive ? C.accent : C.faint, "font-family": "var(--mono)" });
      t1.textContent = pf[i] < 0.001 ? "0" : pf[i].toFixed(3); sv.appendChild(t1);
      const t2 = svgEl("text", { x: x + (bw - 12) / 2, y: H + 24, "text-anchor": "middle", "font-size": 10, fill: C.faint, "font-family": "var(--mono)" });
      t2.textContent = logits[i].toFixed(1); sv.appendChild(t2);
      if (labels[i]) {
        const t3 = svgEl("text", { x: x + (bw - 12) / 2, y: H + 39, "text-anchor": "middle", "font-size": 10, fill: C.dim });
        t3.textContent = labels[i]; sv.appendChild(t3);
      }
    }
    const yl = svgEl("text", { x: 4, y: 16, "font-size": 10, fill: C.faint, "font-family": "var(--mono)" });
    yl.textContent = "p"; sv.appendChild(yl);
    const xl = svgEl("text", { x: 4, y: H + 24, "font-size": 10, fill: C.faint, "font-family": "var(--mono)" });
    xl.textContent = "logit"; sv.appendChild(xl);
    wrap.appendChild(sv);

    const H_ent = -pf.filter(v => v > 0).reduce((a, v) => a + v * Math.log2(v), 0);
    info.innerHTML =
      `T = ${T.toFixed(2)} &nbsp;·&nbsp; kept ${keep.size}/${n} tokens &nbsp;·&nbsp; Σp = ${pf.reduce((a, b) => a + b, 0).toFixed(4)} &nbsp;·&nbsp; ` +
      `entropy = ${H_ent.toFixed(2)} bits &nbsp;·&nbsp; perplexity = ${Math.pow(2, H_ent).toFixed(2)}<br>` +
      `<span style="color:${C.faint}">T→0 gives greedy (one spike). T→∞ gives uniform. Truncation renormalises what survives.</span>`;
  }
  draw();
}

/* ============================================================
   5. ATTENTION — scores → mask → softmax → weighted values
   ============================================================ */
function vizAttention(el, d) {
  const T = +d.t || 7;
  const toks = (d.tokens || "The cat sat on the warm mat").split(" ").slice(0, T);
  panel(el, "Attention matrix: from scores to weights",
    "Row i = 'what token i attends to'. Every row of the final matrix sums to 1.");
  const cp = coupler(el);
  const row = ctlRow(el);
  let causal = d.causal !== "false", scale = true, dk = +d.dk || 64;

  const r = rng(11);
  const raw = [];
  for (let i = 0; i < T; i++) { raw.push([]); for (let j = 0; j < T; j++) raw[i].push((r() * 2 - 1) * 9); }
  for (let i = 0; i < T; i++) raw[i][i] += 3;                       // mild self-affinity
  for (let i = 1; i < T; i++) raw[i][i - 1] += 2;                   // mild locality

  const wrap = document.createElement("div"); wrap.style.overflowX = "auto"; el.appendChild(wrap);
  const note = document.createElement("div");
  note.style.cssText = "margin-top:12px;font-size:12px;font-family:var(--mono);color:var(--ink-2);line-height:1.8";
  el.appendChild(note);

  toggle(row, "causal mask", causal, v => { causal = v; draw(); });
  toggle(row, "divide by √d_k", scale, v => { scale = v; draw(); });
  slider(row, "d_k", 4, 256, dk, v => { dk = v; draw(); });

  function draw() {
    cp.reset();
    const cs = 40, pad = 96;
    const W = pad + T * cs + 50, Hh = pad + T * cs + 20;   // room for the row-sum column
    wrap.innerHTML = "";
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${Hh}`, width: W, height: Hh });

    const s = raw.map(rw => rw.map(x => scale ? x / Math.sqrt(dk) * 8 : x));
    const P = [];
    for (let i = 0; i < T; i++) {
      const rowv = s[i].map((x, j) => (causal && j > i) ? -Infinity : x);
      const mx = Math.max(...rowv.filter(v => v > -Infinity));
      const ex = rowv.map(x => x === -Infinity ? 0 : Math.exp(x - mx));
      const Z = ex.reduce((a, b) => a + b, 0);
      P.push(ex.map(x => x / Z));
    }
    // column headers (keys)
    for (let j = 0; j < T; j++) {
      const t = svgEl("text", {
        x: pad + j * cs + cs / 2, y: pad - 10, "text-anchor": "start", "font-size": 11,
        fill: axisCol("time"), "font-family": "var(--mono)",
        transform: `rotate(-52 ${pad + j * cs + cs / 2} ${pad - 10})`,
      });
      t.textContent = toks[j] || "t" + j; sv.appendChild(t); cp.label(t, "time", false);
    }
    const kl = svgEl("text", { x: pad + T * cs / 2, y: 16, "text-anchor": "middle", "font-size": 11, fill: axisCol("time"), "font-family": "var(--mono)", "font-weight": 700 });
    kl.textContent = "KEYS · time  (what is available to look at) →"; sv.appendChild(kl); cp.label(kl, "time", true);

    for (let i = 0; i < T; i++) {
      const t = svgEl("text", { x: pad - 10, y: pad + i * cs + cs / 2 + 4, "text-anchor": "end", "font-size": 11, fill: axisCol("time"), "font-family": "var(--mono)" });
      t.textContent = toks[i] || "t" + i; sv.appendChild(t); cp.label(t, "time", false);
      for (let j = 0; j < T; j++) {
        const p = P[i][j];
        const masked = causal && j > i;
        sv.appendChild(svgEl("rect", {
          x: pad + j * cs, y: pad + i * cs, width: cs - 2, height: cs - 2, rx: 3,
          fill: masked ? C.cell : C.accent,
          "fill-opacity": masked ? 1 : (0.06 + p * 0.9).toFixed(3),
          stroke: masked ? C.grid : C.gridHi, "stroke-width": 1,
          "stroke-dasharray": masked ? "2 2" : "",
        }));
        const t2 = svgEl("text", {
          x: pad + j * cs + (cs - 2) / 2, y: pad + i * cs + (cs - 2) / 2 + 4, "text-anchor": "middle",
          "font-size": 9.5, "font-family": "var(--mono)",
          fill: masked ? C.fainter : (p > 0.45 ? C.paper : C.dim), "font-weight": p > 0.45 ? 700 : 400,
        });
        t2.textContent = masked ? "−∞" : p.toFixed(2); sv.appendChild(t2);
      }
      const rs = svgEl("text", { x: pad + T * cs + 6, y: pad + i * cs + cs / 2 + 4, "font-size": 9.5, fill: C.faint, "font-family": "var(--mono)" });
      rs.textContent = "Σ" + P[i].reduce((a, b) => a + b, 0).toFixed(2); sv.appendChild(rs);
    }
    const ql = svgEl("text", { x: 14, y: pad + T * cs / 2, "text-anchor": "middle", "font-size": 11, fill: axisCol("time"), "font-family": "var(--mono)", "font-weight": 700, transform: `rotate(-90 14 ${pad + T * cs / 2})` });
    ql.textContent = "← QUERIES · time (who is asking)"; sv.appendChild(ql); cp.label(ql, "time", true);
    wrap.appendChild(sv);

    const maxOff = Math.max(...P.flat());
    note.innerHTML =
      `shape of this matrix: ${shapeHTML([[`T=${T}`, "time"], [`T=${T}`, "time"]])} — one row per query position, one column per key position: <b>both axes are time</b>.<br>` +
      (causal ? `Causal mask ON: position i cannot see j &gt; i. Those cells got −∞ <i>before</i> softmax, so they become exactly 0.<br>` :
                `<span style="color:${C.danger}">Causal mask OFF</span>: every token sees the future. Fine for BERT, catastrophic for a generator.<br>`) +
      (scale ? `Scaled by 1/√d_k with d_k=${dk} → scores stay in a range where softmax has usable gradients. Peak weight ${maxOff.toFixed(2)}.` :
               `<span style="color:${C.danger}">Unscaled</span>: raw dot products grow like d_k, softmax saturates, gradients vanish. Peak weight ${maxOff.toFixed(2)}.`);
    cp.apply();
  }
  draw();
}

/* ============================================================
   7. ROPE — rotary position embedding on a 2D pair
   ============================================================ */
function vizRope(el, d) {
  panel(el, "RoPE: position as rotation",
    "Every adjacent pair of features is treated as a point in a 2D plane and rotated by an angle proportional to position.");
  const row = ctlRow(el);
  let m = 3, pairIdx = 0, dhead = 64, base = 10000;
  const wrap = document.createElement("div"); el.appendChild(wrap);
  const note = document.createElement("div");
  note.style.cssText = "margin-top:12px;font-family:var(--mono);font-size:12px;color:var(--ink-2);line-height:1.85";
  el.appendChild(note);
  slider(row, "position m", 0, 32, 3, v => { m = v; draw(); });
  slider(row, "pair index i", 0, 15, 0, v => { pairIdx = v; draw(); });

  function draw() {
    const theta = Math.pow(base, -2 * pairIdx / dhead);
    const ang = m * theta;
    const R = 78, cx = 110, cy = 110;
    wrap.innerHTML = "";
    const sv = svgEl("svg", { viewBox: "0 0 520 230", width: "100%", height: 230, preserveAspectRatio: "xMidYMid meet" });
    sv.appendChild(svgEl("circle", { cx, cy, r: R, fill: "none", stroke: C.grid, "stroke-dasharray": "3 4" }));
    sv.appendChild(svgEl("line", { x1: cx - R - 14, y1: cy, x2: cx + R + 14, y2: cy, stroke: C.grid }));
    sv.appendChild(svgEl("line", { x1: cx, y1: cy - R - 14, x2: cx, y2: cy + R + 14, stroke: C.grid }));

    const arrow = (a, col, lbl, dash) => {
      const x = cx + R * Math.cos(a), y = cy - R * Math.sin(a);
      sv.appendChild(svgEl("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: col, "stroke-width": 2.3, "stroke-dasharray": dash || "" }));
      sv.appendChild(svgEl("circle", { cx: x, cy: y, r: 4, fill: col }));
      const t = svgEl("text", { x: x + (Math.cos(a) > 0 ? 9 : -9), y: y + (Math.sin(a) > 0 ? -7 : 14), "font-size": 11, fill: col, "font-family": "var(--mono)", "text-anchor": Math.cos(a) > 0 ? "start" : "end" });
      t.textContent = lbl; sv.appendChild(t);
    };
    arrow(0, C.faint, "original (m=0)", "4 3");
    arrow(ang, C.accent, `rotated by mθ`);

    // arc
    const steps = 26, pts = [];
    for (let s = 0; s <= steps; s++) { const a = ang * s / steps; pts.push(`${cx + 34 * Math.cos(a)},${cy - 34 * Math.sin(a)}`); }
    sv.appendChild(svgEl("polyline", { points: pts.join(" "), fill: "none", stroke: C.stamp, "stroke-width": 1.6 }));
    const at = svgEl("text", { x: cx + 42 * Math.cos(ang / 2), y: cy - 42 * Math.sin(ang / 2) + 4, "font-size": 11, fill: C.stamp, "font-family": "var(--mono)" });
    at.textContent = `mθ`; sv.appendChild(at);

    // right panel: relative-position property
    const tx = 245;
    const rows = [
      [`θ_i = base^(−2i/d_head)`, C.dim],
      [`    = ${base}^(−${2 * pairIdx}/${dhead}) = ${theta.toExponential(3)}`, C.faint],
      [`m·θ_i = ${m} × ${theta.toExponential(3)} = ${ang.toFixed(4)} rad`, C.accent],
      [``, C.dim],
      [`Rotate q by mθ, k by nθ. Then:`, C.dim],
      [`⟨R(m)q, R(n)k⟩ = ⟨q, R(n−m)k⟩`, C.stamp],
      [``, C.dim],
      [`The dot product depends ONLY on (n−m).`, C.accent],
      [`That is relative position, for free,`, C.accent],
      [`with zero extra parameters.`, C.accent],
      [``, C.dim],
      [`Low i → fast rotation → local detail`, C.faint],
      [`High i → slow rotation → long range`, C.faint],
    ];
    rows.forEach(([txt, col], idx) => {
      const t = svgEl("text", { x: tx, y: 26 + idx * 15.5, "font-size": 11.5, fill: col, "font-family": "var(--mono)" });
      t.textContent = txt; sv.appendChild(t);
    });
    wrap.appendChild(sv);

    const wavelength = 2 * Math.PI / theta;
    note.innerHTML = `Pair i=${pairIdx} of d_head=${dhead} completes one full turn every <b style="color:${C.stamp}">${wavelength.toFixed(1)}</b> tokens. ` +
      `Pair 0 turns every 6.3 tokens; pair ${dhead / 2 - 1} turns every ~${(2 * Math.PI / Math.pow(base, -2 * (dhead / 2 - 1) / dhead)).toExponential(1)} tokens. ` +
      `That spread is what lets one mechanism encode both "next word" and "10k tokens ago".`;
  }
  draw();
}

/* ============================================================
   8. KV CACHE — memory growth calculator
   ============================================================ */
function vizKV(el, d) {
  panel(el, "KV cache size calculator",
    "This is the number that decides how many users fit on one GPU. Play with it until the scaling is intuitive.");
  const row = ctlRow(el), row2 = ctlRow(el);
  let L = 32, nh = 32, dh = 128, kvh = 32, seq = 4096, bs = 1, bytes = 2, mode = "gqa";
  const out = document.createElement("div");
  out.style.cssText = "font-family:var(--mono);font-size:13px;line-height:2;color:var(--ink)";
  el.appendChild(out);

  slider(row, "layers", 4, 128, L, v => { L = v; draw(); });
  slider(row, "query heads", 4, 128, nh, v => { nh = v; kvSel.max = v; draw(); });
  slider(row, "d_head", 32, 256, dh, v => { dh = v; draw(); });
  const kvSel = slider(row2, "KV heads", 1, 128, kvh, v => { kvh = v; draw(); });
  slider(row2, "seq len", 512, 131072, seq, v => { seq = v; draw(); });
  slider(row2, "batch", 1, 64, bs, v => { bs = v; draw(); });

  function fmt(b) {
    if (b > 1024 ** 3) return (b / 1024 ** 3).toFixed(2) + " GB";
    if (b > 1024 ** 2) return (b / 1024 ** 2).toFixed(1) + " MB";
    return (b / 1024).toFixed(0) + " KB";
  }
  function draw() {
    const mha = 2 * L * nh * dh * seq * bs * bytes;
    const gqa = 2 * L * kvh * dh * seq * bs * bytes;
    const mqa = 2 * L * 1 * dh * seq * bs * bytes;
    const mla = L * (512 + 64) * seq * bs * bytes;   // DeepSeek-V3 latent (512) + decoupled RoPE key (64)
    const bar = (label, val, ref, col) => {
      const pct = Math.min(100, val / ref * 100);
      return `<div style="display:flex;align-items:center;gap:10px;margin:3px 0">
        <span style="width:150px;color:${col}">${label}</span>
        <span style="flex:1;max-width:280px;height:13px;background:var(--line);overflow:hidden">
          <span style="display:block;height:100%;width:${pct}%;background:${col}"></span></span>
        <span style="width:92px;text-align:right;color:${col};font-weight:700">${fmt(val)}</span>
        <span style="color:var(--ink-3);font-size:11px">${(mha / val).toFixed(1)}× smaller</span></div>`;
    };
    out.innerHTML =
      `<div style="color:var(--ink-2);font-size:12px;margin-bottom:10px">` +
      `bytes = 2 (K and V) × n_layers × n_kv_heads × d_head × seq_len × batch × dtype_bytes &nbsp;<span style="color:var(--ink-3)">(fp16 = 2 bytes)</span></div>` +
      bar(`MHA (${nh} KV heads)`, mha, mha, C.accent) +
      bar(`GQA (${kvh} KV heads)`, gqa, mha, C.accent) +
      bar(`MQA (1 KV head)`, mqa, mha, C.accent) +
      bar(`MLA (576/token/layer)`, mla, mha, C.accent) +
      `<div style="margin-top:12px;color:var(--ink-3);font-size:11.5px;line-height:1.7">` +
      `For reference: one H100 has 80 GB total, and the <i>weights</i> of a 70B fp16 model already take ~140 GB (so it needs ≥2 GPUs). ` +
      `Everything left over is cache — and cache is what limits how many concurrent users you can serve.</div>`;
  }
  draw();
}

/* ============================================================
   9. SHAPE TRACER — step through a forward pass
   ============================================================ */
function vizTrace(el, d) {
  const steps = JSON.parse(el.querySelector("script")?.textContent || d.steps || "[]");
  el.querySelector("script")?.remove();
  panel(el, d.title || "Forward pass, one shape at a time", "Step through. The shape line is the thing to memorise.");
  let i = 0;
  let amap = {};
  try { amap = JSON.parse(d.axismap || "{}"); } catch (e) { amap = {}; }
  /* "(2, 8, 6, 64)" -> badges, by value via data-axismap; "(B=2, T=6)" by symbol */
  const shapeLine = txt => txt.replace(/\(([^()]*)\)/g, (m, inner) => {
    if (!inner.trim()) return m;
    const parts = inner.split(/\s*,\s*/).map(tok => {
      const sym = tok.split("=")[0].trim();
      const a = amap[tok.trim()] || resolveAxis(sym);
      return [tok.trim(), a || null];
    });
    return parts.some(p => p[1]) ? shapeHTML(parts) : m;
  });
  const body = document.createElement("div"); el.appendChild(body);
  const row = ctlRow(el); row.style.marginTop = "14px"; row.style.marginBottom = "0";
  const prev = document.createElement("button"); prev.className = "btn"; prev.textContent = "← back";
  const next = document.createElement("button"); next.className = "btn primary"; next.textContent = "next step →";
  const cnt = document.createElement("span"); cnt.style.cssText = "font-family:var(--mono);font-size:12px;color:var(--ink-3)";
  row.append(prev, next, cnt);
  prev.onclick = () => { i = Math.max(0, i - 1); render(); };
  next.onclick = () => { i = Math.min(steps.length - 1, i + 1); render(); };

  function render() {
    const s = steps[i];
    body.innerHTML =
      `<div style="display:flex;gap:9px;margin-bottom:12px;flex-wrap:wrap">` +
      steps.map((_, j) => `<span style="width:26px;height:5px;background:${j <= i ? C.accent : "var(--line)"}"></span>`).join("") +
      `</div>` +
      `<div style="font-weight:700;font-size:1.02rem;margin-bottom:7px">${s.name}</div>` +
      `<div style="font-family:var(--mono);font-size:13.5px;background:var(--surface);border:1px solid var(--line);border-left:3px solid ${C.stamp};padding:11px 14px;margin-bottom:11px">` +
      `<span style="color:var(--ink-3)">shape</span> &nbsp;<span style="font-weight:700;color:var(--ink)">${shapeLine(s.shape)}</span>` +
      (s.op ? `<br><span style="color:var(--ink-3)">op&nbsp;&nbsp;&nbsp;</span> &nbsp;<span style="color:${C.accent}">${s.op}</span>` : "") +
      (s.params ? `<br><span style="color:var(--ink-3)">params</span> &nbsp;<span style="color:${C.dim}">${s.params}</span>` : "") +
      `</div>` +
      `<div style="color:var(--ink-2);font-size:.92rem;line-height:1.7">${/[<>]/.test(s.why) ? s.why : shapeLine(s.why)}</div>`;
    cnt.textContent = `step ${i + 1} / ${steps.length}`;
    prev.disabled = i === 0; next.disabled = i === steps.length - 1;
    if (window.tfsEnhance) window.tfsEnhance();
  }
  render();
}

/* ============================================================
   10. ONLINE SOFTMAX — FlashAttention's running rescale
   ============================================================ */
function vizOnline(el, d) {
  const xs = (d.x || "1.0,3.0,2.0,5.0,4.0,0.5,6.0,2.5").split(",").map(Number);
  panel(el, "Online (streaming) softmax",
    "FlashAttention never materialises the full score row. It keeps a running max and a running sum, and rescales as it goes.");
  const wrap = document.createElement("div"); el.appendChild(wrap);
  const row = ctlRow(el); row.style.marginTop = "12px";
  let step = Math.min(4, xs.length);
  slider(row, "tokens seen", 1, xs.length, step, v => { step = v; draw(); });

  function draw() {
    const seen = xs.slice(0, step);
    let mPrev = -Infinity, lPrev = 0;
    const hist = [];
    for (let t = 0; t < seen.length; t++) {
      const x = seen[t];
      const mNew = Math.max(mPrev, x);
      const corr = mPrev === -Infinity ? 0 : Math.exp(mPrev - mNew);
      const lNew = lPrev * corr + Math.exp(x - mNew);
      hist.push({ x, mPrev, mNew, corr, lPrev, lNew });
      mPrev = mNew; lPrev = lNew;
    }
    // reference: true softmax over the seen prefix
    const mx = Math.max(...seen);
    const ref = seen.reduce((a, x) => a + Math.exp(x - mx), 0);
    wrap.innerHTML =
      `<table style="font-family:var(--mono);font-size:12px;margin:0">
      <thead><tr><th>t</th><th class="num">x_t</th><th class="num">m_old</th><th class="num">m_new</th>
      <th class="num">rescale e^(m_old−m_new)</th><th class="num">ℓ_new</th></tr></thead><tbody>` +
      hist.map((h, t) => `<tr${t === hist.length - 1 ? ' style="background:var(--surface-2)"' : ""}>
        <td>${t}</td><td class="num">${h.x.toFixed(1)}</td>
        <td class="num" style="color:${C.faint}">${h.mPrev === -Infinity ? "−∞" : h.mPrev.toFixed(2)}</td>
        <td class="num" style="color:${C.dim}">${h.mNew.toFixed(2)}</td>
        <td class="num" style="color:${C.fg}">${h.corr.toFixed(4)}</td>
        <td class="num" style="color:${C.accent}">${h.lNew.toFixed(4)}</td></tr>`).join("") +
      `</tbody></table>
      <div style="margin-top:11px;font-family:var(--mono);font-size:12px;color:var(--ink-2);line-height:1.8">
        streaming ℓ = <b style="color:${C.accent}">${lPrev.toFixed(6)}</b><br>
        one-shot  ℓ = <b style="color:${C.dim}">${ref.toFixed(6)}</b><br>
        <span style="color:${C.accent}">Identical — to floating-point rounding. That equality is the whole theorem behind FlashAttention:
        you can compute an exact softmax without ever holding the full row in memory.</span>
      </div>`;
  }
  draw();
}

/* ============================================================
   12. LOSS CURVE / scaling law explorer
   ============================================================ */
function vizChinchilla(el, d) {
  panel(el, "Compute-optimal training (Chinchilla)",
    "Given a fixed compute budget C ≈ 6ND, how should you split it between model size N and tokens D?");
  const row = ctlRow(el);
  let logC = 22, N = 7e9;
  const out = document.createElement("div");
  out.style.cssText = "font-family:var(--mono);font-size:12.5px;line-height:1.95;color:var(--ink-2)";
  el.appendChild(out);
  slider(row, "log10(FLOPs)", 18, 27, logC, v => { logC = v; draw(); });
  slider(row, "log10(params)", 7, 12, Math.round(Math.log10(N)), v => { N = Math.pow(10, v); draw(); });

  function draw() {
    const Cflops = Math.pow(10, logC);
    const D = Cflops / (6 * N);
    // Hoffmann et al. 2022 fitted form
    const E = 1.69, A = 406.4, B = 410.7, a = 0.34, b = 0.28;
    const loss = n => dd => E + A / Math.pow(n, a) + B / Math.pow(dd, b);
    const L = loss(N)(D);
    // optimal split: N* ∝ C^0.5, D* ∝ C^0.5  (≈20 tokens/param)
    const Nopt = Math.sqrt(Cflops / 6 / 20), Dopt = Cflops / (6 * Nopt);
    const Lopt = loss(Nopt)(Dopt);
    const fm = x => x >= 1e12 ? (x / 1e12).toFixed(2) + "T" : x >= 1e9 ? (x / 1e9).toFixed(2) + "B" : (x / 1e6).toFixed(1) + "M";
    out.innerHTML =
      `budget C = 10^${logC} FLOPs &nbsp;(GPT-3 ≈ 3×10^23, GPT-4-class ≈ 10^25–10^26)<br>` +
      `your choice:&nbsp; N = <b style="color:${C.fg}">${fm(N)}</b> params, so D = C/(6N) = <b style="color:${C.fg}">${fm(D)}</b> tokens ` +
      `&nbsp;(<span style="color:${C.accent}">${(D / N).toFixed(1)} tokens/param</span>) → predicted loss <b style="color:${C.accent}">${L.toFixed(3)}</b><br>` +
      `<span style="color:${C.accent}">compute-optimal:</span> N* = ${fm(Nopt)}, D* = ${fm(Dopt)} (20 tok/param) → loss <b style="color:${C.accent}">${Lopt.toFixed(3)}</b><br>` +
      `<span style="color:${C.faint}">Δ = ${(L - Lopt).toFixed(3)} nats. ` +
      `Note the curve is <i>flat</i> near the optimum — which is why real labs deliberately over-train small models ` +
      `(Llama-3-8B saw 15T tokens ≈ 1875 tok/param). Training compute is paid once; inference compute is paid forever.</span>`;
  }
  draw();
}

/* ============================================================
   13. FLOAT — the bit explorer (Ch 00, Ch 12)
   Click a bit, or type a number. Everything shown is computed from
   (E, M, flavour) by numfmt.js — the engine tests/check_numfmt.js
   verifies against PyTorch's real dtypes.
   ============================================================ */
function vizFloat(el, d) {
  const NF = window.TFS_NUMFMT;
  if (!NF) { el.innerHTML = '<div class="tiny">numfmt.js not loaded</div>'; return; }
  const order = (d.formats || "fp32,bf16,fp16,e4m3,e5m2,e2m1").split(",");
  panel(el, "The bit explorer",
    "Click any bit to flip it, or type a number to see where it lands. Every field is decoded live, using this format's own bias.");
  const css = n => (getComputedStyle(document.documentElement).getPropertyValue(n) || "").trim();
  const COL = { s: css("--stamp") || C.danger, e: C.accent, m: C.dim };

  // State lives on the host element, which survives a theme rebuild, so the
  // bit pattern you built by clicking is not lost when the palette changes.
  let st = {};
  try { st = JSON.parse(el.dataset.vizFloat || "{}"); } catch (e) { st = {}; }
  let key = order.includes(st.key) ? st.key : (order.includes(d.format) ? d.format : order[0]);
  let typed = "typed" in st ? (st.typed === null ? NaN : st.typed) : (d.value !== undefined ? +d.value : 0.1);
  if (st.negZero) typed = -0;
  let pattern = typeof st.pattern === "number" ? st.pattern : null;
  const save = () => {
    el.dataset.vizFloat = JSON.stringify({ key, pattern,
      typed: Number.isFinite(typed) ? typed : (Number.isNaN(typed) ? null : typed > 0 ? 1e999 : -1e999),
      negZero: Object.is(typed, -0) });
  };

  const row = ctlRow(el);
  const fl = document.createElement("label");
  fl.innerHTML = 'format <select class="vf-fmt">' +
    order.map(k => `<option value="${k}">${NF.formats[k].label}</option>`).join("") + "</select>";
  row.appendChild(fl);
  const sel = fl.querySelector("select"); sel.value = key;
  const nl = document.createElement("label");
  nl.innerHTML = 'number <input class="vf-num" type="text" inputmode="decimal" size="12" aria-label="number to encode">';
  row.appendChild(nl);
  const num = nl.querySelector("input");
  const presets = document.createElement("div");
  presets.className = "vf-presets";
  presets.style.cssText = "display:flex;flex-wrap:wrap;gap:.35rem;margin:-.4rem 0 .9rem";
  el.appendChild(presets);

  const bitsWrap = document.createElement("div");
  bitsWrap.className = "vf-bits";
  bitsWrap.style.cssText = "overflow-x:auto;padding:2px 2px 6px";
  el.appendChild(bitsWrap);
  const out = document.createElement("div");
  out.className = "vf-out";
  out.style.cssText = "margin-top:10px;font-family:var(--mono);font-size:12.5px;line-height:1.75;color:var(--ink-2);overflow-wrap:anywhere";
  out.setAttribute("aria-live", "polite");
  el.appendChild(out);
  const rng = document.createElement("div");
  rng.className = "vf-range";
  rng.style.cssText = "margin-top:12px;padding-top:10px;border-top:2px solid var(--line);font-family:var(--mono);font-size:12px;line-height:1.7;color:var(--ink-2);overflow-wrap:anywhere";
  el.appendChild(rng);

  const fmtNum = v => {
    if (Number.isNaN(v)) return "NaN";
    if (v === Infinity) return "+inf"; if (v === -Infinity) return "-inf";
    if (v === 0) return Object.is(v, -0) ? "-0" : "0";
    const a = Math.abs(v);
    if (a >= 1e6 || a < 1e-4) return v.toPrecision(6).replace(/\.?0+e/, "e");
    return String(+v.toPrecision(8));
  };
  const toBits = (p, n) => { let b = ""; for (let i = n - 1; i >= 0; i--) b += Math.floor(p / Math.pow(2, i)) % 2; return b; };

  function setPresets(f) {
    const r = NF.range(f);
    const list = [["1", 1], ["0.1", 0.1], ["1/3", 1 / 3], ["max", r.max], ["min normal", r.minNormal],
                  ["min subnormal", r.minSubnormal], ["-0", -0]];
    if (f.flavour === "ieee") list.push(["inf", Infinity]);
    presets.innerHTML = "";
    list.forEach(([lab, v]) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "btn vf-preset"; b.textContent = lab;
      b.style.cssText = "padding:.2rem .55rem;font-size:.7rem";
      b.onclick = () => { typed = v; pattern = null; num.value = fmtNum(v); draw(); };
      presets.appendChild(b);
    });
  }

  function draw() {
    const f = NF.formats[key], n = NF.bits(f), b = NF.bias(f), r = NF.range(f);
    if (pattern === null) {
      const p = NF.encode(f, typed);
      pattern = p === null ? 0 : p;          // fp4/fp6 have no NaN: show zero, say why below
    }
    const dv = NF.decode(f, pattern);
    save();

    bitsWrap.innerHTML = "";
    const line = document.createElement("div");
    line.style.cssText = "display:flex;gap:6px;align-items:flex-end;min-width:max-content";
    const group = (name, from, count, col) => {
      const g = document.createElement("div");
      g.style.cssText = "display:flex;flex-direction:column;gap:3px";
      const lab = document.createElement("div");
      lab.style.cssText = `font-family:var(--label);font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:${col}`;
      lab.textContent = name + (count > 1 ? " · " + count : "");
      const cells = document.createElement("div");
      cells.style.cssText = `display:flex;gap:${n > 16 ? 1 : 2}px`;
      for (let i = from + count - 1; i >= from; i--) {
        const bit = Math.floor(pattern / Math.pow(2, i)) % 2;
        const c = document.createElement("button");
        c.type = "button"; c.className = "vf-bit"; c.dataset.bit = i;
        c.textContent = bit;
        c.setAttribute("aria-pressed", bit ? "true" : "false");
        c.setAttribute("aria-label", `${name} bit ${i - from}, currently ${bit}`);
        // 32 cells must fit a desktop panel without scrolling; narrower formats get bigger targets
        const cw = n > 16 ? 15 : n > 8 ? 22 : 28;
        c.style.cssText = `width:${cw}px;height:30px;padding:0;cursor:pointer;` +
          `font-family:var(--mono);font-size:${n > 16 ? 11 : 13}px;font-weight:700;border-radius:3px;` +
          `border:2px solid ${col};background:${bit ? col : "var(--surface)"};color:${bit ? "var(--surface)" : col}`;
        c.onclick = () => {
          pattern = bit ? pattern - Math.pow(2, i) : pattern + Math.pow(2, i);
          typed = NF.decode(f, pattern).value; num.value = fmtNum(typed); draw();
          const again = bitsWrap.querySelector(`[data-bit="${i}"]`); if (again) again.focus();
        };
        cells.appendChild(c);
      }
      g.appendChild(lab); g.appendChild(cells); line.appendChild(g);
    };
    group("sign", f.E + f.M, 1, COL.s);
    group("exponent", f.M, f.E, COL.e);
    group("mantissa", 0, f.M, COL.m);
    bitsWrap.appendChild(line);

    const span = (txt, col) => `<b style="color:${col}">${txt}</b>`;
    const mFrac = toBits(dv.m, f.M);
    let eq;
    if (dv.kind === "normal")
      eq = `(&minus;1)<sup>${span(dv.s, COL.s)}</sup> &times; ${span("1." + mFrac, COL.m)}<sub>2</sub> &times; 2<sup>${span(dv.e, COL.e)} &minus; ${b}</sup>` +
           ` = ${dv.s ? "&minus;" : ""}${fmtNum(1 + dv.m / Math.pow(2, f.M))} &times; 2<sup>${dv.e - b}</sup>`;
    else if (dv.kind === "subnormal")
      eq = `exponent field is 0 &rarr; <b>subnormal</b>: no hidden 1, and the exponent is pinned at 1 &minus; ${b}<br>` +
           `(&minus;1)<sup>${span(dv.s, COL.s)}</sup> &times; ${span("0." + mFrac, COL.m)}<sub>2</sub> &times; 2<sup>${1 - b}</sup>`;
    else if (dv.kind === "zero") eq = `exponent and mantissa all zero &rarr; <b>${dv.s ? "negative " : ""}zero</b>`;
    else if (dv.kind === "inf") eq = `exponent all ones, mantissa zero &rarr; <b>${dv.s ? "&minus;" : "+"}infinity</b>`;
    else eq = f.flavour === "fn"
      ? "S.1111.111 &rarr; <b>NaN</b> — the only NaN pattern, which is why this format has no infinity"
      : "exponent all ones, mantissa non-zero &rarr; <b>NaN</b>";

    let near = "";
    if (Number.isFinite(dv.value)) {
      const u = NF.ulp(f, dv.value === 0 ? r.minSubnormal : dv.value);
      near = `<br>gap to the next value here (1 ulp) = ${fmtNum(u)}` +
             (dv.value !== 0 ? ` &nbsp;·&nbsp; relative gap = ${fmtNum(u / Math.abs(dv.value))}` : "");
    }
    let err = "";
    if (Number.isNaN(typed) && f.flavour === "none")
      err = `<br><b style="color:${C.danger}">this format has no NaN pattern at all</b> — every one of its ${Math.pow(2, n)} codes is a number`;
    else if (Number.isFinite(typed) && Number.isFinite(dv.value) && typed !== 0 && typed !== dv.value) {
      const rel = Math.abs(dv.value - typed) / Math.abs(typed);
      err = `<br>you asked for ${fmtNum(typed)} &rarr; stored ${fmtNum(dv.value)} &nbsp;·&nbsp; error ${(rel * 100).toPrecision(3)}%` +
            (dv.value === 0 ? ` <b style="color:${C.danger}">— underflowed to zero</b>` : "") +
            (Math.abs(typed) > r.max ? ` <b style="color:${C.danger}">— saturated at the max</b>` : "");
    } else if (Number.isFinite(typed) && !Number.isFinite(dv.value) && f.flavour === "ieee")
      err = `<br>you asked for ${fmtNum(typed)} <b style="color:${C.danger}">— overflowed to infinity</b> (max is ${fmtNum(r.max)})`;
    out.innerHTML = `bits &nbsp;${span(toBits(dv.s, 1), COL.s)} ${span(toBits(dv.e, f.E), COL.e)} ${span(mFrac, COL.m)}` +
      ` &nbsp;·&nbsp; class: <b>${dv.kind}</b><br>${eq}<br>value = <b class="vf-value" style="font-size:14px;color:var(--ink)">${fmtNum(dv.value)}</b>` + near + err;

    const topE = f.flavour === "ieee" ? Math.pow(2, f.E) - 2 : Math.pow(2, f.E) - 1;
    rng.innerHTML =
      `<b style="color:var(--ink)">${f.label}</b> &nbsp; ${n} bits = 1 + ${f.E} + ${f.M} &nbsp;·&nbsp; ` +
      (f.flavour === "ieee" ? "IEEE-style specials" : f.flavour === "fn" ? "no infinity, one NaN" : "no infinity, no NaN") + "<br>" +
      `bias = 2<sup>${f.E}&minus;1</sup> &minus; 1 = <b>${b}</b><br>` +
      `max = ${fmtNum(r.max * Math.pow(2, -(topE - b)))} &times; 2<sup>${topE} &minus; ${b}</sup> = <b class="vf-max">${fmtNum(r.max)}</b><br>` +
      `min normal = 2<sup>1 &minus; ${b}</sup> = <b>${fmtNum(r.minNormal)}</b> &nbsp;·&nbsp; ` +
      `min subnormal = 2<sup>1 &minus; ${b} &minus; ${f.M}</sup> = <b>${fmtNum(r.minSubnormal)}</b><br>` +
      `eps = 2<sup>&minus;${f.M}</sup> = <b>${fmtNum(r.eps)}</b> &nbsp;·&nbsp; ≈ ${r.digits.toFixed(1)} decimal digits`;
  }

  sel.onchange = () => {
    key = sel.value; setPresets(NF.formats[key]);
    pattern = null;                         // re-encode the same number in the new format
    num.value = fmtNum(typed); draw();
  };
  num.onchange = () => {
    const t = num.value.trim().toLowerCase();
    const v = t === "nan" ? NaN : (t === "inf" || t === "+inf") ? Infinity : t === "-inf" ? -Infinity
            : t === "-0" ? -0 : /^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/.test(t) ? Number(t)
            : /^[-+]?\d+\/\d+$/.test(t) ? t.split("/").map(Number).reduce((a, c) => a / c) : undefined;
    if (v === undefined) { num.setAttribute("aria-invalid", "true"); return; }
    num.removeAttribute("aria-invalid");
    typed = v; pattern = null; draw();
  };
  setPresets(NF.formats[key]);
  num.value = fmtNum(typed);
  draw();
}

/* ============================================================
   registry + boot
   ============================================================ */
const VIZ = {
  matmul: vizMatmul, tensor: vizTensor, broadcast: vizBroadcast, softmax: vizSoftmax,
  attention: vizAttention, heads: vizHeads, rope: vizRope, kv: vizKV, trace: vizTrace,
  online: vizOnline, gqa: vizGQA, chinchilla: vizChinchilla, float: vizFloat,
};
function renderOne(el) {
  const fn = VIZ[el.dataset.viz];
  if (!fn) return;
  try {
    fn(el, el.dataset);
  } catch (e) {
    el.innerHTML = `<div class="tiny" style="color:var(--stop)">viz error: ${e.message}</div>`;
  }
}

function initViz() {
  readTheme();
  document.querySelectorAll("[data-viz]").forEach(el => {
    // Cache the pristine markup so a theme change can rebuild from scratch.
    // (The trace widget consumes an inline <script type="application/json">.)
    if (el.dataset.seed === undefined) el.dataset.seed = el.innerHTML;
    renderOne(el);
  });
}

/* A theme change rebuilds every widget from its seed markup, because widgets
   draw with resolved colour values. That used to throw away whatever the
   reader had set: drag a slider, toggle dark mode, and it snapped back to the
   default with no explanation. So snapshot every control first, by position,
   and replay it into the fresh widget with the same input/change events a
   real interaction fires — each widget then redraws through its own handlers,
   exactly as if the reader had set the value again. */
function snapshotControls(el) {
  return [...el.querySelectorAll("input, select")].map(c =>
    c.type === "checkbox" || c.type === "radio" ? { checked: c.checked } : { value: c.value });
}
function restoreControls(el, snap) {
  const ctl = [...el.querySelectorAll("input, select")];
  snap.forEach((s, i) => {
    const c = ctl[i];
    if (!c) return;
    if ("checked" in s) {
      if (c.checked === s.checked) return;
      c.checked = s.checked;
      c.dispatchEvent(new Event("change", { bubbles: true }));
    } else {
      if (c.value === s.value) return;
      c.value = s.value;
      c.dispatchEvent(new Event(c.tagName === "SELECT" || c.type === "text" ? "change" : "input", { bubbles: true }));
    }
  });
}
function rebuildViz() {
  readTheme();
  document.querySelectorAll("[data-viz]").forEach(el => {
    if (el.dataset.seed === undefined) return;
    const snap = snapshotControls(el);
    el.classList.remove("panel");
    el.innerHTML = el.dataset.seed;
    renderOne(el);
    try { restoreControls(el, snap); } catch (e) { /* a widget that cannot replay keeps its defaults */ }
  });
}

/* Widgets draw with RESOLVED colour values, so they must be redrawn when the
   palette changes — on the explicit toggle and on an OS-level change alike. */
window.addEventListener("themechange", rebuildViz);
if (window.matchMedia) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    requestAnimationFrame(rebuildViz);   // let the stylesheet recompute first
  });
}

window.initViz = initViz;
