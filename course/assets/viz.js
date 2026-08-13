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
    q:      v("--dim-1", "#A8761C"),
    k:      v("--dim-2", "#4A51B8"),
    v:      v("--dim-3", "#A8484F"),
    o:      v("--dim-4", "#5A7A2E"),
    accent: v("--accent", "#0B6E62"),
    grid:   v("--line", "#DFE0DD"),
    gridHi: v("--line-2", "#C6C8C4"),
    cell:   v("--surface-2", "#F2F3F1"),
    paper:  v("--paper", "#FAFAF9"),
    fg:     v("--ink", "#18202B"),
    dim:    v("--ink-2", "#46505E"),
    faint:  v("--ink-3", "#6E7885"),
    fainter:v("--ink-4", "#9AA2AC"),
    danger: v("--stop", "#A33028"),
    ok:     v("--ok", "#2F6B3A"),
  });
  return C;
}
/* hue family used where a widget needs N distinguishable colours */
function hues() {
  return [C.q, C.k, C.v, C.o, C.accent, C.danger, C.ok, C.dim];
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
    grid(sv, 10, y0, A, m, k, C.q, hi, -1);
    label(sv, 10 + aw / 2, 22, `A  (${m}, ${k})`, C.q, 12);
    label(sv, 10 + aw / 2, y0 + m * cs + 18, `${k} columns`, C.faint, 10);

    // ×
    label(sv, 10 + aw + gap / 2, y0 + Math.max(m, k) * cs / 2, "×", C.dim, 20);

    // B
    const bx = 10 + aw + gap;
    grid(sv, bx, y0, B, k, n, C.k, -1, hj);
    label(sv, bx + bw / 2, 22, `B  (${k}, ${n})`, C.k, 12);
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
        fill: on ? C.o + "44" : C.cell, stroke: on ? C.o : C.grid, "stroke-width": on ? 2 : 1,
      }));
      const t = svgEl("text", {
        x: cx + j * cs + (cs - 2) / 2, y: y0 + i * cs + (cs - 2) / 2 + 4, "text-anchor": "middle",
        "font-size": 11, "font-family": "var(--mono)", fill: on ? C.o : C.faint, "font-weight": on ? 800 : 400,
      });
      t.textContent = Cm[i][j]; g.appendChild(t);
    }
    sv.appendChild(g);
    label(sv, cx + cw / 2, 22, `C  (${m}, ${n})`, C.o, 12);
    wrap.appendChild(sv);

    const terms = A[hi].map((a, t) => `${a}×${B[t][hj]}`).join(" + ");
    const prods = A[hi].map((a, t) => a * B[t][hj]).join(" + ");
    expl.innerHTML =
      `<span style="color:${C.o}">C[${hi},${hj}]</span> = dot( <span style="color:${C.q}">A[${hi},:]</span> , <span style="color:${C.k}">B[:,${hj}]</span> )<br>` +
      `&nbsp;&nbsp;= ${terms}<br>&nbsp;&nbsp;= ${prods} = <b style="color:${C.o}">${Cm[hi][hj]}</b><br>` +
      `<span style="color:${C.faint}">The shared dimension k=${k} is summed away. It does not appear in the output shape.</span>`;
  }
  draw();
}

/* ============================================================
   2. TENSOR — isometric view of a rank-1/2/3 tensor
   ============================================================ */
function vizTensor(el, d) {
  const dims = (d.dims || "2,4,3").split(",").map(Number);
  const names = (d.names || "batch,time,channel").split(",");
  panel(el, d.title || `Tensor of shape (${dims.join(", ")})`,
    d.sub || "Each small square is one number. Rightmost axis is contiguous in memory.");

  const wrap = document.createElement("div"); wrap.style.overflowX = "auto"; el.appendChild(wrap);
  const [B, T, Ch] = dims.length === 3 ? dims : dims.length === 2 ? [1, ...dims] : [1, 1, dims[0]];
  const cs = 22, dx = 13, dy = -9;
  const W = Ch * cs + (B - 1) * Math.abs(dx) + 200, H = T * cs + (B - 1) * Math.abs(dy) + 90;
  const sv = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: Math.min(W, 820), height: H });
  const ox = 60, oy = 50 + (B - 1) * Math.abs(dy);

  for (let b = B - 1; b >= 0; b--) {
    const sx = ox + b * dx, sy = oy + b * dy;
    const alpha = 1 - b * 0.13;
    for (let t = 0; t < T; t++) for (let c = 0; c < Ch; c++) {
      sv.appendChild(svgEl("rect", {
        x: sx + c * cs, y: sy + t * cs, width: cs - 2, height: cs - 2, rx: 3,
        fill: C.cell, stroke: b === 0 ? C.accent : C.grid, "stroke-width": b === 0 ? 1.3 : 1,
        opacity: alpha,
      }));
    }
    if (B > 1) {
      const t = svgEl("text", { x: sx - 8, y: sy + 12, "text-anchor": "end", "font-size": 10, fill: C.faint, "font-family": "var(--mono)" });
      t.textContent = `[${b}]`; sv.appendChild(t);
    }
  }
  const lab = (x, y, txt, col, size = 11, anchor = "middle") => {
    const t = svgEl("text", { x, y, "text-anchor": anchor, "font-size": size, fill: col, "font-family": "var(--mono)", "font-weight": 600 });
    t.textContent = txt; sv.appendChild(t);
  };
  // axis labels
  if (dims.length === 3) {
    lab(ox + (B - 1) * dx / 2 + 40, oy + (B - 1) * dy - 22, `axis 0: ${names[0]} = ${B}`, C.v, 11, "start");
    lab(ox - 14, oy + T * cs / 2, `axis 1`, C.k, 11, "end");
    lab(ox - 14, oy + T * cs / 2 + 13, `${names[1]}=${T}`, C.k, 10, "end");
    lab(ox + Ch * cs / 2, oy + T * cs + 20, `axis 2: ${names[2]} = ${Ch}  ← contiguous`, C.q, 11);
  } else {
    lab(ox - 14, oy + T * cs / 2, `${names[0] || "rows"}=${T}`, C.k, 10, "end");
    lab(ox + Ch * cs / 2, oy + T * cs + 20, `${names[1] || "cols"}=${Ch}`, C.q, 11);
  }
  lab(W - 150, oy + 4, `numel = ${dims.reduce((a, b) => a * b, 1)}`, C.dim, 11, "start");
  lab(W - 150, oy + 22, `rank  = ${dims.length}`, C.dim, 11, "start");
  lab(W - 150, oy + 40, `${(dims.reduce((a, b) => a * b, 1) * 4 / 1024).toFixed(2)} KB @ fp32`, C.faint, 10, "start");
  wrap.appendChild(sv);
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
      `<div><span style="display:inline-block;width:52px;color:${C.q}">A</span>${rowHtml(pa, C.q)}  <span style="color:${C.faint}">← (${a.join(", ")})</span></div>` +
      `<div><span style="display:inline-block;width:52px;color:${C.k}">B</span>${rowHtml(pb, C.k)}  <span style="color:${C.faint}">← (${b.join(", ")})</span></div>` +
      `<div style="border-top:1px solid ${C.grid};margin:6px 0 4px"></div>` +
      `<div><span style="display:inline-block;width:52px;color:${okAll ? C.o : C.danger}">${okAll ? "out" : "ERROR"}</span>` +
      outs.map(o => cell(o.v, o.ok ? C.o : C.danger)).join("") +
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
    const cs = 40, pad = 96;
    const W = pad + T * cs + 30, Hh = pad + T * cs + 20;
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
        fill: C.k, "font-family": "var(--mono)",
        transform: `rotate(-52 ${pad + j * cs + cs / 2} ${pad - 10})`,
      });
      t.textContent = toks[j] || "t" + j; sv.appendChild(t);
    }
    const kl = svgEl("text", { x: pad + T * cs / 2, y: 16, "text-anchor": "middle", "font-size": 11, fill: C.k, "font-family": "var(--mono)", "font-weight": 700 });
    kl.textContent = "KEYS  (what is available to look at) →"; sv.appendChild(kl);

    for (let i = 0; i < T; i++) {
      const t = svgEl("text", { x: pad - 10, y: pad + i * cs + cs / 2 + 4, "text-anchor": "end", "font-size": 11, fill: C.q, "font-family": "var(--mono)" });
      t.textContent = toks[i] || "t" + i; sv.appendChild(t);
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
      const rs = svgEl("text", { x: pad + T * cs + 6, y: pad + i * cs + cs / 2 + 4, "font-size": 9.5, fill: C.o, "font-family": "var(--mono)" });
      rs.textContent = "Σ" + P[i].reduce((a, b) => a + b, 0).toFixed(2); sv.appendChild(rs);
    }
    const ql = svgEl("text", { x: 14, y: pad + T * cs / 2, "text-anchor": "middle", "font-size": 11, fill: C.q, "font-family": "var(--mono)", "font-weight": 700, transform: `rotate(-90 14 ${pad + T * cs / 2})` });
    ql.textContent = "← QUERIES (who is asking)"; sv.appendChild(ql);
    wrap.appendChild(sv);

    const maxOff = Math.max(...P.flat());
    note.innerHTML =
      `shape of this matrix: <b style="color:${C.accent}">(T=${T}, T=${T})</b> — one row per query, one column per key.<br>` +
      (causal ? `Causal mask ON: position i cannot see j &gt; i. Those cells got −∞ <i>before</i> softmax, so they become exactly 0.<br>` :
                `<span style="color:${C.danger}">Causal mask OFF</span>: every token sees the future. Fine for BERT, catastrophic for a generator.<br>`) +
      (scale ? `Scaled by 1/√d_k with d_k=${dk} → scores stay in a range where softmax has usable gradients. Peak weight ${maxOff.toFixed(2)}.` :
               `<span style="color:${C.danger}">Unscaled</span>: raw dot products grow like d_k, softmax saturates, gradients vanish. Peak weight ${maxOff.toFixed(2)}.`);
  }
  draw();
}

/* ============================================================
   6. HEADS — the multi-head reshape dance
   ============================================================ */
function vizHeads(el, d) {
  const B = 1, T = +d.t || 4, dmodel = +d.dmodel || 12;
  panel(el, "The reshape dance: (B, T, d_model) → (B, n_head, T, d_head)",
    "Nothing is computed here. The same numbers are just re-labelled and re-ordered so each head sees its own slice.");
  const row = ctlRow(el);
  let nh = +d.nh || 3;
  const wrap = document.createElement("div"); wrap.style.overflowX = "auto"; el.appendChild(wrap);
  const code = document.createElement("div");
  code.style.cssText = "margin-top:12px;font-family:var(--mono);font-size:12px;color:var(--ink-2);line-height:1.9";
  el.appendChild(code);
  const opts = [1, 2, 3, 4, 6, 12].filter(x => dmodel % x === 0);
  const sel = document.createElement("label");
  sel.innerHTML = "n_head " + `<select>${opts.map(o => `<option ${o === nh ? "selected" : ""}>${o}</option>`).join("")}</select>`;
  row.appendChild(sel);
  sel.querySelector("select").onchange = e => { nh = +e.target.value; draw(); };

  const HUE = hues();

  function draw() {
    const dh = dmodel / nh, cs = 24;
    wrap.innerHTML = "";
    const W = 700, Hh = 250;
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${Hh}`, width: W, height: Hh });
    const lab = (x, y, txt, col, size = 11, anchor = "start", w = 600) => {
      const t = svgEl("text", { x, y, "text-anchor": anchor, "font-size": size, fill: col, "font-family": "var(--mono)", "font-weight": w });
      t.textContent = txt; sv.appendChild(t);
    };

    // STEP 1: (T, d_model) flat
    lab(10, 18, `1. after W_q:  (B=1, T=${T}, d_model=${dmodel})`, C.dim, 11);
    for (let t = 0; t < T; t++) for (let c = 0; c < dmodel; c++) {
      const h = Math.floor(c / dh);
      sv.appendChild(svgEl("rect", { x: 10 + c * cs, y: 28 + t * cs, width: cs - 2, height: cs - 2, rx: 3, fill: HUE[h % 12] + "26", stroke: HUE[h % 12] + "88" }));
    }
    lab(10 + dmodel * cs / 2, 28 + T * cs + 16, `d_model = ${dmodel} contiguous features`, C.faint, 10, "middle", 400);

    // arrow
    const ay = 28 + T * cs / 2;
    lab(10 + dmodel * cs + 22, ay + 4, "→", C.dim, 20);

    // STEP 2: split into heads
    const x2 = 10 + dmodel * cs + 52;
    lab(x2, 18, `2. .view(B, T, ${nh}, ${dh})`, C.dim, 11);
    for (let hh = 0; hh < nh; hh++) {
      for (let t = 0; t < T; t++) for (let c = 0; c < dh; c++) {
        sv.appendChild(svgEl("rect", {
          x: x2 + hh * (dh * cs + 12) + c * cs, y: 28 + t * cs,
          width: cs - 2, height: cs - 2, rx: 3, fill: HUE[hh % 12] + "26", stroke: HUE[hh % 12] + "88",
        }));
      }
      lab(x2 + hh * (dh * cs + 12) + dh * cs / 2 - 4, 28 + T * cs + 16, `h${hh}`, HUE[hh % 12], 10, "middle");
    }

    // STEP 3
    const y3 = 28 + T * cs + 44;
    lab(10, y3 + 14, `3. .transpose(1, 2)  →  (B, n_head=${nh}, T=${T}, d_head=${dh})   ← each head is now its own independent (T, d_head) matrix`, C.accent, 11);
    for (let hh = 0; hh < nh; hh++) {
      const bx = 10 + hh * (dh * cs + 30);
      for (let t = 0; t < T; t++) for (let c = 0; c < dh; c++) {
        sv.appendChild(svgEl("rect", {
          x: bx + c * cs, y: y3 + 26 + t * cs, width: cs - 2, height: cs - 2, rx: 3,
          fill: HUE[hh % 12] + "26", stroke: HUE[hh % 12] + "88",
        }));
      }
      sv.appendChild(svgEl("rect", {
        x: bx - 5, y: y3 + 21, width: dh * cs + 6, height: T * cs + 8, rx: 6,
        fill: "none", stroke: HUE[hh % 12], "stroke-width": 1.4, "stroke-dasharray": "4 3",
      }));
      lab(bx + dh * cs / 2 - 4, y3 + 26 + T * cs + 16, `head ${hh}: (${T}, ${dh})`, HUE[hh % 12], 10, "middle");
    }
    wrap.appendChild(sv);

    code.innerHTML =
      `<span style="color:${C.faint}"># B=1, T=${T}, d_model=${dmodel}, n_head=${nh}, d_head=${dmodel}/${nh}=${dh}</span><br>` +
      `q = q.view(B, T, ${nh}, ${dh}).transpose(1, 2)&nbsp;&nbsp;<span style="color:${C.faint}"># (1, ${nh}, ${T}, ${dh})</span><br>` +
      `<b style="color:${C.o}">Key insight:</b> d_model is <i>partitioned</i>, not duplicated. ` +
      `Total parameters and FLOPs are the same as one big head — you just get ${nh} independent attention patterns for free.`;
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
    arrow(ang, C.q, `rotated by mθ`);

    // arc
    const steps = 26, pts = [];
    for (let s = 0; s <= steps; s++) { const a = ang * s / steps; pts.push(`${cx + 34 * Math.cos(a)},${cy - 34 * Math.sin(a)}`); }
    sv.appendChild(svgEl("polyline", { points: pts.join(" "), fill: "none", stroke: C.o, "stroke-width": 1.6 }));
    const at = svgEl("text", { x: cx + 42 * Math.cos(ang / 2), y: cy - 42 * Math.sin(ang / 2) + 4, "font-size": 11, fill: C.o, "font-family": "var(--mono)" });
    at.textContent = `mθ`; sv.appendChild(at);

    // right panel: relative-position property
    const tx = 245;
    const rows = [
      [`θ_i = base^(−2i/d_head)`, C.dim],
      [`    = ${base}^(−${2 * pairIdx}/${dhead}) = ${theta.toExponential(3)}`, C.faint],
      [`m·θ_i = ${m} × ${theta.toExponential(3)} = ${ang.toFixed(4)} rad`, C.q],
      [``, C.dim],
      [`Rotate q by mθ, k by nθ. Then:`, C.dim],
      [`⟨R(m)q, R(n)k⟩ = ⟨q, R(n−m)k⟩`, C.o],
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
    note.innerHTML = `Pair i=${pairIdx} of d_head=${dhead} completes one full turn every <b style="color:${C.o}">${wavelength.toFixed(1)}</b> tokens. ` +
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
      bar(`MHA (${nh} KV heads)`, mha, mha, C.q) +
      bar(`GQA (${kvh} KV heads)`, gqa, mha, C.k) +
      bar(`MQA (1 KV head)`, mqa, mha, C.v) +
      bar(`MLA (576/token/layer)`, mla, mha, C.o) +
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
      `<div style="font-family:var(--mono);font-size:13.5px;background:var(--surface);border:1px solid var(--line);border-left:3px solid ${C.o};padding:11px 14px;margin-bottom:11px">` +
      `<span style="color:var(--ink-3)">shape</span> &nbsp;<b style="color:${C.o}">${s.shape}</b>` +
      (s.op ? `<br><span style="color:var(--ink-3)">op&nbsp;&nbsp;&nbsp;</span> &nbsp;<span style="color:${C.accent}">${s.op}</span>` : "") +
      (s.params ? `<br><span style="color:var(--ink-3)">params</span> &nbsp;<span style="color:${C.v}">${s.params}</span>` : "") +
      `</div>` +
      `<div style="color:var(--ink-2);font-size:.92rem;line-height:1.7">${s.why}</div>`;
    cnt.textContent = `step ${i + 1} / ${steps.length}`;
    prev.disabled = i === 0; next.disabled = i === steps.length - 1;
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
        <td class="num" style="color:${C.k}">${h.mNew.toFixed(2)}</td>
        <td class="num" style="color:${C.q}">${h.corr.toFixed(4)}</td>
        <td class="num" style="color:${C.o}">${h.lNew.toFixed(4)}</td></tr>`).join("") +
      `</tbody></table>
      <div style="margin-top:11px;font-family:var(--mono);font-size:12px;color:var(--ink-2);line-height:1.8">
        streaming ℓ = <b style="color:${C.o}">${lPrev.toFixed(6)}</b><br>
        one-shot  ℓ = <b style="color:${C.k}">${ref.toFixed(6)}</b><br>
        <span style="color:${C.accent}">Identical — to floating-point rounding. That equality is the whole theorem behind FlashAttention:
        you can compute an exact softmax without ever holding the full row in memory.</span>
      </div>`;
  }
  draw();
}

/* ============================================================
   11. GQA GROUPING — which query heads share which KV head
   ============================================================ */
function vizGQA(el, d) {
  panel(el, "MHA → GQA → MQA is one slider", "Query heads are grouped; each group shares one K/V head.");
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
    const per = nh / g;
    const cw = 60, W = nh * cw + 40, Hh = 178;
    wrap.innerHTML = "";
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${Hh}`, width: "100%", height: Hh, preserveAspectRatio: "xMidYMid meet" });
    const HUE = hues();
    const lab = (x, y, t, c, s = 11, a = "middle") => {
      const e = svgEl("text", { x, y, "text-anchor": a, "font-size": s, fill: c, "font-family": "var(--mono)", "font-weight": 600 });
      e.textContent = t; sv.appendChild(e);
    };
    lab(20, 20, "Query heads", C.q, 11, "start");
    for (let h = 0; h < nh; h++) {
      const grp = Math.floor(h / per);
      sv.appendChild(svgEl("rect", { x: 20 + h * cw, y: 30, width: cw - 8, height: 34, rx: 6, fill: HUE[grp % 8] + "26", stroke: HUE[grp % 8], "stroke-width": 1.4 }));
      lab(20 + h * cw + (cw - 8) / 2, 51, "Q" + h, HUE[grp % 8], 12);
    }
    lab(20, 104, `KV heads (${g})`, C.k, 11, "start");
    for (let j = 0; j < g; j++) {
      const x = 20 + j * per * cw, w = per * cw - 8;
      sv.appendChild(svgEl("rect", { x, y: 114, width: w, height: 34, rx: 6, fill: HUE[j % 8] + "26", stroke: HUE[j % 8], "stroke-width": 1.4 }));
      lab(x + w / 2, 135, `K${j} / V${j}`, HUE[j % 8], 12);
      for (let h = j * per; h < (j + 1) * per; h++) {
        sv.appendChild(svgEl("line", {
          x1: 20 + h * cw + (cw - 8) / 2, y1: 64, x2: x + w / 2, y2: 114,
          stroke: HUE[j % 8], "stroke-width": 1.2, opacity: .55,
        }));
      }
    }
    wrap.appendChild(sv);
    const name = g === nh ? "MHA (Multi-Head Attention)" : g === 1 ? "MQA (Multi-Query Attention)" : "GQA (Grouped-Query Attention)";
    note.innerHTML =
      `<b style="color:${C.accent}">${name}</b> — ${nh} query heads, ${g} KV head${g > 1 ? "s" : ""}, ${per} quer${per > 1 ? "ies" : "y"} per group.<br>` +
      `KV cache is <b style="color:${C.o}">${(nh / g).toFixed(0)}× smaller</b> than MHA. Q projection is unchanged; only W_k and W_v shrink.<br>` +
      `<span style="color:${C.faint}">In code the K/V tensors are repeat_interleave'd back up to ${nh} heads before the matmul — ` +
      `or, in a fused kernel, simply re-read. The saving is memory and bandwidth, not FLOPs.</span>`;
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
      `your choice:&nbsp; N = <b style="color:${C.q}">${fm(N)}</b> params, so D = C/(6N) = <b style="color:${C.k}">${fm(D)}</b> tokens ` +
      `&nbsp;(<span style="color:${C.o}">${(D / N).toFixed(1)} tokens/param</span>) → predicted loss <b style="color:${C.o}">${L.toFixed(3)}</b><br>` +
      `<span style="color:${C.accent}">compute-optimal:</span> N* = ${fm(Nopt)}, D* = ${fm(Dopt)} (20 tok/param) → loss <b style="color:${C.accent}">${Lopt.toFixed(3)}</b><br>` +
      `<span style="color:${C.faint}">Δ = ${(L - Lopt).toFixed(3)} nats. ` +
      `Note the curve is <i>flat</i> near the optimum — which is why real labs deliberately over-train small models ` +
      `(Llama-3-8B saw 15T tokens ≈ 1875 tok/param). Training compute is paid once; inference compute is paid forever.</span>`;
  }
  draw();
}

/* ============================================================
   registry + boot
   ============================================================ */
const VIZ = {
  matmul: vizMatmul, tensor: vizTensor, broadcast: vizBroadcast, softmax: vizSoftmax,
  attention: vizAttention, heads: vizHeads, rope: vizRope, kv: vizKV, trace: vizTrace,
  online: vizOnline, gqa: vizGQA, chinchilla: vizChinchilla,
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

function rebuildViz() {
  readTheme();
  document.querySelectorAll("[data-viz]").forEach(el => {
    if (el.dataset.seed === undefined) return;
    el.classList.remove("panel");
    el.innerHTML = el.dataset.seed;
    renderOne(el);
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
