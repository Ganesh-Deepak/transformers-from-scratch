/* Number-format engine test — course/assets/numfmt.js against PyTorch.

     cd tests && node check_numfmt.js

   Chapter 00 teaches that a format is just (E, M, flavour) and that every
   range figure follows from five formulas. This test holds the widget that
   demonstrates it to the same standard as the prose: if numfmt.js disagrees
   with real hardware dtypes anywhere, the chapter would be teaching a wrong
   number with an interactive diagram to prove it.

   Three independent checks:
     1. range() reproduces torch.finfo for every format torch implements
     2. encode() reproduces torch's casts on 643 fixed vectors (ties, binade
        edges, subnormals, the zero threshold, overflow, saturation)
     3. for the formats torch cannot cast directly (fp6, fp4) encode() is
        checked against brute force: the nearest of ALL decoded patterns,
        ties broken toward the even mantissa */
const fs = require("fs");
const path = require("path");
const F = require(path.join(__dirname, "..", "course", "assets", "numfmt.js"));

let fail = 0;
const ok = (c, label, extra) => {
  console.log("  " + (c ? "PASS" : "FAIL") + "  " + label + (extra ? "   " + extra : ""));
  if (!c) fail++;
};
const same = (a, b) => (Number.isNaN(a) && Number.isNaN(b)) || a === b || (Math.abs(a - b) <= Math.abs(b) * 1e-12);

console.log("Number-format engine vs PyTorch\n");

/* ---- 1. ranges (values printed by torch.finfo, torch 2.13) ------------------ */
console.log(" ranges");
const FINFO = {
  fp32: { max: 3.4028234663852886e38, minNormal: 1.1754943508222875e-38, eps: 1.1920928955078125e-7 },
  fp16: { max: 65504, minNormal: 6.103515625e-5, eps: 0.0009765625 },
  bf16: { max: 3.3895313892515355e38, minNormal: 1.1754943508222875e-38, eps: 0.0078125 },
  e4m3: { max: 448, minNormal: 0.015625, eps: 0.125 },
  e5m2: { max: 57344, minNormal: 6.103515625e-5, eps: 0.25 },
};
for (const [k, want] of Object.entries(FINFO)) {
  const r = F.range(F.formats[k]);
  ok(same(r.max, want.max) && same(r.minNormal, want.minNormal) && same(r.eps, want.eps),
     (F.formats[k].label).padEnd(10) + " max / min normal / eps", `${r.max} ${r.minNormal} ${r.eps}`);
}
// the small formats, against the OCP MX spec's published maxima
for (const [k, mx] of [["e2m1", 6], ["e2m3", 7.5], ["e3m2", 28]])
  ok(F.range(F.formats[k]).max === mx, F.formats[k].label.padEnd(10) + " max = " + mx);

/* ---- 2. encode() vs torch casts -------------------------------------------- */
console.log("\n encode() vs torch casts");
const raw = fs.readFileSync(path.join(__dirname, "fixtures", "numfmt_vectors.json"), "utf8")
  .replace(/-Infinity/g, "-1e999").replace(/Infinity/g, "1e999").replace(/NaN/g, "null");
const V = JSON.parse(raw);
for (const k of Object.keys(V.expect)) {
  const bad = [];
  V.x32.forEach((x, i) => {
    const got = F.quantize(F.formats[k], x), want = V.expect[k][i];
    if (!same(got, want === null ? NaN : want)) bad.push(`${x} -> ${got} (torch ${want})`);
  });
  ok(bad.length === 0, F.formats[k].label.padEnd(10) + ` ${V.x32.length - bad.length}/${V.x32.length} match`,
     bad.slice(0, 3).join(" | "));
}

/* ---- 3. brute-force oracle for every format of 8 bits or fewer -------------- */
console.log("\n encode() vs brute force (all patterns)");
for (const k of ["e4m3", "e5m2", "e3m2", "e2m3", "e2m1"]) {
  const f = F.formats[k], n = Math.pow(2, F.bits(f));
  const vals = [];
  for (let p = 0; p < n; p++) {
    const d = F.decode(f, p);
    if (Number.isFinite(d.value)) vals.push({ v: d.value, m: d.m });
  }
  const max = F.range(f).max;
  const bad = [];
  // mulberry32: an LCG done in doubles overflows 2^53 and stops being random
  let seed = 7 >>> 0;
  const rnd = () => { seed = (seed + 0x6D2B79F5) >>> 0; let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  for (let i = 0; i < 3000; i++) {
    const x = (rnd() < .5 ? -1 : 1) * max * 1.2 * Math.pow(rnd(), 3);
    let best = null;
    for (const c of vals) {
      if (best === null) { best = c; continue; }
      const dc = Math.abs(c.v - x), db = Math.abs(best.v - x);
      if (dc < db || (dc === db && c.m % 2 === 0 && best.m % 2 === 1)) best = c;
    }
    const want = Math.abs(x) > max ? Math.sign(x) * max : best.v;   // saturating formats
    const got = F.quantize(f, x);
    const wantEff = f.flavour === "ieee" && Math.abs(x) >= max + F.ulp(f, max) / 2 ? Math.sign(x) * Infinity : want;
    if (!same(got, wantEff) && !(got === 0 && wantEff === 0)) bad.push(`${x.toPrecision(5)} -> ${got} (want ${wantEff})`);
  }
  ok(bad.length === 0, f.label.padEnd(10) + " 3000 random values", bad.slice(0, 3).join(" | "));
}

/* ---- 4. the facts the chapter states in prose ------------------------------- */
console.log("\n claims made in the chapter");
const q = (k, x) => F.quantize(F.formats[k], x);
ok(q("bf16", 257) === 256 && q("bf16", 259) === 260, "bf16: 257 -> 256 and 259 -> 260 (ties to even)");
ok(q("fp16", 2049) === 2048, "fp16: 2049 -> 2048");
ok(q("bf16", 0.999) === 1, "bf16: 0.999 rounds to exactly 1.0 (Adam's beta2)");
ok(q("fp16", Math.pow(2, -25)) === 0 && q("fp16", 3e-8) === Math.pow(2, -24),
   "fp16 zero threshold is 2^-25, not 2^-24");
ok(q("e4m3", 300) === 288 && q("e4m3", 500) === 448, "e4m3: 300 -> 288, 500 saturates to 448");
const fp4 = []; for (let p = 0; p < 8; p++) fp4.push(F.decode(F.formats.e2m1, p).value);
ok(fp4.join(",") === "0,0.5,1,1.5,2,3,4,6", "fp4 e2m1 positive values are 0, .5, 1, 1.5, 2, 3, 4, 6", fp4.join(","));

console.log("\n" + (fail === 0 ? "ALL NUMBER-FORMAT CHECKS PASSED" : fail + " NUMBER-FORMAT CHECK(S) FAILED"));
process.exit(fail ? 1 : 0);
