/* ============================================================================
   numfmt.js — every number format in the course, decoded and encoded from the
   bit fields alone.

   One generic engine instead of one function per dtype, because the chapter's
   whole argument is that a format is nothing but three numbers — exponent
   bits E, mantissa bits M, and a rule for the special values — and that the
   range, the precision and every representable value follow mechanically.

   Verified against PyTorch's real dtypes: all 256 patterns of fp8 e4m3fn and
   e5m2 decode identically, and encode() matches torch's casts for fp16, bf16,
   e4m3 and e5m2 on a fixed vector set (tests/check_numfmt.js).

   FLAVOURS — the only thing that differs between formats besides E and M:
     ieee  the all-ones exponent is reserved: mantissa 0 -> inf, else NaN.
           fp32, tf32, fp16, bf16, fp8 e5m2.
     fn    "finite": no infinities. Only S.1111.111 is NaN, so the top
           exponent is otherwise usable — that is how e4m3 reaches 448
           instead of 240. Out-of-range values saturate.
     none  no inf and no NaN at all: every pattern is a number. fp6, fp4.
   ========================================================================= */

var TFS_FORMATS = {
  fp32:  { label: "fp32",      E: 8, M: 23, flavour: "ieee" },
  tf32:  { label: "tf32",      E: 8, M: 10, flavour: "ieee" },
  bf16:  { label: "bf16",      E: 8, M: 7,  flavour: "ieee" },
  fp16:  { label: "fp16",      E: 5, M: 10, flavour: "ieee" },
  e5m2:  { label: "fp8 e5m2",  E: 5, M: 2,  flavour: "ieee" },
  e4m3:  { label: "fp8 e4m3",  E: 4, M: 3,  flavour: "fn" },
  e3m2:  { label: "fp6 e3m2",  E: 3, M: 2,  flavour: "none" },
  e2m3:  { label: "fp6 e2m3",  E: 2, M: 3,  flavour: "none" },
  e2m1:  { label: "fp4 e2m1",  E: 2, M: 1,  flavour: "none" },
};

(function (root) {
  function bias(f) { return Math.pow(2, f.E - 1) - 1; }
  function bitsOf(f) { return 1 + f.E + f.M; }
  function topExpField(f) { return f.flavour === "ieee" ? Math.pow(2, f.E) - 2 : Math.pow(2, f.E) - 1; }
  /* largest mantissa allowed at the top exponent: fn spends 111 on NaN */
  function topMantissa(f) { return f.flavour === "fn" ? Math.pow(2, f.M) - 2 : Math.pow(2, f.M) - 1; }

  /* The five numbers that describe a format, from the formulas in Ch 00. */
  function range(f) {
    var b = bias(f);
    return {
      bits: bitsOf(f), bias: b,
      max: (1 + topMantissa(f) / Math.pow(2, f.M)) * Math.pow(2, topExpField(f) - b),
      minNormal: Math.pow(2, 1 - b),
      minSubnormal: Math.pow(2, 1 - b - f.M),
      eps: Math.pow(2, -f.M),
      digits: (f.M + 1) * Math.LOG10E * Math.LN2,
    };
  }

  /* bit pattern (integer) -> { value, kind, s, e, m } */
  function decode(f, pattern) {
    var M = f.M, E = f.E, b = bias(f);
    var m = pattern % Math.pow(2, M);
    var e = Math.floor(pattern / Math.pow(2, M)) % Math.pow(2, E);
    var s = Math.floor(pattern / Math.pow(2, M + E)) % 2;
    var sign = s ? -1 : 1;
    var allOnes = Math.pow(2, E) - 1;
    if (f.flavour === "ieee" && e === allOnes)
      return { value: m ? NaN : sign * Infinity, kind: m ? "nan" : "inf", s: s, e: e, m: m };
    if (f.flavour === "fn" && e === allOnes && m === Math.pow(2, M) - 1)
      return { value: NaN, kind: "nan", s: s, e: e, m: m };
    if (e === 0) {
      var sub = (m / Math.pow(2, M)) * Math.pow(2, 1 - b);
      return { value: m ? sign * sub : (s ? -0 : 0), kind: m ? "subnormal" : "zero", s: s, e: e, m: m };
    }
    return { value: sign * (1 + m / Math.pow(2, M)) * Math.pow(2, e - b), kind: "normal", s: s, e: e, m: m };
  }

  function pack(f, s, e, m) { return (s * Math.pow(2, f.E) + e) * Math.pow(2, f.M) + m; }

  /* round half to even on a non-negative real */
  function rne(v) {
    var fl = Math.floor(v), d = v - fl;
    if (d > 0.5) return fl + 1;
    if (d < 0.5) return fl;
    return fl % 2 === 0 ? fl : fl + 1;
  }

  /* real number -> nearest representable, round-half-to-even, as PyTorch does.
     Overflow: ieee -> inf; fn and none -> saturate to max (torch's measured
     behaviour for e4m3fn: 500 -> 448). */
  function encode(f, x) {
    var b = bias(f), M = f.M, s = (x < 0 || Object.is(x, -0)) ? 1 : 0;
    var r = range(f);
    if (x !== x) {
      if (f.flavour === "ieee") return pack(f, 0, Math.pow(2, f.E) - 1, Math.pow(2, M - 1));
      if (f.flavour === "fn") return pack(f, 0, Math.pow(2, f.E) - 1, Math.pow(2, M) - 1);
      return null;                                  // no NaN to give
    }
    var a = Math.abs(x);
    var saturate = function () { return pack(f, s, topExpField(f), topMantissa(f)); };
    if (a === Infinity) return f.flavour === "ieee" ? pack(f, s, Math.pow(2, f.E) - 1, 0) : saturate();
    if (a === 0) return pack(f, s, 0, 0);
    var e, n;
    if (a < r.minNormal) {                          // subnormal: fixed spacing 2^(1-bias-M)
      n = rne(a / r.minSubnormal);
      if (n >= Math.pow(2, M)) return pack(f, s, 1, 0);   // rounded up into the normals
      return pack(f, s, 0, n);
    }
    e = Math.floor(Math.log2(a));
    while (Math.pow(2, e) > a) e--;                 // guard log2 rounding at exact powers
    while (Math.pow(2, e + 1) <= a) e++;
    n = rne((a / Math.pow(2, e) - 1) * Math.pow(2, M));
    if (n === Math.pow(2, M)) { n = 0; e++; }       // 1.111.. rounded up to the next binade
    var ef = e + b;
    if (ef > topExpField(f) || (ef === topExpField(f) && n > topMantissa(f)))
      return f.flavour === "ieee" ? pack(f, s, Math.pow(2, f.E) - 1, 0) : saturate();
    return pack(f, s, ef, n);
  }

  function quantize(f, x) { var p = encode(f, x); return p === null ? NaN : decode(f, p).value; }

  /* spacing between representable values around x (one ULP) */
  function ulp(f, x) {
    var a = Math.abs(x), r = range(f);
    if (a < r.minNormal) return r.minSubnormal;
    return Math.pow(2, Math.floor(Math.log2(a)) - f.M);
  }

  var api = { formats: TFS_FORMATS, bias: bias, bits: bitsOf, range: range, decode: decode,
              encode: encode, quantize: quantize, ulp: ulp, pack: pack };
  root.TFS_NUMFMT = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : this);
