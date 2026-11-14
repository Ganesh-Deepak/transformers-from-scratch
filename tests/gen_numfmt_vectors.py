"""Regenerate tests/fixtures/numfmt_vectors.json from PyTorch's real dtypes.

    python tests/gen_numfmt_vectors.py

The fixture is committed, so check_numfmt.js runs without torch. Re-run this
only if the vector set changes. Values are chosen to hit every case that has
gone wrong somewhere in the course: ties, binade edges, subnormals, the zero
threshold, overflow, saturation, and plain random magnitudes."""
import sys, json, math, random, pathlib, torch
sys.stdout.reconfigure(encoding="utf-8")
random.seed(1234)
xs = [0.0, 1.0, -1.0, 0.1, 1/3, 3.0, 300.0, 1000.0, 65504.0, 65520.0, 70000.0, 448.0, 464.0,
      500.0, 57344.0, 61440.0, 1e-7, 7e-8, 3e-8, 2.5e-8, 2**-25, 2**-24, 2**-14, 0.999, 0.995,
      257.0, 258.0, 259.0, 2049.0, 2050.0, 2051.0, 17.0, -2.6, 0.3, 0.02, 0.001, 0.0009,
      1.0625, 1.1875, 6e-5, 3.0e38, -1e-30, 1e-38]
for _ in range(400):
    xs.append(random.choice([-1, 1]) * 10 ** random.uniform(-9, 6))
for _ in range(200):                         # exact ties are where rounding bugs live
    k = random.randint(-10, 10); m = random.randint(0, 255) * 2 + 1
    xs.append((1 + m / 512) * 2.0 ** k)
dts = {"fp16": torch.float16, "bf16": torch.bfloat16,
       "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
out = {"x": xs, "expect": {}}
for name, dt in dts.items():
    t = torch.tensor(xs, dtype=torch.float64).to(torch.float32).to(dt).to(torch.float64).tolist()
    out["expect"][name] = [None if (isinstance(v, float) and math.isinf(v)) and False else v for v in t]
# x is first rounded to fp32 by torch, so record the fp32 value the cast actually saw
out["x32"] = torch.tensor(xs, dtype=torch.float64).to(torch.float32).to(torch.float64).tolist()
p = pathlib.Path(__file__).parent / "fixtures" / "numfmt_vectors.json"
p.write_text(json.dumps(out, allow_nan=True))
print("wrote", p, len(xs), "vectors x", len(dts), "dtypes")
