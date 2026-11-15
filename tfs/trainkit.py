"""
The parts of a pretraining run that tfs/train.py leaves out — Chapter 11.6-11.9.

    grad_noise_scale(...)       McCandlish et al. 2018: the batch size past which
                                more data per step stops buying progress
    adam_output_change(...)     why the best learning rate moves with width, and
                                the muP rule that stops it
    flops_per_token / mfu       how fast the run is, as a fraction of the hardware
    z_loss, SpikeDetector       keeping a big run from blowing up
    ResumableRun                checkpoint everything, resume bit-identically

tfs/train.py's behaviour is not changed by anything here.
"""

from __future__ import annotations

import math
import random
import time

import torch

from tfs.model import Config, Model
from tfs.train import lr_at, make_optimizer


# ============================================================ batch size

def _grad_sq_norm(model, data, B, T, gen):
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + T] for i in ix])
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(x, targets=x)
    loss.backward()
    return sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)


def grad_noise_scale(model, data, T, b_small, b_big, draws=8, seed=0):
    """
    The 'simple' gradient noise scale B_simple = tr(Sigma) / |G|^2.

    A batch-B gradient estimate satisfies E|G_B|^2 = |G|^2 + tr(Sigma)/B, so two
    batch sizes give unbiased estimates of both unknowns (McCandlish et al. 2018,
    Appendix A):
        |G|^2     = (B_big |G_big|^2 - B_small |G_small|^2) / (B_big - B_small)
        tr(Sigma) = (|G_small|^2 - |G_big|^2) / (1/B_small - 1/B_big)
    Returns (B_simple, |G|^2, tr(Sigma)) averaged over `draws` pairs of batches.
    B is counted in SEQUENCES of length T; multiply by T for tokens.
    """
    gen = torch.Generator().manual_seed(seed)
    gs = sum(_grad_sq_norm(model, data, b_small, T, gen) for _ in range(draws)) / draws
    gb = sum(_grad_sq_norm(model, data, b_big, T, gen) for _ in range(draws)) / draws
    G2 = (b_big * gb - b_small * gs) / (b_big - b_small)
    S = (gs - gb) / (1 / b_small - 1 / b_big)
    model.zero_grad(set_to_none=True)
    return S / G2, G2, S


def steps_and_examples(B, B_crit):
    """
    The trade-off curve (S/S_min - 1)(E/E_min - 1) = 1 with B = E/S and
    B_crit = E_min/S_min. Returns (S/S_min, E/E_min) at batch size B:
    at B = B_crit you pay 2x the minimum steps AND 2x the minimum data.
    """
    return 1 + B_crit / B, 1 + B / B_crit


# ============================================================ muP

def adam_output_change(width, lr, mup=False, base_width=256, seed=0):
    """
    One Adam step on a width x width linear layer, measuring how much its output
    moves per coordinate. Adam's first step is lr * sign(grad) on every entry, so
    the change in y = W x adds up `width` aligned terms: |dy| grows LINEARLY with
    width at a fixed learning rate. muP (Yang et al. 2022) scales the hidden-layer
    Adam LR by base_width / width, which makes |dy| width-independent — and that
    is why an LR tuned on a small model transfers to a big one.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(width, width, generator=g) / math.sqrt(width))
    x = torch.randn(64, width, generator=g)
    target = torch.randn(64, width, generator=g)
    eff_lr = lr * (base_width / width if mup else 1.0)
    opt = torch.optim.Adam([W], lr=eff_lr)
    y0 = (x @ W.T).detach()
    ((x @ W.T - target) ** 2).mean().backward()
    opt.step()
    return ((x @ W.T).detach() - y0).abs().mean().item()


# ============================================================ throughput

def flops_per_token(cfg: Config, T: int) -> int:
    """
    Training FLOPs per token (PaLM appendix B accounting):
        6 * N_matmul            forward (2) + backward (4) for every weight matmul,
                                including the LM head; embeddings are a lookup
        + 12 * L * H * dh * T   attention scores and the weighted sum of V, fwd+bwd,
                                counted without the causal discount
    """
    C, H, KV, dh = cfg.d_model, cfg.n_head, cfg.n_kv_head, cfg.d_head
    per_layer = C * H * dh * 2 + 2 * C * KV * dh + 3 * C * cfg.d_ff
    n_matmul = cfg.n_layer * per_layer + cfg.vocab_size * C          # + LM head
    return 6 * n_matmul + 12 * cfg.n_layer * H * dh * T


def measure_peak_matmul(n=2048, reps=5, dtype=torch.float32):
    """The best FLOP/s this machine reaches on one big square GEMM: our 'peak'."""
    a, b = torch.randn(n, n, dtype=dtype), torch.randn(n, n, dtype=dtype)
    a @ b
    best = 0.0
    for _ in range(reps):
        t = time.perf_counter()
        a @ b
        best = max(best, 2 * n ** 3 / (time.perf_counter() - t))
    return best


def measure_train_throughput(cfg: Config, B: int, T: int, steps=6, warm=2, seed=0):
    """Tokens/s of full training steps (forward, backward, AdamW) on the real Model."""
    torch.manual_seed(seed)
    model = Model(cfg)
    opt = make_optimizer(model, 1e-3)
    x = torch.randint(cfg.vocab_size, (B, T))
    times = []
    for i in range(warm + steps):
        t = time.perf_counter()
        _, loss, _ = model(x, targets=x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i >= warm:
            times.append(time.perf_counter() - t)
    return B * T / (sorted(times)[len(times) // 2])


def mfu(tokens_per_s, flops_tok, peak):
    return tokens_per_s * flops_tok / peak


# ============================================================ stability

def z_loss(logits, coef=1e-4):
    """PaLM's auxiliary loss coef * log(Z)^2, Z = sum(exp(logits)): keeps log Z near 0."""
    return coef * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


class SpikeDetector:
    """
    Flags a step whose loss exceeds the running mean of the last `window` steps by
    more than `k` running standard deviations. The policy that goes with it
    (PaLM, 2022): restart from a checkpoint ~100 steps before the spike and skip
    the ~200-500 batches around it.
    """

    def __init__(self, window=50, k=6.0):
        self.window, self.k, self.hist = window, k, []

    def update(self, loss: float) -> bool:
        h = self.hist[-self.window:]
        spike = False
        if len(h) >= 10:
            m = sum(h) / len(h)
            sd = (sum((x - m) ** 2 for x in h) / len(h)) ** 0.5
            spike = loss > m + self.k * max(sd, 1e-3)
        if not spike:
            self.hist.append(loss)
        return spike


# ============================================================ exact resumption

class Loader:
    """
    A data loader whose ENTIRE state is three integers: the seed, the epoch and
    the position in that epoch's permutation. Saving those three is enough to
    resume the exact same stream of batches.
    """

    def __init__(self, data, B, T, seed=0):
        self.data, self.B, self.T, self.seed = data, B, T, seed
        self.n = (len(data) - 1) // T
        self.epoch, self.pos = 0, 0
        self._perm()

    def _perm(self):
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        self.order = torch.randperm(self.n, generator=g)

    def next(self):
        if self.pos + self.B > self.n:
            self.epoch, self.pos = self.epoch + 1, 0
            self._perm()
        idx = self.order[self.pos:self.pos + self.B]
        self.pos += self.B
        return torch.stack([self.data[i * self.T:(i + 1) * self.T] for i in idx])

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, s):
        self.seed, self.epoch, self.pos = s["seed"], s["epoch"], s["pos"]
        self._perm()


class ResumableRun:
    """
    A training loop whose checkpoint captures everything that influences the
    next step: weights, optimizer moments AND its step counts, the step number
    (which sets the LR), the loader position, and every RNG (torch, python —
    dropout and any sampling draw from them). Resuming must be bit-identical
    to never having stopped; tests/test_training.py checks that it is.
    """

    def __init__(self, cfg, data, B, T, lr=1e-3, total=40, warmup=5, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        self.cfg, self.lr, self.total, self.warmup = cfg, lr, total, warmup
        self.model = Model(cfg)
        self.opt = make_optimizer(self.model, lr)
        self.loader = Loader(data, B, T, seed)
        self.step = 0

    def train(self, n):
        losses = []
        for _ in range(n):
            for g in self.opt.param_groups:
                g["lr"] = lr_at(self.step, self.lr, self.warmup, self.total)
            x = self.loader.next()
            _, loss, _ = self.model(x, targets=x)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.step += 1
            losses.append(loss.item())
        return losses

    def state_dict(self):
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "step": self.step, "loader": self.loader.state_dict(),
                "rng_torch": torch.get_rng_state(), "rng_python": random.getstate()}

    def load_state_dict(self, s):
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["opt"])
        self.step = s["step"]
        self.loader.load_state_dict(s["loader"])
        torch.set_rng_state(s["rng_torch"])
        random.setstate(s["rng_python"])


# ============================================================ the chapter's measurements

def main():
    """
    python -m tfs.trainkit   — reproduces Chapter 11's measured numbers on the
    Exercise 11.1 model (character-level Tiny Shakespeare, 2.4 M parameters):
    throughput and MFU, then the gradient noise scale as training proceeds.
    """
    import sys
    from tfs.tokenizer import CharTokenizer
    from tfs.train import get_corpus, make_dataset
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    text = get_corpus()
    tok = CharTokenizer(text)
    tr, _ = make_dataset(text, tok)
    cfg = Config(vocab_size=tok.vocab_size, d_model=192, n_layer=6, n_head=6, n_kv_head=2,
                 d_head=32, d_ff=512, max_T=256)
    B, T = 32, 128
    n = sum(p.numel() for p in Model(cfg).parameters())
    f = flops_per_token(cfg, T)
    peak = measure_peak_matmul()
    tps = measure_train_throughput(cfg, B, T)
    print(f"params {n:,}  flops/token {f:,}  (6N = {6 * n:,})")
    print(f"peak fp32 GEMM {peak / 1e9:.1f} GFLOP/s   train {tps:,.0f} tok/s   "
          f"achieved {tps * f / 1e9:.1f} GFLOP/s   MFU {mfu(tps, f, peak):.1%}")

    torch.manual_seed(1337)
    model = Model(cfg)
    opt = make_optimizer(model, 1e-3)
    g = torch.Generator().manual_seed(1)
    done = 0
    for target in (0, 100, 400, 1200):
        while done < target:
            for pg in opt.param_groups:
                pg["lr"] = lr_at(done, 1e-3, 200, 2500)
            ix = torch.randint(len(tr) - T - 1, (B,), generator=g)
            x = torch.stack([tr[i:i + T] for i in ix])
            _, loss, _ = model(x, targets=x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            done += 1
        bs, G2, S = grad_noise_scale(model, tr, T, 4, 64, draws=8, seed=target)
        print(f"step {target:5d}  B_simple = {bs:8.1f} sequences = {bs * T:9,.0f} tokens"
              f"   |G|^2 {G2:.3e}  tr(S) {S:.3e}")


if __name__ == "__main__":
    main()
