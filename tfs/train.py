"""
Training loop — Chapter 11.

    python -m tfs.train                      # trains a ~3M model on Tiny Shakespeare
    python -m tfs.train --steps 2000 --d_model 256

Everything here runs on CPU. Expect ~5-15 minutes for the default run.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfs.model import Config, Model
from tfs.tokenizer import BPETokenizer, CharTokenizer

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ------------------------------------------------------------------ data

def get_corpus() -> str:
    """Download Tiny Shakespeare, or fall back to a bundled synthetic corpus."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    try:
        print("downloading Tiny Shakespeare ...")
        urllib.request.urlretrieve(DATA_URL, path)
        return open(path, encoding="utf-8").read()
    except Exception as e:
        print(f"  download failed ({e}); using the bundled fallback corpus.")
        return _fallback_corpus()


def _fallback_corpus() -> str:
    """Offline fallback: enough structure that a tiny model can learn something."""
    import random
    rng = random.Random(0)
    names = ["ALICE", "BERTRAM", "CORDELIA", "DUNCAN", "EDMUND", "FLORIZEL"]
    verbs = ["speaks", "waits", "departs", "returns", "considers", "answers"]
    nouns = ["the crown", "the letter", "the garden", "the morning", "her brother",
             "the ancient oath", "a quiet word", "the king's men"]
    advs = ["softly", "at once", "in silence", "before dawn", "with great care"]
    out = []
    for _ in range(4000):
        who = rng.choice(names)
        out.append(f"{who}:\n")
        for _ in range(rng.randint(1, 3)):
            out.append(f"  {rng.choice(names).title()} {rng.choice(verbs)} of "
                       f"{rng.choice(nouns)} {rng.choice(advs)}.\n")
        out.append("\n")
    return "".join(out)


def make_dataset(text: str, tok: BPETokenizer, val_frac: float = 0.1):
    ids = torch.tensor(tok.encode(text, allow_special=False), dtype=torch.long)
    n = int(len(ids) * (1 - val_frac))
    return ids[:n], ids[n:]


def get_batch(data: torch.Tensor, B: int, T: int, device="cpu"):
    """
    Chapter 11.5 — packed sampling, no padding.

    Returns ONE tensor (B, T). Do NOT also shift it here.

    `Model.forward(ids, targets=...)` performs the next-token shift internally
    (`logits[:, :-1]` against `targets[:, 1:]`), so you pass the SAME sequence as
    both input and targets. Passing a pre-shifted `y` shifts twice and silently
    trains the model to predict TWO tokens ahead -- the loss curve looks fine,
    teacher-forced output is off by one, and free-running generation is garbage.
    See `test_generation_reproduces_memorised_text`.
    """
    ix = torch.randint(len(data) - T - 1, (B,))
    return torch.stack([data[i:i + T] for i in ix]).to(device)


# ------------------------------------------------------- lr schedule (Ch 11.3)

def lr_at(step, base_lr, warmup, total, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return base_lr * min_ratio
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


# ------------------------------------------------------------- optimizer

def make_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    """Ch 11.2 — decay 2-D weights only; never biases or norm gains."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=1e-8)


# -------------------------------------------------------------------- eval

@torch.no_grad()
def estimate_loss(model, data, B, T, iters=20, device="cpu"):
    model.eval()
    losses = []
    for _ in range(iters):
        x = get_batch(data, B, T, device)
        _, loss, _ = model(x, targets=x)      # forward does the shift internally
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def bits_per_byte(loss_nats, bytes_per_token):
    """
    THE tokenizer-independent quality metric.

    Cross-entropy in nats/token is NOT comparable across tokenizers: a token that
    carries 4 characters should cost more than one carrying 1. Dividing by the
    bytes each token represents (and converting nats -> bits) normalises that away.

        char-level Shakespeare, a good small model  ~2.1 bits/byte
        GPT-2 on web text                           ~1.0 bits/byte
        strong modern LLMs                          ~0.6-0.8 bits/byte

    Use this whenever you compare runs with different vocab sizes.
    """
    return loss_nats / math.log(2) / max(bytes_per_token, 1e-9)


# ------------------------------------------------------------------- train

def train(cfg: Config, train_data, val_data, steps=3000, B=16, T=128,
          lr=3e-4, warmup=200, eval_every=250, device="cpu", log=print,
          bytes_per_token=1.0):
    torch.manual_seed(1337)
    model = Model(cfg).to(device)
    n = sum(p.numel() for p in model.parameters())
    log(f"model: {n:,} parameters  ({n / 1e6:.2f} M)")
    log(f"expected initial loss = ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}")

    # Ch 21: a rough tokens-per-parameter check. Far below ~20 means you will
    # overfit long before you converge.
    tok_per_param = len(train_data) / n
    log(f"train tokens/param = {tok_per_param:.2f}"
        + ("   <- VERY LOW: expect overfitting (see Ch 21)" if tok_per_param < 5 else ""))

    opt = make_optimizer(model, lr)
    history = []
    best = (float("inf"), -1)
    best_state = None
    t0 = time.time()

    for step in range(steps):
        cur_lr = lr_at(step, lr, warmup, steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x = get_batch(train_data, B, T, device)
        _, loss, _ = model(x, targets=x)      # forward does the shift internally

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            vl = estimate_loss(model, val_data, B, T, device=device)
            history.append((step, loss.item(), vl))
            flag = ""
            if vl < best[0]:
                best = (vl, step)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                flag = " *best"
            elif loss.item() < vl - 0.25:
                flag = " <- overfitting (train << val)"
            log(f"step {step:5d}  train {loss.item():.4f}  val {vl:.4f}  "
                f"bpb {bits_per_byte(vl, bytes_per_token):.3f}  "
                f"lr {cur_lr:.2e}  gnorm {gnorm:.2f}  {time.time() - t0:5.0f}s{flag}")

    if best_state is not None and best[1] != history[-1][0]:
        log(f"\nrestoring best checkpoint from step {best[1]} (val {best[0]:.4f}); "
            f"the final step had val {history[-1][2]:.4f}")
        model.load_state_dict(best_state)

    return model, history


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_kv_head", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--char", action="store_true",
                    help="character-level tokenizer instead of BPE. This is what "
                         "Chapter 11 Exercise 11.1 uses; target val loss ~1.5.")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    text = get_corpus()
    print(f"corpus: {len(text):,} characters")

    if args.char:
        tok = CharTokenizer(text)
        print(f"tokenizer: character-level, {tok.vocab_size} tokens, "
              f"1.00 bytes/token")
    else:
        tok_path = os.path.join(DATA_DIR, f"bpe_{args.vocab}.json")
        if os.path.exists(tok_path):
            tok = BPETokenizer.load(tok_path)
        else:
            print(f"training BPE tokenizer (vocab {args.vocab}) ...")
            tok = BPETokenizer().train(text[:400_000], vocab_size=args.vocab)
            tok.save(tok_path)
        print(f"tokenizer: {tok.vocab_size} tokens, "
              f"{tok.compression_ratio(text[:20000]):.2f} bytes/token")
    bpt = 1.0 if args.char else tok.compression_ratio(text[:20000])
    print(f"\nNOTE: loss is in nats PER TOKEN and is NOT comparable across vocab\n"
          f"      sizes. Compare runs using bits-per-byte (bpb) instead.\n")

    tr, va = make_dataset(text, tok)
    print(f"train {len(tr):,} tokens   val {len(va):,} tokens")

    cfg = Config(
        vocab_size=tok.vocab_size, d_model=args.d_model, n_layer=args.n_layer,
        n_head=args.n_head, n_kv_head=args.n_kv_head,
        d_head=args.d_model // args.n_head,
        d_ff=256 * round(int(8 * args.d_model / 3) / 256) or 256,
        max_T=max(args.seq, 256),
    )
    model, _ = train(cfg, tr, va, steps=args.steps, B=args.batch, T=args.seq,
                     lr=args.lr, bytes_per_token=bpt)

    print("\n--- sample (temperature 0.8, top-p 0.95) ---")
    start = torch.tensor([tok.encode("\n", allow_special=False)], dtype=torch.long)
    out = model.generate(start, max_new_tokens=300, temperature=0.8, top_p=0.95)
    print(tok.decode(out[0].tolist()))

    if args.save:
        torch.save({"cfg": cfg, "state": model.state_dict()}, args.save)
        print(f"\nsaved to {args.save}")


if __name__ == "__main__":
    main()
