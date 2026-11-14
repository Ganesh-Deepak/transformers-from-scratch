"""
DPO — Chapter 24.

    L = -log sigmoid( beta * [ (logp_c - logp_c_ref) - (logp_r - logp_r_ref) ] )

Three details that decide whether this works (Ch 24.5):
  1. SUM log-probs over the response, do not average. (Averaging is SimPO.)
  2. Mask the prompt — only response tokens count.
  3. The reference model is frozen and evaluated under no_grad.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logp(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Summed log P(token) over positions where labels != -100.

    ids:    (B, T)  full sequence (prompt + response)
    labels: (B, T)  same, but prompt positions set to -100
    returns (B,)
    """
    logits, _, _ = model(ids)
    logits = logits[:, :-1]                       # (B, T-1, V)  drop last prediction
    labels = labels[:, 1:]                        # (B, T-1)     shift targets

    mask = labels != -100                         # (B, T-1)
    safe = labels.masked_fill(~mask, 0)           # gather needs valid indices

    logprobs = logits.log_softmax(-1)             # (B, T-1, V)
    per_token = logprobs.gather(2, safe.unsqueeze(2)).squeeze(2)   # (B, T-1)
    return (per_token * mask).sum(-1)             # (B,)


def dpo_loss(policy_chosen_logps: torch.Tensor,
             policy_rejected_logps: torch.Tensor,
             ref_chosen_logps: torch.Tensor,
             ref_rejected_logps: torch.Tensor,
             beta: float = 0.1):
    """
    All inputs are (B,) summed log-probs.

    Returns (loss, chosen_rewards, rejected_rewards, accuracy).
    The "rewards" are the implicit reward beta*log(pi/pi_ref) from Ch 24.3 —
    log them, they are the most informative diagnostic you have.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss, chosen_rewards, rejected_rewards, accuracy


def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               chosen_lens, rejected_lens, beta: float = 2.0, gamma: float = 0.5):
    """
    SimPO — Ch 25.1. Length-normalised, reference-free, with an explicit margin.
    logps are summed; we divide by length here to get the average.
    """
    logits = (beta * policy_chosen_logps / chosen_lens
              - beta * policy_rejected_logps / rejected_lens - gamma)
    return -F.logsigmoid(logits).mean()


def build_dpo_batch(tokenizer, prompt: str, chosen: str, rejected: str, device=None):
    """Tokenize one preference triple into (ids, labels) pairs with prompt masked."""
    p = tokenizer.encode(prompt)

    def pack(response: str):
        r = tokenizer.encode(response)
        ids = torch.tensor(p + r, device=device)
        labels = ids.clone()
        labels[: len(p)] = -100                   # Ch 23.2 — mask the prompt
        return ids, labels

    return pack(chosen), pack(rejected)


@torch.no_grad()
def precompute_reference_logps(ref_model, batches):
    """
    Ch 24.5: the reference never changes, so compute its log-probs ONCE for the
    whole dataset. Halves memory and time versus running it every step.
    """
    ref_model.eval()
    out = []
    for ids, labels in batches:
        out.append(sequence_logp(ref_model, ids, labels))
    return torch.cat(out)


# ============================================================ Ch 24.8 — online / iterative DPO
#
# A tabular "language model": one prompt, Y possible responses, a logit per
# response. Small enough that every probability is exact, rich enough to show
# the one thing that matters: offline DPO can only rearrange the responses its
# dataset contains, online DPO keeps finding better ones.

def bandit_logps(logits: torch.Tensor) -> torch.Tensor:
    return logits.log_softmax(-1)                                    # (Y,)


def bandit_expected_reward(logits: torch.Tensor, rewards: torch.Tensor) -> float:
    return float((logits.softmax(-1) * rewards).sum())


def bandit_dpo_grad(logits, ref_logps, pairs, beta):
    """
    The DPO gradient for a tabular policy, in closed form. Because
    d(log pi(w) - log pi(l))/d logits = e_w - e_l exactly (the softmax normaliser
    cancels, just as Z(x) does in 24.4), each pair moves ONLY its own two logits:
        grad = -(beta/N) * sum_i sigmoid(-beta m_i) (e_w - e_l)
    A response that appears in no pair gets exactly zero gradient.
    """
    lp = bandit_logps(logits)
    g = torch.zeros_like(logits)
    for w, l in pairs:
        m = (lp[w] - ref_logps[w]) - (lp[l] - ref_logps[l])
        c = beta * torch.sigmoid(-beta * m) / len(pairs)
        g[w] -= c
        g[l] += c
    return g


def _dpo_steps(logits, ref_logps, pairs, beta, lr, steps):
    """
    Full-batch gradient descent on the DPO loss over (chosen, rejected) ids, using
    bandit_dpo_grad's closed form in plain Python floats (torch per-element ops on an
    8-vector are ~50x slower than the arithmetic itself).
    """
    if not pairs:
        return logits
    import math
    th, ref = logits.tolist(), ref_logps.tolist()
    n = len(pairs)
    for _ in range(steps):
        lse = math.log(sum(math.exp(t) for t in th))
        lp = [t - lse for t in th]
        g = [0.0] * len(th)
        for w, l in pairs:
            m = (lp[w] - ref[w]) - (lp[l] - ref[l])
            c = beta / (1 + math.exp(beta * m)) / n          # beta * sigmoid(-beta m) / N
            g[w] -= c
            g[l] += c
        th = [t - lr * gi for t, gi in zip(th, g)]
    return torch.tensor(th, dtype=logits.dtype)


def _label(samples, rewards):
    """Best-vs-worst of a group, as a reward model or judge would label it."""
    best = max(samples, key=lambda y: rewards[y])
    worst = min(samples, key=lambda y: rewards[y])
    return (best, worst) if rewards[best] > rewards[worst] else None


def offline_dpo(ref_logits, rewards, rounds=20, k=8, beta=0.1, lr=1.0, steps=10, seed=0):
    """All data sampled ONCE from the reference policy, then trained on for the same compute."""
    g = torch.Generator().manual_seed(seed)
    ref_logps = bandit_logps(ref_logits)
    pairs = []
    for _ in range(rounds):
        s = torch.multinomial(ref_logits.softmax(-1), k, replacement=True, generator=g).tolist()
        p = _label(s, rewards)
        if p:
            pairs.append(p)
    logits = _dpo_steps(ref_logits.clone(), ref_logps, pairs, beta, lr, steps * rounds)
    return logits, pairs


def online_dpo(ref_logits, rewards, rounds=20, k=8, beta=0.1, lr=1.0, steps=10, seed=0):
    """
    Iterative DPO: each round samples k responses from the CURRENT policy, labels the
    best and worst, adds the pair, and trains. Same number of samples and steps as
    offline_dpo. Returns (final logits, pairs, expected reward after each round).
    """
    g = torch.Generator().manual_seed(seed)
    ref_logps = bandit_logps(ref_logits)
    logits = ref_logits.clone()
    pairs, curve = [], []
    for _ in range(rounds):
        s = torch.multinomial(logits.softmax(-1), k, replacement=True, generator=g).tolist()
        p = _label(s, rewards)
        if p:
            pairs.append(p)
        logits = _dpo_steps(logits, ref_logps, pairs, beta, lr, steps)
        curve.append(bandit_expected_reward(logits, rewards))
    return logits, pairs, curve
