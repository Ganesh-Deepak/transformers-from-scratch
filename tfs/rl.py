"""
RL for language models — Chapters 23 and 25. Small, exact functions for the
quantities the chapters derive, so every number on the page can be recomputed.

    gae                    Ch 23  generalised advantage estimation, both forms
    kl_shaped_rewards      Ch 23  the per-token KL penalty as reward shaping
    forward_kl/reverse_kl  Ch 23  why on-policy distillation is mode-seeking
    rejection_yield        Ch 23  how many samples rejection-sampling FT needs
    grpo_advantages        Ch 25  z-scored (GRPO) and mean-only (Dr. GRPO)
    token_weights          Ch 25  how each loss normalisation weights a token
    gspo_ratio             Ch 25  GSPO's sequence-level importance ratio
    mc_step_values         Ch 25  Math-Shepherd-style process labels
    agent_loss_mask        Ch 25  only the policy's own tokens get gradient
"""

from __future__ import annotations

import torch


# ============================================================ Ch 23 — GAE

def gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 1.0, lam: float = 0.95):
    """
    rewards: (T,)    r_t for t = 0..T-1
    values:  (T+1,)  V(s_0) .. V(s_T); V(s_T) = 0 at the end of a completion
    returns (advantages (T,), returns (T,))

        delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma lam A_{t+1}          (the backward recursion)
                = sum_l (gamma lam)^l delta_{t+l}       (the definition)
    """
    T = rewards.shape[0]
    deltas = rewards + gamma * values[1:] - values[:-1]          # (T,)
    adv = torch.zeros(T, dtype=rewards.dtype)
    running = 0.0
    for t in reversed(range(T)):
        running = deltas[t] + gamma * lam * running
        adv[t] = running
    return adv, adv + values[:-1]                                # returns are the value targets


def gae_explicit(rewards, values, gamma=1.0, lam=0.95):
    """The definition, summed term by term. Used to test the recursion."""
    T = rewards.shape[0]
    deltas = rewards + gamma * values[1:] - values[:-1]
    return torch.stack([sum((gamma * lam) ** l * deltas[t + l] for l in range(T - t))
                        for t in range(T)])


def discounted_returns(rewards, gamma=1.0):
    out, g = torch.zeros_like(rewards), 0.0
    for t in reversed(range(rewards.shape[0])):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


# ============================================================ Ch 23 — KL as reward

def kl_shaped_rewards(logp: torch.Tensor, logp_ref: torch.Tensor, score: float, beta: float):
    """
    InstructGPT-style per-token reward. logp, logp_ref: (T,) log-probs of the
    tokens the policy actually sampled. The reward model's score arrives only
    on the last token; every token pays -beta * (log pi - log pi_ref).

    Summed over the completion this is score - beta * log(pi(y)/pi_ref(y)):
    the sequence-level objective, delivered one token at a time.
    """
    r = -beta * (logp - logp_ref)                                 # (T,)
    r = r.clone()
    r[-1] = r[-1] + score
    return r


# ============================================================ Ch 23 — distillation

def forward_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q): teacher p, student q. Off-policy distillation (SFT on teacher samples)."""
    m = p > 0
    return float((p[m] * (p[m].log() - q[m].log())).sum())


def reverse_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(q || p): what on-policy distillation minimises, on the student's own samples."""
    m = q > 0
    return float((q[m] * (q[m].log() - p[m].log())).sum())


def rejection_yield(p_correct: float, k: int) -> float:
    """P(at least one of k samples passes the filter) = 1 - (1 - p)^k."""
    return 1 - (1 - p_correct) ** k


# ============================================================ Ch 25 — GRPO family

def grpo_advantages(rewards, scale_by_std: bool = True, eps: float = 1e-4) -> torch.Tensor:
    """GRPO: (r - mean) / std. Dr. GRPO (Liu et al. 2025): r - mean, no std."""
    r = torch.as_tensor(rewards, dtype=torch.float64)
    a = r - r.mean()
    return a / (r.std(unbiased=False) + eps) if scale_by_std else a


def token_weights(advantages, lengths, mode: str, max_len: int | None = None) -> list:
    """
    The weight one token of completion i gets in the loss, by normalisation:
      'grpo'   : A_i / |o_i|, then averaged over the G completions   (per-sequence mean)
      'dapo'   : A_i / sum_j |o_j|                                    (token-level, per group)
      'dr_grpo': A_i / max_len                                        (a constant: no length bias)
    """
    G = len(lengths)
    if mode == "grpo":
        return [float(a) / (L * G) for a, L in zip(advantages, lengths)]
    if mode == "dapo":
        tot = sum(lengths)
        return [float(a) / tot for a in advantages]
    if mode == "dr_grpo":
        return [float(a) / (max_len * G) for a in advantages]
    raise ValueError(mode)


def gspo_ratio(logp: torch.Tensor, logp_old: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    GSPO (Zheng et al. 2025): one importance ratio per SEQUENCE, length-normalised,
        s_i = ( pi(y_i|x) / pi_old(y_i|x) ) ^ (1/|y_i|) = exp( mean_t log ratio_t ).
    logp, logp_old, mask: (G, T). Returns (G,).
    """
    lr = (logp - logp_old) * mask
    return torch.exp(lr.sum(-1) / mask.sum(-1))


def token_ratios(logp, logp_old):
    """GRPO / PPO: one ratio per token."""
    return torch.exp(logp - logp_old)


# ============================================================ Ch 25 — process rewards

def mc_step_values(step_successes: list[list[int]]) -> list[float]:
    """
    Math-Shepherd-style automatic process labels: for each prefix ending at step k,
    roll out N completions and record which reach the right answer. The value of
    the step is the success fraction.
    """
    return [sum(s) / len(s) for s in step_successes]


def step_advantages(values: list[float], v0: float) -> list[float]:
    """The advantage of a step is how much it changed the estimated chance of success."""
    prev, out = v0, []
    for v in values:
        out.append(v - prev)
        prev = v
    return out


# ============================================================ Ch 25 — agents

def agent_loss_mask(segments: list[tuple[str, int]]) -> torch.Tensor:
    """
    A multi-turn rollout is a sequence of segments: ('system'|'user'|'assistant'|'tool', n_tokens).
    Only 'assistant' tokens are actions of the policy; tool outputs and environment
    observations are NOT, and must be masked out of the policy-gradient loss.
    """
    return torch.cat([torch.full((n,), float(role == "assistant")) for role, n in segments])
