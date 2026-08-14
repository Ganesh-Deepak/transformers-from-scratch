"""
DPO — Chapter 21.

    L = -log sigmoid( beta * [ (logp_c - logp_c_ref) - (logp_r - logp_r_ref) ] )

Three details that decide whether this works (Ch 21.5):
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
    The "rewards" are the implicit reward beta*log(pi/pi_ref) from Ch 21.3 —
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
    SimPO — Ch 22.1. Length-normalised, reference-free, with an explicit margin.
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
        labels[: len(p)] = -100                   # Ch 20.2 — mask the prompt
        return ids, labels

    return pack(chosen), pack(rejected)


@torch.no_grad()
def precompute_reference_logps(ref_model, batches):
    """
    Ch 21.5: the reference never changes, so compute its log-probs ONCE for the
    whole dataset. Halves memory and time versus running it every step.
    """
    ref_model.eval()
    out = []
    for ids, labels in batches:
        out.append(sequence_logp(ref_model, ids, labels))
    return torch.cat(out)
