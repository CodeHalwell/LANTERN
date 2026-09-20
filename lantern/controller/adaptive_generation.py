"""
Adaptive generation for LANTERN.

The generation loop the design describes, wired to the real model:

    for each new token:
        1. forward at steps_base through the KV cache
        2. read a per-token signal at the last position
              entropy   : H[p]                          (free)
              probe     : epistemic probe on h           (free, needs Phase 2)
              step_kl   : KL(p_T || p_{T-1}) over the last two recursion
                          steps of the final block       (free)
              none      : never escalate
        3. if signal > threshold: escalate
              rewind the cache to before this token and run again at
              steps_deep, optionally with latent pause steps
        4. sample from whichever logits were produced last

Thresholds are absolute. Use ``calibrate_threshold`` on held-out text to
pick one that escalates a chosen fraction of tokens; that turns the
"always escalate the top 10%" behaviour of the EMA controller into a fixed
operating point that can be compared across runs.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from lantern.models.lantern_model import LANTERNModel, sample_from_logits

SIGNALS = ("entropy", "probe", "step_kl", "none")


@dataclass
class AdaptiveGenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    eos_token_id: Optional[int] = None

    signal: str = "entropy"
    threshold: float = float("inf")  # never escalate until calibrated

    steps_base: Optional[int] = None  # config.steps_base if None
    steps_deep: Optional[int] = None  # config.steps_reasoning if None
    pause_steps: int = 0              # latent pause cycles when escalating


@dataclass
class TokenTrace:
    token_id: int
    signal: float
    escalated: bool
    depth: int
    pause_steps: int
    entropy: float
    probe: Optional[float] = None
    step_kl: Optional[float] = None


@dataclass
class AdaptiveGenerationResult:
    tokens: torch.Tensor                      # [batch, prompt + generated]
    trace: List[List[TokenTrace]] = field(default_factory=list)  # per batch row

    @property
    def escalation_rate(self) -> float:
        n = sum(len(t) for t in self.trace)
        if n == 0:
            return 0.0
        return sum(tt.escalated for t in self.trace for tt in t) / n


def compute_signals(
    model: LANTERNModel,
    logits_last: torch.Tensor,
    hidden_last: torch.Tensor,
    step_states: List[torch.Tensor],
) -> dict:
    """All three signals at the last position -> dict of [batch] tensors."""
    log_p = F.log_softmax(logits_last.float(), dim=-1)
    entropy = -(log_p.exp() * log_p).sum(-1)
    probe = model.probe_uncertainty(hidden_last.float()) if hidden_last is not None else None
    step_kl = model.step_kl(step_states) if step_states else None
    return {"entropy": entropy, "probe": probe, "step_kl": step_kl}


class AdaptiveGenerator:
    """Uncertainty-triggered depth and latent pause at generation time."""

    def __init__(self, model: LANTERNModel, config: Optional[AdaptiveGenerationConfig] = None):
        if config is not None and config.signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}")
        if config is not None and config.pause_steps < 0:
            raise ValueError("pause_steps must be >= 0")
        self.model = model
        self.config = config or AdaptiveGenerationConfig()
        # The pause module clamps to max_pause_steps; record what actually runs.
        self.effective_pause_steps = min(self.config.pause_steps, model.config.max_pause_steps)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> AdaptiveGenerationResult:
        model, cfg = self.model, self.config
        model.eval()
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        steps_base = cfg.steps_base or model.config.steps_base
        steps_deep = cfg.steps_deep or model.config.steps_reasoning

        budget = min(cfg.max_new_tokens, model.config.max_position - prompt_len)
        result = AdaptiveGenerationResult(tokens=input_ids, trace=[[] for _ in range(batch_size)])
        if budget <= 0:
            return result

        caches = model.create_kv_caches(
            batch_size, prompt_len + budget, device, dtype=model.token_embedding.weight.dtype
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        cur_input, start_pos = input_ids, 0
        generated = input_ids
        for _ in range(budget):
            step_states: List[torch.Tensor] = []
            logits, hidden, _ = model(
                cur_input, steps_per_block=steps_base, return_hidden_states=True,
                step_states=step_states, kv_caches=caches, start_pos=start_pos,
            )
            logits_last, hidden_last = logits[:, -1, :], hidden[:, -1, :]
            signals = compute_signals(model, logits_last, hidden_last, step_states)
            signal = signals[cfg.signal] if cfg.signal != "none" else None

            escalate = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if signal is not None:
                escalate = signal > cfg.threshold
            depth_used = torch.full((batch_size,), steps_base, device=device)
            pause_used = torch.zeros(batch_size, dtype=torch.long, device=device)

            if bool(escalate.any()):
                # Rewind the cache for these positions and recompute deeper.
                # The whole batch is recomputed together, so snapshot the
                # shallow cache entries and restore them for rows that did
                # not escalate; those rows keep their shallow logits too.
                end = start_pos + cur_input.shape[1]
                keep = ~escalate
                snapshot = [
                    (c.k_cache[:, keep, :, start_pos:end].clone(),
                     c.v_cache[:, keep, :, start_pos:end].clone())
                    for c in caches
                ]
                for c in caches:
                    c.truncate(start_pos)
                deep_logits, _, _ = model(
                    cur_input, steps_per_block=steps_deep, pause_steps=cfg.pause_steps,
                    kv_caches=caches, start_pos=start_pos,
                )
                for c, (k_snap, v_snap) in zip(caches, snapshot):
                    c.k_cache[:, keep, :, start_pos:end] = k_snap
                    c.v_cache[:, keep, :, start_pos:end] = v_snap
                logits_last = torch.where(
                    escalate.unsqueeze(-1), deep_logits[:, -1, :], logits_last
                )
                depth_used = torch.where(escalate, torch.full_like(depth_used, steps_deep), depth_used)
                pause_used = torch.where(
                    escalate, torch.full_like(pause_used, self.effective_pause_steps), pause_used
                )

            next_token = sample_from_logits(
                logits_last, temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p
            )
            if cfg.eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1), torch.full_like(next_token, cfg.eos_token_id), next_token
                )

            for b in range(batch_size):
                if finished[b]:
                    continue
                result.trace[b].append(TokenTrace(
                    token_id=int(next_token[b, 0]),
                    signal=float(signal[b]) if signal is not None else 0.0,
                    escalated=bool(escalate[b]),
                    depth=int(depth_used[b]),
                    pause_steps=int(pause_used[b]),
                    entropy=float(signals["entropy"][b]),
                    probe=float(signals["probe"][b]) if signals["probe"] is not None else None,
                    step_kl=float(signals["step_kl"][b]) if signals["step_kl"] is not None else None,
                ))

            if cfg.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == cfg.eos_token_id)

            generated = torch.cat([generated, next_token], dim=1)
            start_pos = generated.shape[1] - 1
            cur_input = next_token
            if cfg.eos_token_id is not None and bool(finished.all()):
                break

        result.tokens = generated
        return result


@torch.no_grad()
def collect_signals(
    model: LANTERNModel,
    batches,
    steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Teacher-forced signals over held-out batches -> dict of flat CPU tensors
    (entropy, probe, step_kl), one value per token.
    """
    model.eval()
    steps = steps or model.config.steps_base
    out = {"entropy": [], "probe": [], "step_kl": []}
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"]
        if device is not None:
            input_ids = input_ids.to(device)
        step_states: List[torch.Tensor] = []
        logits, hidden, _ = model(
            input_ids, steps_per_block=steps, return_hidden_states=True, step_states=step_states
        )
        log_p = F.log_softmax(logits.float(), dim=-1)
        out["entropy"].append((-(log_p.exp() * log_p).sum(-1)).reshape(-1).cpu())
        out["probe"].append(model.probe_uncertainty(hidden.float()).reshape(-1).cpu())
        out["step_kl"].append(model.step_kl(step_states, last_only=False).reshape(-1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def calibrate_threshold(signal_values: torch.Tensor, escalate_fraction: float) -> float:
    """Threshold above which ``escalate_fraction`` of the observed values fall."""
    if not 0.0 < escalate_fraction < 1.0:
        raise ValueError("escalate_fraction must be in (0, 1)")
    return float(torch.quantile(signal_values.float(), 1.0 - escalate_fraction))
