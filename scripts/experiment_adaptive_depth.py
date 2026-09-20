#!/usr/bin/env python
"""
Step 3 experiment: does routing extra depth to uncertain tokens beat
spending the same compute uniformly?

Teacher-forced on held-out text. For every token we record the loss at the
shallow depth and at the deep depth, plus the three signals read at the
shallow depth. A policy escalates the top ``f`` fraction of tokens by its
signal and takes the deep-depth loss for those tokens. Its compute is
reported under two cost models, each compared with the fixed-depth curve
interpolated (in log-depth) at the matching mean depth:

    resume  : escalated tokens continue the shallow pass, so the mean depth
              is (1 - f) * d_lo + f * d_hi. Exact for single-block models,
              a lower bound for stacks.
    restart : the shallow pass is thrown away and the token is rerun at
              d_hi, which is what AdaptiveGenerator does today, so the mean
              depth is d_lo + f * d_hi.

Negative delta = adaptive wins at that cost. A policy has to beat the
restart column to pay for itself as implemented; the resume column is
what a resumable implementation would get.

Policies: entropy, probe, step_kl, random (control, same fraction), and an
oracle that escalates the tokens with the largest actual gain (upper bound).

    python scripts/experiment_adaptive_depth.py --checkpoint outputs/phase2_final.pt \\
        --data_dir data/tinystories --depth_lo 2 --depth_hi 8 --fixed_depths 1 2 4 8 \\
        --fractions 0.1 0.25 0.5 --batches 50 --out results/adaptive.json

Caveat: in teacher-forced evaluation the deep pass also deepens the
context, whereas at generation time only escalated tokens are deep. This
measures the per-token benefit of depth, which is the quantity the
trigger has to predict.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lantern.data import MemmapDataset  # noqa: E402
from train import load_checkpoint  # noqa: E402


@torch.no_grad()
def collect(model, loader, depth_lo, depth_hi, fixed_depths, pause_hi, device, max_batches):
    """Per-token losses at each fixed depth and signals at depth_lo."""
    depths = sorted(set(fixed_depths) | {depth_lo, depth_hi})
    losses = {d: [] for d in depths}
    if pause_hi:
        losses[("hi_pause", pause_hi)] = []
    sig = {"entropy": [], "probe": [], "step_kl": []}

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        for d in depths:
            step_states = []
            logits, hidden, _ = model(
                x, steps_per_block=d, return_hidden_states=True, step_states=step_states
            )
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), y.view(-1), reduction="none"
            )
            losses[d].append(nll.cpu())
            if d == depth_lo:
                log_p = F.log_softmax(logits.float(), dim=-1)
                sig["entropy"].append((-(log_p.exp() * log_p).sum(-1)).reshape(-1).cpu())
                sig["probe"].append(model.probe_uncertainty(hidden.float()).reshape(-1).cpu())
                sig["step_kl"].append(model.step_kl(step_states, last_only=False).reshape(-1).cpu())
        if pause_hi:
            logits, _, _ = model(x, steps_per_block=depth_hi, pause_steps=pause_hi)
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), y.view(-1), reduction="none"
            )
            losses[("hi_pause", pause_hi)].append(nll.cpu())

    losses = {k: torch.cat(v).numpy() for k, v in losses.items()}
    sig = {k: torch.cat(v).numpy() for k, v in sig.items()}
    return losses, sig


def required_fixed_depths(fixed_depths, d_lo, d_hi, fractions, deep_cost=None):
    """
    The fixed depths to evaluate so that every matched-compute depth (under
    both cost models) is bracketed by the curve. Adds the integer floor /
    ceiling of the extreme mean depths when the requested list does not
    already cover them. ``deep_cost`` is the cost of an escalated token in
    depth units (d_hi plus any pause work); defaults to d_hi.
    """
    deep_cost = d_hi if deep_cost is None else deep_cost
    depths = {int(d) for d in fixed_depths} | {int(d_lo), int(d_hi)}
    lo = min((1 - f) * d_lo + f * deep_cost for f in fractions)
    hi = max(d_lo + f * deep_cost for f in fractions)
    if min(depths) > lo:
        depths.add(max(1, math.floor(lo)))
    if max(depths) < hi:
        depths.add(math.ceil(hi))
    return sorted(depths)


def interp_fixed(fixed_curve, mean_depth):
    """Linear interpolation of fixed-depth loss in log(depth); refuses to extrapolate."""
    ds = np.array(sorted(fixed_curve))
    if not ds.min() <= mean_depth <= ds.max():
        raise ValueError(
            f"mean depth {mean_depth:.2f} lies outside the evaluated fixed depths "
            f"{ds.tolist()}; add a bracketing depth to --fixed_depths"
        )
    ls = np.array([fixed_curve[d] for d in ds])
    return float(np.interp(math.log(mean_depth), np.log(ds), ls))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--depth_lo", type=int, default=None, help="default config.steps_base")
    ap.add_argument("--depth_hi", type=int, default=None, help="default config.steps_reasoning")
    ap.add_argument("--fixed_depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pause_hi", type=int, default=0,
                    help="Escalated tokens also get this many latent pause cycles (as AdaptiveGenerator "
                         "does with --pause_steps); charged as pause_hi/num_blocks depth units each")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--seq_length", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    model.to(args.device).eval()
    d_lo = args.depth_lo or model.config.steps_base
    d_hi = args.depth_hi or model.config.steps_reasoning
    seq_len = args.seq_length or min(512, model.config.max_position)
    ds = MemmapDataset(Path(args.data_dir) / "val.bin", seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # An escalated token costs d_hi recursion steps plus pause_hi pause cycles.
    # One pause cycle is one attention + FFN layer, i.e. 1/num_blocks of a
    # recursion step across the whole stack, so it is charged that way.
    num_blocks = model.config.num_blocks
    deep_cost = d_hi + args.pause_hi / num_blocks
    fixed_depths = required_fixed_depths(args.fixed_depths, d_lo, d_hi, args.fractions, deep_cost)
    extra = sorted(set(fixed_depths) - set(args.fixed_depths) - {d_lo, d_hi})
    if extra:
        print(f"note: also evaluating fixed depth(s) {extra} so every matched-compute "
              f"depth is bracketed by the fixed curve")
    losses, sig = collect(model, loader, d_lo, d_hi, fixed_depths, args.pause_hi,
                          args.device, args.batches)
    n_tokens = len(losses[d_lo])
    fixed_curve = {d: float(losses[d].mean()) for d in losses if isinstance(d, int)}

    print(f"tokens: {n_tokens:,}\n")
    print("fixed depth curve")
    for d in sorted(fixed_curve):
        print(f"  depth {d:>2}: loss {fixed_curve[d]:.4f}  ppl {math.exp(fixed_curve[d]):.2f}")
    # What an escalated token gets: depth d_hi, plus pause_hi pause cycles if asked.
    deep_loss = losses[("hi_pause", args.pause_hi)] if args.pause_hi else losses[d_hi]
    deep_label = f"depth {d_hi}" + (f" + {args.pause_hi} pause" if args.pause_hi else "")
    if args.pause_hi:
        lp = float(deep_loss.mean())
        print(f"  {deep_label}: loss {lp:.4f}  ppl {math.exp(lp):.2f}  "
              f"(escalated tokens use this; cost {deep_cost:.2f} depth units)")

    gain = losses[d_lo] - deep_loss  # positive = escalation helps this token
    print(f"\nper-token gain from depth {d_lo} -> {deep_label}: mean {gain.mean():+.4f}, "
          f"helps {100 * (gain > 0).mean():.1f}% of tokens")
    for name in ("entropy", "probe", "step_kl"):
        r = np.corrcoef(sig[name], gain)[0, 1]
        print(f"  corr({name}, gain) = {r:+.3f}")

    policies = dict(sig)
    policies["random"] = rng.random(n_tokens)
    policies["oracle"] = gain

    # Two cost models for the adaptive policy, both reported (deep_cost is
    # d_hi plus the pause work):
    #   resume : an escalated token continues the shallow pass.
    #            mean depth = (1-f)*d_lo + f*deep_cost. Exact for
    #            single-block models; a lower bound otherwise.
    #   restart: the shallow pass is discarded and the token is rerun,
    #            which is what AdaptiveGenerator does today.
    #            mean depth = d_lo + f*deep_cost.
    results = {"fixed_curve": fixed_curve, "d_lo": d_lo, "d_hi": d_hi, "pause_hi": args.pause_hi,
               "deep_cost": deep_cost, "n_tokens": n_tokens, "rows": []}
    print(f"\n{'policy':<10}{'frac':>6}{'adaptive':>10}"
          f"{'d_resume':>9}{'fixed':>8}{'delta':>8}"
          f"{'d_restart':>10}{'fixed':>8}{'delta':>8}")
    for f in args.fractions:
        k = int(round(f * n_tokens))
        depth_resume = (1 - f) * d_lo + f * deep_cost
        depth_restart = d_lo + f * deep_cost
        fixed_resume = interp_fixed(fixed_curve, depth_resume)
        fixed_restart = interp_fixed(fixed_curve, depth_restart)
        for name, values in policies.items():
            idx = np.argpartition(-values, k - 1)[:k] if k > 0 else np.array([], dtype=int)
            mask = np.zeros(n_tokens, dtype=bool)
            mask[idx] = True
            adaptive = float(np.where(mask, deep_loss, losses[d_lo]).mean())
            row = {
                "policy": name, "fraction": f, "adaptive_loss": adaptive,
                "mean_depth_resume": depth_resume, "fixed_loss_resume": fixed_resume,
                "delta_resume": adaptive - fixed_resume,
                "mean_depth_restart": depth_restart, "fixed_loss_restart": fixed_restart,
                "delta_restart": adaptive - fixed_restart,
            }
            results["rows"].append(row)
            print(f"{name:<10}{f:>6.2f}{adaptive:>10.4f}"
                  f"{depth_resume:>9.2f}{fixed_resume:>8.4f}{row['delta_resume']:>+8.4f}"
                  f"{depth_restart:>10.2f}{fixed_restart:>8.4f}{row['delta_restart']:>+8.4f}")
        print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
