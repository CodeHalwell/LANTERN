#!/usr/bin/env python
"""
Step 1 experiment: does recursion depth help at all?

Evaluates one checkpoint at several recursion depths, with the same weights,
and reports validation loss and perplexity per depth. Optionally adds latent
pause steps at each depth (only meaningful after Phase 3).

    python scripts/eval_depth.py --checkpoint outputs/phase1_final.pt --data_dir data/tinystories \\
        --depths 1 2 4 8 --batches 50
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lantern.data import MemmapDataset  # noqa: E402
from train import load_checkpoint  # noqa: E402


@torch.no_grad()
def loss_at_depth(model, loader, depth, pause_steps, device, max_batches):
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits, _, _ = model(x, steps_per_block=depth, pause_steps=pause_steps)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1)).item()
        n += 1
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pause_steps", type=int, nargs="+", default=[0])
    ap.add_argument("--seq_length", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--out", default=None, help="Write results as JSON")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    model.to(args.device).eval()
    seq_len = args.seq_length or min(512, model.config.max_position)
    ds = MemmapDataset(Path(args.data_dir) / f"{args.split}.bin", seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    max_depth = max(args.depths)
    if max_depth > model.config.max_steps:
        print(f"note: depths above max_steps={model.config.max_steps} reuse the last step embedding")

    rows = []
    print(f"{'depth':>6}{'pause':>7}{'loss':>10}{'ppl':>10}")
    for d in args.depths:
        for p in args.pause_steps:
            loss = loss_at_depth(model, loader, d, p, args.device, args.batches)
            rows.append({"depth": d, "pause_steps": p, "loss": loss, "ppl": math.exp(loss)})
            print(f"{d:>6}{p:>7}{loss:>10.4f}{math.exp(loss):>10.2f}", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"checkpoint": args.checkpoint, "phase": ckpt.get("phase"), "rows": rows}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
