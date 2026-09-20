#!/usr/bin/env python
"""
Generate text from a trained LANTERN checkpoint.

Fixed depth::

    python generate.py --checkpoint outputs/phase1_final.pt --prompt "Once upon a time" --depth 4

Uncertainty-triggered depth and latent pause. The threshold is calibrated on
validation data so that roughly ``--escalate_fraction`` of tokens escalate::

    python generate.py --checkpoint outputs/phase3_final.pt --prompt "Once upon a time" \\
        --signal step_kl --escalate_fraction 0.2 --pause_steps 2 --data_dir data/tinystories --trace
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lantern.controller.adaptive_generation import (
    AdaptiveGenerationConfig,
    AdaptiveGenerator,
    calibrate_threshold,
    collect_signals,
)
from lantern.data import MemmapDataset
from lantern.utils.bpe_tokenizer import BPETokenizer
from train import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (defaults to the one in the checkpoint)")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--depth", type=int, default=None, help="Recursion depth (config.steps_base if unset)")
    ap.add_argument("--deep_depth", type=int, default=None, help="Escalated depth (config.steps_reasoning)")
    ap.add_argument("--pause_steps", type=int, default=0, help="Latent pause cycles (every token at fixed depth, escalated tokens otherwise)")
    ap.add_argument("--signal", choices=["none", "entropy", "probe", "step_kl"], default="none")
    ap.add_argument("--threshold", type=float, default=None, help="Absolute threshold; overrides calibration")
    ap.add_argument("--escalate_fraction", type=float, default=0.2)
    ap.add_argument("--data_dir", default=None, help="Data dir with val.bin for calibration")
    ap.add_argument("--calib_batches", type=int, default=8)
    ap.add_argument("--trace", action="store_true", help="Print per-token signal and escalation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    model.to(args.device).eval()
    tok_path = args.tokenizer or ckpt.get("tokenizer_path")
    if not tok_path or not Path(tok_path).exists():
        raise SystemExit("Pass --tokenizer; the checkpoint does not point at an existing tokenizer.json")
    tokenizer = BPETokenizer.load(tok_path)
    eos = model.config.eos_token_id if model.config.eos_token_id is not None else tokenizer.eos_token_id

    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], device=args.device)

    if args.signal == "none":
        out = model.generate(
            input_ids, max_new_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, eos_token_id=eos,
            steps_per_block=args.depth, pause_steps=args.pause_steps,
        )
        print(tokenizer.decode(out[0].tolist()))
        return

    threshold = args.threshold
    if threshold is None:
        if not args.data_dir:
            raise SystemExit("Calibration needs --data_dir (or pass --threshold)")
        val = MemmapDataset(Path(args.data_dir) / "val.bin", seq_length=min(256, model.config.max_position))
        loader = DataLoader(val, batch_size=8, shuffle=False)
        signals = collect_signals(model, loader, steps=args.depth, device=torch.device(args.device),
                                  max_batches=args.calib_batches)
        threshold = calibrate_threshold(signals[args.signal], args.escalate_fraction)
        print(f"calibrated {args.signal} threshold = {threshold:.4f} "
              f"(escalates ~{args.escalate_fraction:.0%} of validation tokens)")

    cfg = AdaptiveGenerationConfig(
        max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, eos_token_id=eos, signal=args.signal, threshold=threshold,
        steps_base=args.depth, steps_deep=args.deep_depth, pause_steps=args.pause_steps,
    )
    result = AdaptiveGenerator(model, cfg).generate(input_ids)
    print(tokenizer.decode(result.tokens[0].tolist()))
    print(f"\nescalated {result.escalation_rate:.0%} of generated tokens")
    if args.trace:
        print(f"\n{'token':<16}{'signal':>10}{'esc':>5}{'depth':>7}{'pause':>7}")
        for t in result.trace[0]:
            text = tokenizer.decode([t.token_id], skip_special_tokens=False).replace("\n", "\\n")
            print(f"{text!r:<16}{t.signal:>10.4f}{'*' if t.escalated else '':>5}{t.depth:>7}{t.pause_steps:>7}")


if __name__ == "__main__":
    main()
