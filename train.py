#!/usr/bin/env python
"""
Train LANTERN with the three-phase curriculum.

Phase 1  backbone pretraining, random recursion depth per step
Phase 2  freeze backbone, distil MC-dropout variance into the epistemic probe
Phase 3  unfreeze, enable ACT halting + ponder cost, train the latent pause module

Data comes from a directory made by ``scripts/prepare_data.py`` (train.bin,
val.bin, tokenizer.json, meta.json), or from a plain text file with a BPE
tokenizer trained on the fly (small experiments only).

Examples::

    # Tiny CPU smoke test on the bundled example text
    python train.py --data_path example_data.txt --config small --vocab_size 512 \\
        --seq_length 64 --phase all --phase1_steps 20 --phase2_steps 5 --phase3_steps 10

    # TinyStories, all phases, one GPU
    python train.py --data_dir data/tinystories --config base --phase all \\
        --phase1_steps 20000 --phase2_steps 2000 --phase3_steps 5000 --batch_size 32

    # 300M model on FineWeb-Edu, phase 1 only, resume later for phases 2 and 3
    python train.py --data_dir data/fineweb-edu --config 300m --phase 1 \\
        --phase1_steps 60000 --batch_size 8 --grad_accum 8 --seq_length 1024 --compile
"""

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from lantern.data import MemmapDataset, TokenizedTextDataset, load_data_dir_meta
from lantern.models.lantern_model import LANTERNModel
from lantern.training import Phase1Trainer, Phase2Trainer, Phase3Trainer
from lantern.utils.bpe_tokenizer import BPETokenizer
from lantern.utils.config import (
    LANTERNConfig,
    create_300m_config,
    create_base_config,
    create_small_config,
    create_tiny_lantern_config,
)

CONFIGS = {
    "small": create_small_config,
    "tiny": create_tiny_lantern_config,
    "base": create_base_config,
    "300m": create_300m_config,
}


# ----------------------------------------------------------------- checkpoints
def save_checkpoint(model: LANTERNModel, path: Path, phase: int, step: int,
                    tokenizer_path: Optional[str], extra: Optional[dict] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    payload = {
        "model_state_dict": raw.state_dict(),
        "config": asdict(raw.config),
        "phase": phase,
        "step": step,
        "tokenizer_path": tokenizer_path,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  saved {path}")


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg.get("global_token_indices"), list):
        cfg["global_token_indices"] = set(cfg["global_token_indices"])
    config = LANTERNConfig(**cfg)
    model = LANTERNModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


# ----------------------------------------------------------------- data
def build_datasets(args, output_dir: Path):
    """Returns (train_ds, val_ds, vocab_size, tokenizer_path, eos_id)."""
    if args.data_dir:
        d = Path(args.data_dir)
        meta = load_data_dir_meta(d)
        train_ds = MemmapDataset(d / "train.bin", args.seq_length)
        val_ds = MemmapDataset(d / "val.bin", args.seq_length) if (d / "val.bin").exists() else None
        return train_ds, val_ds, meta["vocab_size"], str(d / "tokenizer.json"), meta["eos_token_id"]

    if not args.data_path:
        raise SystemExit("Pass --data_dir (from scripts/prepare_data.py) or --data_path")

    if args.tokenizer:
        tokenizer = BPETokenizer.load(args.tokenizer)
        tok_path = args.tokenizer
    else:
        print(f"Training a {args.vocab_size}-token BPE tokenizer on {args.data_path} ...")
        tokenizer = BPETokenizer.train_from_files([args.data_path], vocab_size=args.vocab_size)
        tok_path = str(output_dir / "tokenizer.json")
        tokenizer.save(tok_path)
    train_ds = TokenizedTextDataset(args.data_path, tokenizer, args.seq_length)
    val_ds = TokenizedTextDataset(args.val_data_path, tokenizer, args.seq_length) if args.val_data_path else None
    return train_ds, val_ds, tokenizer.vocab_size, tok_path, tokenizer.eos_token_id


def make_loader(ds: Optional[Dataset], batch_size: int, shuffle: bool, num_workers: int, device: str):
    if ds is None:
        return None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=device.startswith("cuda"), drop_last=True,
    )


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# ----------------------------------------------------------------- phases
class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, **kv):
        with open(self.path, "a") as fh:
            fh.write(json.dumps(kv) + "\n")


def run_phase(phase: int, trainer, train_loader, steps: int, args, output_dir: Path,
              tokenizer_path: str, log: Logger, val_loader=None):
    model = trainer.model
    model.train()
    batches = infinite(train_loader)
    t0 = time.time()
    best_val = math.inf
    for step in range(1, steps + 1):
        batch = next(batches)
        out = trainer.train_step(batch)
        metrics = out if isinstance(out, dict) else {"loss": out}

        if step % args.log_interval == 0 or step == steps:
            lr = trainer.optimizer.param_groups[0]["lr"]
            msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            print(f"[phase {phase}] step {step}/{steps} {msg} lr={lr:.2e} {time.time() - t0:.0f}s", flush=True)
            log(phase=phase, step=step, lr=lr, elapsed=time.time() - t0, **metrics)

        if val_loader is not None and (step % args.eval_interval == 0 or step == steps):
            val = evaluate(model, val_loader, args.device, args.eval_batches,
                           use_halting=(phase == 3))
            print(f"[phase {phase}] step {step} val_loss={val:.4f} ppl={math.exp(val):.2f}", flush=True)
            log(phase=phase, step=step, val_loss=val)
            if val < best_val:
                best_val = val
                save_checkpoint(model, output_dir / f"phase{phase}_best.pt", phase, step, tokenizer_path)

        if step % args.save_interval == 0 and step != steps:
            save_checkpoint(model, output_dir / f"phase{phase}_step{step}.pt", phase, step, tokenizer_path)

    save_checkpoint(model, output_dir / f"phase{phase}_final.pt", phase, steps, tokenizer_path)


@torch.no_grad()
def evaluate(model, loader, device, max_batches, use_halting=False, depth=None):
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
        with ctx:
            logits, _, _ = model(x, steps_per_block=depth, use_adaptive_halting=use_halting)
        total += torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), y.view(-1)
        ).item()
        n += 1
    model.train(was_training)
    return total / max(1, n)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    ap.add_argument("--data_dir", type=str, default=None, help="Directory from scripts/prepare_data.py")
    ap.add_argument("--data_path", type=str, default=None, help="Plain text file (one doc per line)")
    ap.add_argument("--val_data_path", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default=None, help="tokenizer.json to use with --data_path")
    ap.add_argument("--vocab_size", type=int, default=8192, help="Only used when training a tokenizer")
    ap.add_argument("--seq_length", type=int, default=512)
    # model
    ap.add_argument("--config", choices=list(CONFIGS), default="small")
    ap.add_argument("--hidden_size", type=int, default=None)
    ap.add_argument("--num_heads", type=int, default=None)
    ap.add_argument("--num_blocks", type=int, default=None)
    ap.add_argument("--window_size", type=int, default=None)
    ap.add_argument("--attn_impl", choices=["eager", "sdpa", "flex"], default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--resume_from", type=str, default=None, help="Checkpoint to start from")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    # schedule
    ap.add_argument("--phase", choices=["1", "2", "3", "all"], default="1")
    ap.add_argument("--phase1_steps", type=int, default=10000)
    ap.add_argument("--phase2_steps", type=int, default=2000)
    ap.add_argument("--phase3_steps", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=None, help="Alias for --phase1_steps")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--mc_samples", type=int, default=5, help="Phase 2 MC-dropout samples")
    ap.add_argument("--ponder_lambda", type=float, default=0.01)
    ap.add_argument("--backbone_lr", type=float, default=1e-5, help="Phase 3 backbone LR")
    ap.add_argument("--reasoning_lr", type=float, default=1e-3, help="Phase 3 heads LR")
    # io
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_steps is not None:
        args.phase1_steps = args.max_steps
    use_bf16 = not args.no_bf16

    train_ds, val_ds, vocab_size, tokenizer_path, eos_id = build_datasets(args, output_dir)
    print(f"train windows: {len(train_ds):,}  val windows: {len(val_ds) if val_ds else 0:,}  vocab: {vocab_size}")

    if args.resume_from:
        model, ckpt = load_checkpoint(args.resume_from, args.device)
        print(f"Resumed {args.resume_from} (phase {ckpt.get('phase')}, step {ckpt.get('step')})")
    else:
        config = CONFIGS[args.config]()
        config.vocab_size = vocab_size
        config.eos_token_id = eos_id
        for name in ("hidden_size", "num_heads", "num_blocks", "window_size", "attn_impl", "dropout"):
            v = getattr(args, name)
            if v is not None:
                setattr(config, name, v)
        if config.max_position < args.seq_length:
            config.max_position = args.seq_length
        if args.phase in ("3", "all"):
            config.use_adaptive_halting = True
        model = LANTERNModel(config)
    model.to(args.device)
    c = model.config
    print(f"model: {c.hidden_size}d x {c.num_blocks} blocks, steps {c.steps_base}/{c.steps_reasoning}/{c.max_steps}, "
          f"window {c.window_size}, attn {c.attn_impl}, params {model.get_num_params(non_embedding=False) / 1e6:.1f}M "
          f"({model.get_num_params() / 1e6:.1f}M non-embedding)")
    if args.compile:
        model = torch.compile(model)

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, args.device)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, args.device)
    log = Logger(output_dir / "training_log.jsonl")
    with open(output_dir / "args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    phases = [1, 2, 3] if args.phase == "all" else [int(args.phase)]
    for phase in phases:
        print("=" * 70 + f"\nPhase {phase}\n" + "=" * 70)
        if phase == 1:
            trainer = Phase1Trainer(
                model, train_loader, val_loader,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps, max_steps=args.phase1_steps,
                grad_clip=args.grad_clip, device=args.device,
                use_bfloat16=use_bf16, grad_accum_steps=args.grad_accum,
            )
            run_phase(1, trainer, train_loader, args.phase1_steps, args, output_dir, tokenizer_path, log, val_loader)
        elif phase == 2:
            trainer = Phase2Trainer(
                model, train_loader, num_mc_samples=args.mc_samples,
                learning_rate=1e-3, max_steps=args.phase2_steps, device=args.device,
            )
            run_phase(2, trainer, train_loader, args.phase2_steps, args, output_dir, tokenizer_path, log)
            trainer.cleanup()
        else:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            if not raw.config.use_adaptive_halting:
                raise SystemExit("Phase 3 needs use_adaptive_halting=True; start the run with --phase all "
                                 "or a config that enables halting.")
            trainer = Phase3Trainer(
                model, train_loader, val_loader,
                backbone_lr=args.backbone_lr, reasoning_lr=args.reasoning_lr,
                ponder_lambda=args.ponder_lambda, weight_decay=args.weight_decay,
                max_steps=args.phase3_steps, grad_clip=args.grad_clip,
                use_bfloat16=use_bf16, device=args.device,
                warmup_steps=min(args.warmup_steps, args.phase3_steps // 10),
                grad_accum_steps=args.grad_accum,
            )
            run_phase(3, trainer, train_loader, args.phase3_steps, args, output_dir, tokenizer_path, log, val_loader)

    save_checkpoint(model, output_dir / "final_model.pt", phases[-1], 0, tokenizer_path)
    print("Done.")


if __name__ == "__main__":
    main()
