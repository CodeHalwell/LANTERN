# Training LANTERN

The pipeline is: prepare data → three training phases → depth sweep →
adaptive-depth experiment → generation. Every step has a script.

```
scripts/prepare_data.py            corpus → BPE tokenizer + train.bin / val.bin
train.py                           Phase 1 / 2 / 3 (or all)
scripts/eval_depth.py              loss vs recursion depth from one checkpoint
scripts/experiment_adaptive_depth.py   adaptive vs fixed depth at matched compute
generate.py                        fixed-depth or uncertainty-triggered generation
```

## 1. Data

```bash
pip install -e ".[data]"

# TinyStories (~470M tokens). Good for models up to ~50M parameters.
python scripts/prepare_data.py --dataset tinystories --out_dir data/tinystories --vocab_size 8192

# FineWeb-Edu 10B-token sample. Use this for the 300M model.
python scripts/prepare_data.py --dataset fineweb-edu --out_dir data/fineweb-edu \
    --vocab_size 32000 --max_train_docs 3000000

# Any text file, one document per line
python scripts/prepare_data.py --dataset text --text_path my.txt --out_dir data/mine
```

This writes `tokenizer.json`, `train.bin`, `val.bin` (flat `uint16`, documents
separated by `<eos>`) and `meta.json`. Special ids are fixed: `<pad>=0`,
`<bos>=1`, `<eos>=2`.

## 2. The three phases

| Phase | What trains | Loss | Notes |
|---|---|---|---|
| 1 | everything except probe/pause/halting | next-token CE | recursion depth sampled uniformly in `1..max_steps` per step so every step embedding is trained |
| 2 | epistemic probe only | MSE to MC-dropout variance | backbone frozen and in eval mode; only `nn.Dropout` layers are switched on for sampling |
| 3 | everything, ACT on | CE + λ·ponder | λ ramps linearly; backbone LR 1e-5, reasoning heads 1e-3; half the steps also apply 1..`max_pause_steps` latent pause cycles so the pause module is trained the way decoding uses it |

```bash
# CPU smoke test (a couple of minutes)
python train.py --data_path example_data.txt --config small --vocab_size 512 --seq_length 64 \
    --phase all --phase1_steps 50 --phase2_steps 10 --phase3_steps 20 --output_dir outputs/smoke

# TinyStories, base config (31M), one GPU
python train.py --data_dir data/tinystories --config base --phase all \
    --phase1_steps 20000 --phase2_steps 2000 --phase3_steps 5000 \
    --batch_size 32 --seq_length 512 --output_dir outputs/base-ts

# Resume a later phase from a checkpoint
python train.py --data_dir data/tinystories --resume_from outputs/base-ts/phase1_final.pt \
    --phase 2 --phase2_steps 2000 --output_dir outputs/base-ts
```

Checkpoints are `phase{N}_best.pt`, `phase{N}_final.pt` and `final_model.pt`.
Each holds the model state, the config and the tokenizer path. Metrics go to
`training_log.jsonl`.

Useful flags: `--grad_accum N`, `--compile`, `--attn_impl flex` (CUDA),
`--no_bf16`, `--window_size`, `--dropout`.

## 3. The 300M model

```bash
python train.py --data_dir data/fineweb-edu --config 300m --phase 1 \
    --phase1_steps 60000 --batch_size 8 --grad_accum 8 --seq_length 1024 \
    --learning_rate 3e-4 --warmup_steps 2000 --compile --output_dir outputs/300m
```

What `--config 300m` is:

| | |
|---|---|
| hidden / heads / MLP | 1536 / 12 / 6144 |
| blocks (weight-shared) | 6 |
| steps base / reasoning / max | 2 / 4 / 4 |
| window | 512 |
| parameters | 317M total, 265M non-embedding |

Things to know before you press go:

- **Compute is set by depth, not parameters.** At `steps_base=2` a forward
  pass costs the same as a 12-layer dense model of this width; at
  `steps_reasoning=4` it is 24 layers. Phase 1 samples depths 1..4, so it
  averages about 15 layers. Budget as if training a ~600M dense model.
- **Tokens.** The command above sees 8 × 8 × 1024 × 60000 ≈ 3.9B tokens. That
  is under-trained by Chinchilla standards (6B for 300M) but enough to answer
  the research question. TinyStories is too small and too easy for this size.
- **Memory.** bf16 weights + AdamW state ≈ 4GB; activations dominate. On a
  24GB card use `--batch_size 4 --grad_accum 16`; on 80GB, `--batch_size 32`.
- **Attention.** `sdpa` (default) uses fused kernels that never materialise
  the L×L score matrix, but still computes all of it. `--attn_impl flex`
  is genuinely block-sparse and is the one that delivers the O(L·w) cost;
  it needs CUDA and benefits from `--compile`. Attention dropout is
  skipped on the flex path.
- **Dropout stays on (0.1).** Phase 2 distils MC-dropout variance; with
  dropout 0 there is nothing to distil.

Run the base config on TinyStories first. If depth does not help there,
it will not help at 300M either, and you will have found out in an hour
rather than a week.

## 4. Step 1: does depth help?

```bash
python scripts/eval_depth.py --checkpoint outputs/base-ts/phase1_final.pt \
    --data_dir data/tinystories --depths 1 2 4 8 --pause_steps 0 2 --batches 100
```

Prints loss and perplexity per depth from the same weights. Expected: loss
falls with depth up to `max_steps` and flattens. If depth 8 is not better
than depth 2, the backbone has not learned to use recursion and the
adaptive question is moot.

## 5. Step 3: adaptive vs fixed depth at matched compute

```bash
python scripts/experiment_adaptive_depth.py --checkpoint outputs/base-ts/phase3_final.pt \
    --data_dir data/tinystories --depth_lo 2 --depth_hi 8 --fixed_depths 1 2 4 8 \
    --pause_hi 2 --fractions 0.1 0.25 0.5 --batches 100 --out results/adaptive.json
```

Teacher-forced on validation text. For every token it records the loss at
the shallow and deep depth plus three signals read at the shallow depth:
entropy, the epistemic probe, and step-KL (how much the distribution moved
between the last two recursion steps). A policy escalates the top `f`
fraction of tokens by its signal and is compared with the fixed-depth
curve interpolated at the same mean depth, under two cost models.

Read the table like this:

- Two cost columns. `d_resume` assumes an escalated token continues the
  shallow pass (mean depth `(1-f)·d_lo + f·d_hi`; exact for single-block
  models, a lower bound for stacks). `d_restart` is what the generator
  does today: the shallow pass is discarded and the token reruns at
  `d_hi` (mean depth `d_lo + f·d_hi`). A policy has to beat the restart
  column to pay for itself as implemented.
- `--pause_hi k` gives escalated tokens `k` latent pause cycles on top of
  `d_hi`, exactly what `--pause_steps k` does at generation time. Their
  loss uses the paused deep pass and their cost is charged as
  `k / num_blocks` extra depth units (one pause cycle is one
  attention + FFN layer).
- `delta < 0` means the adaptive policy beats uniform depth at that
  average compute.
- `random` is the control. A signal that does not beat random is not a
  signal.
- `oracle` escalates the tokens where depth actually helped most. It is the
  ceiling; if even the oracle barely beats fixed depth, per-token routing
  cannot win on this data.
- `corr(signal, gain)` is the cheap summary: which signal predicts where
  depth helps.

Caveat: the deep pass deepens the whole context, whereas generation only
deepens escalated tokens. This measures the per-token benefit of depth,
which is what the trigger must predict.

## 6. Generation

```bash
# fixed depth
python generate.py --checkpoint outputs/base-ts/final_model.pt --prompt "Once upon a time" --depth 4

# uncertainty-triggered: calibrate a threshold so ~20% of tokens escalate,
# escalated tokens run at steps_reasoning plus 2 latent pause cycles
python generate.py --checkpoint outputs/base-ts/final_model.pt --prompt "Once upon a time" \
    --signal step_kl --escalate_fraction 0.2 --pause_steps 2 --data_dir data/tinystories --trace
```

`--trace` prints each token with its signal value, whether it escalated and
the depth it got. Thresholds are absolute; `--escalate_fraction` picks one
from validation data so runs are comparable.

In Python:

```python
from lantern import AdaptiveGenerator, AdaptiveGenerationConfig
from lantern.controller.adaptive_generation import collect_signals, calibrate_threshold

signals = collect_signals(model, val_loader, device=device, max_batches=20)
thr = calibrate_threshold(signals["step_kl"], escalate_fraction=0.2)
gen = AdaptiveGenerator(model, AdaptiveGenerationConfig(signal="step_kl", threshold=thr, pause_steps=2))
result = gen.generate(input_ids)
print(tokenizer.decode(result.tokens[0].tolist()), result.escalation_rate)
```

## Troubleshooting

- **Loss not falling in Phase 1:** lower the LR to 1e-4, raise warmup, check
  the tokenizer round-trips your text.
- **Phase 3 loss jumps:** lower `--ponder_lambda` or lengthen its warmup;
  ACT is sensitive to λ.
- **Out of memory:** smaller `--batch_size` with larger `--grad_accum`,
  shorter `--seq_length`, or `--attn_impl flex` on CUDA.
- **Probe outputs are all the same:** Phase 2 ran with dropout 0, or too few
  steps. Check `mc_variance` is not ~0 in the logs.
