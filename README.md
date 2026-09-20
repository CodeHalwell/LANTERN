# LANTERN

**Low-parameter Adaptive Neural Transformer for Entropy-guided ReasoNing**

A small recursive transformer that decides, per token, how much compute to
spend. The same weight-shared block is applied a variable number of times;
a cheap uncertainty signal read after the shallow pass decides whether to
recurse deeper and run extra latent "pause" cycles before emitting a token.

The research question the repository is built to answer:

> Does routing extra recursion depth to the tokens the model is uncertain
> about beat spending the same compute uniformly?

Everything here exists to make that a one-command experiment. See
[TRAINING.md](TRAINING.md) for the full pipeline and the 300M recipe.

## What is in the box

- **Recursive sparse transformer.** `num_blocks` weight-shared blocks, each
  applied `steps` times with a learned step embedding per iteration so the
  representation does not collapse. Sliding-window causal attention with
  global tokens and RoPE.
- **Three attention backends.** `eager` (reference), `sdpa` (fused kernels,
  default) and `flex` (FlexAttention block mask, CUDA). Only `flex` skips
  the masked blocks and gives the O(L·w) cost; `sdpa` computes everything
  but never materialises the score matrix.
- **Depth-indexed KV cache.** One cache slot per recursion step, with
  carry-forward so tokens that stopped early are still visible to later
  tokens that recurse deeper. Prefill plus one-token decoding matches a full
  forward pass to float precision.
- **Latent pause module.** Cross-attention from the token being decided
  over the frozen sequence context, plus an FFN, repeated up to
  `max_pause_steps` times with pause-step embeddings. Same code path in
  training and cached decoding; each pause cycle costs one attention layer.
- **Three free uncertainty signals** at the last position of a single
  forward pass: entropy, an epistemic probe distilled from MC dropout, and
  step-KL (how far the distribution moved between the last two recursion
  steps).
- **Adaptive generator.** Shallow pass → read signal → if above an
  absolute, calibrated threshold, rewind the cache and re-run deep with
  pause steps → sample. Per-token trace of what happened.
- **Three-phase curriculum.** Backbone with random depth; probe
  distillation on a frozen backbone; ACT halting with ponder cost and pause
  training with differential learning rates.
- **Experiment scripts** that give the answer as a table: fixed-depth
  curve, adaptive vs matched-compute fixed depth for each signal, a random
  control and an oracle ceiling.

## Quick start

```bash
pip install -e ".[data,dev]"
pytest tests/ -q

# data → all three phases → depth sweep → the experiment → generation
python scripts/prepare_data.py --dataset tinystories --out_dir data/tinystories --vocab_size 8192
python train.py --data_dir data/tinystories --config base --phase all --batch_size 32 --output_dir outputs/base
python scripts/eval_depth.py --checkpoint outputs/base/phase1_final.pt --data_dir data/tinystories --depths 1 2 4 8
python scripts/experiment_adaptive_depth.py --checkpoint outputs/base/phase3_final.pt --data_dir data/tinystories
python generate.py --checkpoint outputs/base/final_model.pt --prompt "Once upon a time" \
    --signal step_kl --escalate_fraction 0.2 --pause_steps 2 --data_dir data/tinystories --trace
```

## Using the model directly

```python
import torch
from lantern import LANTERNModel, AdaptiveGenerator, AdaptiveGenerationConfig
from lantern.utils.config import create_base_config

model = LANTERNModel(create_base_config()).eval()
x = torch.randint(0, 32000, (1, 16))

# fixed depth, with the per-step trace for the step-KL signal
steps = []
logits, hidden, _ = model(x, steps_per_block=4, return_hidden_states=True, step_states=steps)
print(model.step_kl(steps), model.probe_uncertainty(hidden)[:, -1])

# deeper, with two latent pause cycles
logits, _, _ = model(x, steps_per_block=8, pause_steps=2)

# cached sampling at fixed depth
out = model.generate(x, max_new_tokens=50, steps_per_block=4, pause_steps=1)

# uncertainty-triggered depth (threshold from calibrate_threshold on held-out text)
gen = AdaptiveGenerator(model, AdaptiveGenerationConfig(signal="step_kl", threshold=0.05, pause_steps=2))
result = gen.generate(x)
print(result.escalation_rate, result.trace[0][:3])
```

## Configurations

| name | hidden | blocks | steps base/reason/max | window | params |
|---|---|---|---|---|---|
| `small` | 256 | 1 | 2 / 4 / 8 | 64 | 10.6M (1.3M non-embedding) |
| `base` | 512 | 2 | 4 / 8 / 8 | 256 | 31M (13M) |
| `300m` | 1536 | 6 | 2 / 4 / 4 | 512 | 317M (265M) |

Compute per token scales with `blocks × steps`, not with parameters: the
300M config at reasoning depth costs a 24-layer dense model of the same
width. TRAINING.md has the memory and token budgets.

## How the pieces connect

```
input_ids ──► embed ──► [block_1 × steps] ─ … ─ [block_N × steps] ──► h
                                                 │ per-step states
                                                 ▼
                             step-KL ◄── ln_f + lm_head of last two steps
                             probe   ◄── epistemic_probe(ln_f(h))
                             entropy ◄── softmax(lm_head(ln_f(h)))
                                                 │
       signal > threshold? ──yes──► rewind cache, rerun at steps_deep
                                    + pause_module(h, context=h) × k
                                                 │
                                                 ▼
                                         ln_f ► lm_head ► sample
```

## Honest notes

- The uncertainty controller with EMA thresholds
  (`lantern/controller/uncertainty_controller.py`) escalates a fixed
  fraction of tokens by construction. The adaptive generator uses absolute
  thresholds calibrated once on validation data instead; that is what
  makes runs comparable.
- Semantic dispersion is an unbounded squared embedding distance and is
  computed on the tied output embeddings. It is kept for the legacy
  controller but is not one of the three signals the experiment tests.
- The THINK-token mechanism in the legacy `GenerationController` is not
  used by the new path. Injecting a token the model was never trained on
  hurts; latent pause steps do the same job without touching the token
  stream.
- The teacher-forced experiment deepens the whole context for the deep
  pass. It measures the per-token benefit of depth, which is what the
  trigger must predict, not the exact generation-time cost.

## Testing

```bash
pytest tests/ -q
```

## License

Apache License 2.0
