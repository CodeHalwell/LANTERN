# LANTERN Examples

Worked commands for the common cases. TRAINING.md explains what each phase
and script does; this file is just recipes.

## Smoke test on CPU (a couple of minutes)

```bash
python train.py --data_path example_data.txt --config small --vocab_size 512 --seq_length 64 \
    --batch_size 4 --phase all --phase1_steps 50 --phase2_steps 10 --phase3_steps 20 \
    --eval_interval 25 --output_dir outputs/smoke

python generate.py --checkpoint outputs/smoke/final_model.pt --prompt "The" --max_tokens 20
```

The tokenizer trained on the fly is saved as `outputs/smoke/tokenizer.json`
and recorded in every checkpoint.

## TinyStories, base model, one GPU

```bash
python scripts/prepare_data.py --dataset tinystories --out_dir data/tinystories --vocab_size 8192

python train.py --data_dir data/tinystories --config base --phase all \
    --phase1_steps 20000 --phase2_steps 2000 --phase3_steps 5000 \
    --batch_size 32 --seq_length 512 --eval_interval 1000 --save_interval 5000 \
    --output_dir outputs/base-ts
```

Then the two experiments:

```bash
python scripts/eval_depth.py --checkpoint outputs/base-ts/phase1_final.pt \
    --data_dir data/tinystories --depths 1 2 4 8 --batches 100

python scripts/experiment_adaptive_depth.py --checkpoint outputs/base-ts/phase3_final.pt \
    --data_dir data/tinystories --depth_lo 2 --depth_hi 8 --pause_hi 2 \
    --fractions 0.1 0.25 0.5 --batches 100 --out results/base-ts.json
```

## Resume a phase

```bash
python train.py --data_dir data/tinystories --resume_from outputs/base-ts/phase1_final.pt \
    --phase 2 --phase2_steps 2000 --output_dir outputs/base-ts
python train.py --data_dir data/tinystories --resume_from outputs/base-ts/phase2_final.pt \
    --phase 3 --phase3_steps 5000 --output_dir outputs/base-ts
```

Phase 3 needs halting enabled. `--phase all` turns it on when the model is
created; for a resumed Phase 3 the checkpoint must have been created with
`use_adaptive_halting=True` (start the original run with `--phase all`).

## 300M on FineWeb-Edu

```bash
python scripts/prepare_data.py --dataset fineweb-edu --out_dir data/fineweb-edu \
    --vocab_size 32000 --max_train_docs 3000000

python train.py --data_dir data/fineweb-edu --config 300m --phase 1 \
    --phase1_steps 60000 --batch_size 8 --grad_accum 8 --seq_length 1024 \
    --warmup_steps 2000 --compile --attn_impl flex --output_dir outputs/300m
```

See the 300M section of TRAINING.md for the compute and memory budget.

## Generation

```bash
# fixed depth, two latent pause cycles per token
python generate.py --checkpoint outputs/base-ts/final_model.pt --prompt "Once upon a time" \
    --depth 4 --pause_steps 2 --temperature 0.8

# escalate ~20% of tokens by step-KL to depth 8 + 2 pause cycles, show the trace
python generate.py --checkpoint outputs/base-ts/final_model.pt --prompt "Once upon a time" \
    --signal step_kl --escalate_fraction 0.2 --deep_depth 8 --pause_steps 2 \
    --data_dir data/tinystories --trace

# same with the epistemic probe, explicit threshold
python generate.py --checkpoint outputs/base-ts/final_model.pt --prompt "Once upon a time" \
    --signal probe --threshold 0.3 --pause_steps 2
```

## Custom architecture

Any preset can be overridden from the command line:

```bash
python train.py --data_dir data/tinystories --config base \
    --hidden_size 768 --num_heads 12 --num_blocks 3 --window_size 128 --dropout 0.05
```

Or in Python:

```python
from lantern.utils.config import LANTERNConfig
config = LANTERNConfig(hidden_size=768, num_heads=12, num_blocks=3,
                       steps_base=3, steps_reasoning=6, max_steps=6, attn_impl="sdpa")
```
