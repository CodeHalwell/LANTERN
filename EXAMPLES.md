# LANTERN Training Examples

This directory contains example scripts and data for training LANTERN models.

## Quick Start Examples

### 1. Test Training (Small Model, Synthetic Data)

Verify the training pipeline works:

```bash
python train.py \
    --config small \
    --data_path example_data.txt \
    --max_steps 100 \
    --batch_size 2 \
    --seq_length 64 \
    --output_dir ./outputs/test_run
```

### 2. Character-Level Text Generation

Train on the provided example data:

```bash
python train.py \
    --config small \
    --data_path example_data.txt \
    --max_steps 5000 \
    --batch_size 8 \
    --seq_length 128 \
    --eval_interval 500 \
    --save_interval 1000 \
    --output_dir ./outputs/char_model
```

### 3. Base Model Training

For production-quality models:

```bash
python train.py \
    --config base \
    --data_path your_large_dataset.txt \
    --val_data_path your_validation_data.txt \
    --max_steps 100000 \
    --batch_size 16 \
    --seq_length 512 \
    --learning_rate 3e-4 \
    --warmup_steps 1000 \
    --eval_interval 1000 \
    --save_interval 5000 \
    --output_dir ./outputs/base_model
```

### 4. Resume Training from Checkpoint

Continue training from a saved checkpoint:

```bash
python train.py \
    --resume_from ./outputs/base_model/checkpoint_step_50000.pt \
    --data_path your_dataset.txt \
    --max_steps 150000
```

## Using Trained Models

### Basic Generation

Generate text using a trained model:

```bash
python generate.py \
    --checkpoint ./outputs/char_model/best_model.pt \
    --prompt "1,2,3,4,5" \
    --max_tokens 50 \
    --temperature 0.8
```

### Custom Model Sizes

Train with custom architecture:

```bash
python train.py \
    --config base \
    --hidden_size 768 \
    --num_heads 12 \
    --num_blocks 3 \
    --data_path your_dataset.txt \
    --output_dir ./outputs/custom_model
```

## Dataset Preparation

### Character-Level

The training script includes simple character-level tokenization by default:
- Each unique character gets a token ID
- Suitable for small datasets and testing
- Works out-of-the-box with any text file

### Token-Level (Recommended for Production)

For production use, replace the `TextDataset` class to use proper tokenization:

```python
# Example with Hugging Face tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize your data
text = "Your training text here"
tokens = tokenizer.encode(text)
```

Then modify `train.py` to use these pre-tokenized datasets.

## Training Tips

### GPU Training

LANTERN works best on GPU:

```bash
python train.py \
    --config base \
    --device cuda \
    --batch_size 32 \
    --data_path your_dataset.txt
```

### Mixed Precision (Optional)

For faster training on modern GPUs, you can add mixed precision support:

```python
from torch.cuda.amp import autocast, GradScaler

# In the training loop
scaler = GradScaler()

with autocast():
    loss = compute_loss(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training (Optional)

For multi-GPU training, use PyTorch DDP:

```bash
torchrun --nproc_per_node=4 train.py \
    --config base \
    --data_path your_dataset.txt
```

## Monitoring Training

### Training Logs

Metrics are saved to `training_log.jsonl`:

```bash
# View training progress
tail -f outputs/your_model/training_log.jsonl

# Parse with Python
import json
with open('outputs/your_model/training_log.jsonl') as f:
    for line in f:
        metrics = json.loads(line)
        print(f"Step {metrics['step']}: Loss {metrics.get('train_loss', 'N/A')}")
```

### Tensorboard (Optional)

Add Tensorboard logging for visualization:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs/experiment_name')
writer.add_scalar('Loss/train', loss, step)
```

Then view with:

```bash
tensorboard --logdir=./runs
```

## Model Variants

### Small (Testing/Prototyping)
- Hidden size: 256
- Heads: 4
- Blocks: 1
- Parameters: ~660K
- Good for: Testing, small datasets, CPU training

### Base (Production)
- Hidden size: 512
- Heads: 8
- Blocks: 2
- Parameters: ~2.6M
- Good for: Most applications, balanced quality/speed

### Custom (Advanced)
- Define your own architecture
- Scale up for better quality
- Scale down for faster inference

## Advanced Features

### Uncertainty-Aware Generation

After training, use the uncertainty controller for adaptive generation:

```python
from lantern import GenerationController, UncertaintyController

# See generate.py for full example
controller = GenerationController(...)
```

### Bayesian Refinement

Enable Bayesian refinement during generation for high-uncertainty decisions:

```python
gen_config = GenerationConfig(
    num_bayesian_samples=5,  # More samples = better uncertainty estimation
)
```

### Adaptive Recursion

The model automatically adjusts computation depth based on difficulty:
- Base mode: 4 steps (config.steps_base)
- Reasoning mode: 8 steps (config.steps_reasoning)
- Triggered by uncertainty thresholds

## Common Issues

### Out of Memory
- Reduce `--batch_size`
- Reduce `--seq_length`
- Use smaller model (`--config small`)
- Enable gradient checkpointing (requires code modification)

### Training Unstable
- Increase `--warmup_steps`
- Decrease `--learning_rate`
- Check `--grad_clip` is enabled (default: 1.0)
- Verify data quality

### Poor Generation Quality
- Train for more steps
- Use larger model (`--config base`)
- Increase dataset size
- Lower temperature during generation
- Use validation set to prevent overfitting

## Next Steps

1. Train a small model on example data
2. Evaluate on your specific task
3. Scale up model size as needed
4. Integrate with your application
5. Experiment with uncertainty-aware generation

For more details, see [TRAINING.md](TRAINING.md) in the repository root.
