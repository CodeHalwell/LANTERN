# Training LANTERN for Text Generation

This guide explains how to train the LANTERN model on NLP tasks like text generation.

## Quick Start

### Basic Training

Train on a text file with default settings:

```bash
python train.py --data_path your_data.txt
```

### Small Model (for testing/prototyping)

```bash
python train.py \
    --config small \
    --data_path example_data.txt \
    --max_steps 10000 \
    --batch_size 8 \
    --output_dir ./outputs/small_model
```

### Base Model (for production use)

```bash
python train.py \
    --config base \
    --data_path your_data.txt \
    --val_data_path your_val_data.txt \
    --max_steps 100000 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --output_dir ./outputs/base_model
```

## Data Format

The training script expects plain text files. Each file should contain the text you want the model to learn from.

Example (`data.txt`):
```
This is the first sentence in your training data.
This is the second sentence.
You can have as many sentences as you want.
```

The script will:
1. Load the text file
2. Create sequences of length `--seq_length` (default: 512)
3. Train the model to predict the next token in each sequence

**Note**: The current implementation uses character-level tokenization for simplicity. For production use, replace this with a proper tokenizer like BPE or SentencePiece by modifying the `TextDataset` class.

## Configuration Options

### Model Configuration

- `--config`: Choose preset configuration (`small` or `base`)
  - `small`: 256 hidden size, 4 heads, 1 block (for testing)
  - `base`: 512 hidden size, 8 heads, 2 blocks (for production)

- `--vocab_size`: Vocabulary size (default: 32000)
- `--hidden_size`: Override hidden dimension
- `--num_heads`: Override number of attention heads
- `--num_blocks`: Override number of transformer blocks

### Training Configuration

- `--batch_size`: Training batch size (default: 8)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--weight_decay`: Weight decay for AdamW (default: 0.1)
- `--max_steps`: Maximum training steps (default: 10000)
- `--warmup_steps`: Learning rate warmup steps (default: 100)
- `--grad_clip`: Gradient clipping threshold (default: 1.0)

### Data Configuration

- `--data_path`: Path to training data file
- `--val_data_path`: Path to validation data file (optional)
- `--seq_length`: Sequence length for training (default: 512)

### Output Configuration

- `--output_dir`: Directory for checkpoints and logs (default: ./outputs)
- `--eval_interval`: Steps between evaluations (default: 500)
- `--save_interval`: Steps between checkpoint saves (default: 1000)

### Device Configuration

- `--device`: Device to train on (`cuda` or `cpu`, auto-detected by default)

## Checkpoints

The trainer saves several types of checkpoints:

1. **Regular checkpoints**: `checkpoint_step_N.pt` - Saved every `--save_interval` steps
2. **Best model**: `best_model.pt` - Saved when validation loss improves
3. **Final model**: `final_model.pt` - Saved at the end of training

### Resuming Training

Resume from a checkpoint:

```bash
python train.py \
    --resume_from ./outputs/checkpoint_step_5000.pt \
    --data_path your_data.txt
```

## Training Logs

Training metrics are logged to `training_log.jsonl` in the output directory. Each line is a JSON object with metrics:

```json
{"step": 100, "epoch": 0, "train_loss": 3.45, "learning_rate": 0.0001, "elapsed_time": 12.3}
{"step": 500, "val_loss": 3.21}
```

## Using Your Trained Model

After training, you can use your model for generation:

```python
import torch
from lantern.models.lantern_model import LANTERNModel
from lantern.utils.config import create_base_config

# Load config
config = create_base_config()

# Create model
model = LANTERNModel(config)

# Load checkpoint securely
try:
    # Use weights_only=True for security (PyTorch >= 1.13)
    checkpoint = torch.load("./outputs/best_model.pt", weights_only=True)
except TypeError:
    # Fallback for older PyTorch versions
    # WARNING: Only load checkpoints from trusted sources
    checkpoint = torch.load("./outputs/best_model.pt")

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text (simple version)
input_ids = torch.tensor([[1, 2, 3]])  # Your input tokens
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
)

print(output)
```

## Advanced: Uncertainty-Aware Generation

For uncertainty-aware generation with THINK tokens and adaptive recursion:

```python
from lantern import GenerationController, UncertaintyController
from lantern.controller.generation import GenerationConfig

# Create uncertainty controller
uncertainty_controller = UncertaintyController(
    tau_low=1.0,
    tau_mid=2.0,
    tau_high=3.0,
)

# Create generation config with THINK token
gen_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    think_token_id=50256,  # Your THINK token ID
    eos_token_id=50257,    # Your EOS token ID
)

# Create generation controller
controller = GenerationController(
    model=model.transformer,
    lm_head=model.lm_head,
    embedding_matrix=model.get_embedding_matrix(),
    uncertainty_controller=uncertainty_controller,
    config=gen_config,
    recur_fn=lambda h, steps_max: model.transformer(h, steps_per_block=steps_max),
)

# Generate with uncertainty awareness
# Note: Requires proper hidden state management
```

## Tips for Better Training

1. **Start small**: Use `--config small` to test your data and training pipeline
2. **Use validation data**: Provide `--val_data_path` to monitor overfitting
3. **Adjust sequence length**: Longer sequences (`--seq_length`) capture more context but use more memory
4. **Learning rate**: Try 3e-4 (default) or 1e-4 for more stable training
5. **Batch size**: Increase for faster training (if you have enough memory)
6. **Gradient clipping**: Keep at 1.0 to prevent exploding gradients
7. **Warmup**: Use 100-1000 warmup steps depending on total training length

## Model Architecture Features

LANTERN combines several advanced features:

- **Recursive Sparse Transformer**: Weight-shared blocks for efficient depth-on-demand
- **Sliding Window Attention**: O(L × w) complexity instead of O(L²)
- **Uncertainty Estimation**: Multi-signal uncertainty (entropy, semantic dispersion, Bayesian)
- **Adaptive Computation**: Dynamic recursion depth based on difficulty

These features are automatically included in the model. The training script focuses on standard language modeling loss, but the trained model can later be used with the uncertainty-aware generation controller for advanced inference.

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Reduce `--seq_length`
- Use `--config small`

### Loss Not Decreasing

- Check your data format
- Increase `--warmup_steps`
- Decrease `--learning_rate`
- Check for data preprocessing issues

### Training Too Slow

- Increase `--batch_size` (if memory allows)
- Use GPU: `--device cuda`
- Reduce `--seq_length` for faster iterations

## Example Training Commands

### Character-level Shakespeare

```bash
# Download shakespeare data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train
python train.py \
    --config small \
    --data_path input.txt \
    --seq_length 256 \
    --batch_size 16 \
    --max_steps 50000 \
    --output_dir ./outputs/shakespeare
```

### Custom Dataset

```bash
python train.py \
    --config base \
    --data_path ./data/train.txt \
    --val_data_path ./data/val.txt \
    --seq_length 512 \
    --batch_size 8 \
    --max_steps 100000 \
    --eval_interval 1000 \
    --save_interval 5000 \
    --output_dir ./outputs/custom_model
```

## Next Steps

After training:

1. Evaluate your model on test data
2. Experiment with uncertainty-aware generation using `GenerationController`
3. Fine-tune on specific downstream tasks
4. Integrate with production inference pipelines
5. Explore the uncertainty estimation features for confidence calibration
