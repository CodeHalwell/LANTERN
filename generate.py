"""
Example script demonstrating how to use a trained LANTERN model for text generation.

This shows both basic generation and uncertainty-aware generation. Prompts are
encoded with the ``CharTokenizer`` that was persisted alongside the checkpoint,
and model outputs are decoded back into readable text.
"""

import argparse
import os
from typing import Optional

import torch

from lantern.models.lantern_model import LANTERNModel
from lantern.utils.config import LANTERNConfig
from lantern.utils.tokenizer import CharTokenizer


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    """
    Load a trained LANTERN model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Tuple of (model, config, checkpoint_dict).
    """
    print(f"Loading model from {checkpoint_path}...")

    try:
        # Try to load with weights_only=True for security (PyTorch >= 1.13)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for older PyTorch versions, or checkpoints that embed
        # non-tensor objects (e.g. the serialized tokenizer dict).
        print("Warning: Loading checkpoint without weights_only protection. "
              "Only load checkpoints from trusted sources.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create config from checkpoint
    config_dict = checkpoint['config']
    config = LANTERNConfig(**config_dict)

    # Create model
    model = LANTERNModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Vocab size: {config.vocab_size}")

    return model, config, checkpoint


def load_tokenizer(
    checkpoint_path: str,
    checkpoint: dict,
    tokenizer_path: Optional[str] = None,
) -> Optional[CharTokenizer]:
    """
    Locate and load the CharTokenizer for a checkpoint.

    Resolution order:
      1. Explicit ``--tokenizer`` path.
      2. A serialized tokenizer dict embedded in the checkpoint.
      3. A ``tokenizer_path`` recorded in the checkpoint.
      4. A ``tokenizer.json`` sitting next to the checkpoint file.

    Returns:
        A CharTokenizer, or None if none could be found.
    """
    # 1. Explicit path.
    if tokenizer_path is not None:
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        print(f"Loading tokenizer from {tokenizer_path}")
        return CharTokenizer.load(tokenizer_path)

    # 2. Embedded dict.
    embedded = checkpoint.get('tokenizer') if isinstance(checkpoint, dict) else None
    if isinstance(embedded, dict):
        print("Loading tokenizer embedded in checkpoint.")
        return CharTokenizer.from_dict(embedded)

    # 3. Recorded path in checkpoint.
    recorded = checkpoint.get('tokenizer_path') if isinstance(checkpoint, dict) else None
    if recorded and os.path.exists(recorded):
        print(f"Loading tokenizer from checkpoint-recorded path {recorded}")
        return CharTokenizer.load(recorded)

    # 4. tokenizer.json next to the checkpoint.
    sibling = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "tokenizer.json")
    if os.path.exists(sibling):
        print(f"Loading tokenizer from {sibling}")
        return CharTokenizer.load(sibling)

    return None


def encode_prompt(
    prompt: str,
    tokenizer: Optional[CharTokenizer],
    config: LANTERNConfig,
    device: str,
) -> torch.Tensor:
    """
    Encode a prompt string into input token IDs.

    Uses the tokenizer when available. Falls back to interpreting the prompt as
    comma-separated token IDs only when no tokenizer is present.

    Returns:
        LongTensor of shape [1, seq_len].
    """
    if tokenizer is not None:
        ids = tokenizer.encode(prompt)
        if len(ids) == 0:
            # Avoid an empty prompt; seed with BOS if available else token 0.
            seed = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
            ids = [seed]
        print(f"\nEncoded prompt '{prompt}' -> {len(ids)} token(s).")
        return torch.tensor([ids], dtype=torch.long, device=device)

    # No tokenizer: fall back to comma-separated token IDs.
    try:
        ids = [int(x.strip()) for x in prompt.split(",")]
        print(f"\nNo tokenizer available. Using token IDs as input: {ids}")
        return torch.tensor([ids], dtype=torch.long, device=device)
    except ValueError:
        print(f"\nNo tokenizer available and prompt is not comma-separated token IDs.")
        print("Using random token IDs for demonstration.")
        return torch.randint(0, min(100, config.vocab_size), (1, 10), device=device)


def generate_simple(
    model: LANTERNModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Simple generation without uncertainty awareness.

    Returns:
        Generated token IDs [1, seq_len + max_new_tokens].
    """
    print("\n" + "=" * 60)
    print("Simple Generation")
    print("=" * 60)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    return output


def generate_with_uncertainty(
    model: LANTERNModel,
    config: LANTERNConfig,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
) -> Optional[torch.Tensor]:
    """
    Uncertainty-aware generation via GenerationController.

    Codes against the contract:
        controller = GenerationController.from_model(model, GenerationConfig(...))
        out_ids = controller.generate_tokens(input_ids, max_new_tokens)

    Returns the generated LongTensor, or None if the API is unavailable.
    """
    print("\n" + "=" * 60)
    print("Uncertainty-Aware Generation")
    print("=" * 60)

    try:
        from lantern.controller.generation import GenerationController, GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            steps_base=config.steps_base,
            steps_reasoning=config.steps_reasoning,
            num_bayesian_samples=config.num_bayesian_samples,
            think_token_id=config.think_token_id,
            unknown_token_id=config.unknown_token_id,
            eos_token_id=config.eos_token_id,
        )

        controller = GenerationController.from_model(model, gen_config)
        out_ids = controller.generate_tokens(input_ids, max_new_tokens)
        return out_ids
    except Exception as e:
        print(f"Uncertainty-aware generation is unavailable: {e}")
        print("Falling back to simple generation.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained LANTERN model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json (defaults to checkpoint-embedded tokenizer "
             "or a tokenizer.json next to the checkpoint)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The",
        help="Text prompt (or comma-separated token IDs if no tokenizer)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--use_uncertainty",
        action="store_true",
        help="Use uncertainty-aware generation",
    )

    args = parser.parse_args()

    # Load model and tokenizer.
    model, config, checkpoint = load_model(args.checkpoint, args.device)
    tokenizer = load_tokenizer(args.checkpoint, checkpoint, args.tokenizer)

    if tokenizer is None:
        print("\nWarning: no tokenizer found. Output will be shown as token IDs.")
        print("Provide --tokenizer or train with a real text file to enable "
              "readable decoding.")

    # Encode the prompt.
    input_ids = encode_prompt(args.prompt, tokenizer, config, args.device)

    eos_token_id = config.eos_token_id
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id

    # Generate.
    output = None
    if args.use_uncertainty:
        output = generate_with_uncertainty(
            model,
            config,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    if output is None:
        output = generate_simple(
            model,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
        )

    n_new = output.shape[1] - input_ids.shape[1]

    # Decode and print readable text when a tokenizer is available.
    if tokenizer is not None:
        text = tokenizer.decode(output[0])
        print("\n" + "=" * 60)
        print("Generated text:")
        print("=" * 60)
        print(text)
    else:
        print(f"\nGenerated token IDs:")
        print(output.tolist())

    print(f"\nGenerated {n_new} new tokens")


if __name__ == "__main__":
    main()
