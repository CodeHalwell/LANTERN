"""
LANTERN Language Model.

Complete model implementation combining embeddings, recursive transformer stack,
and language model head for text generation tasks.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lantern.models.recursive_transformer import RecursiveTransformerStack
from lantern.utils.config import LANTERNConfig


class LANTERNModel(nn.Module):
    """
    Complete LANTERN Language Model.
    
    Integrates:
    - Token embeddings with positional encoding
    - Recursive transformer stack
    - Language model head for next-token prediction
    """
    
    def __init__(self, config: LANTERNConfig):
        """
        Initialize LANTERN model.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Positional embeddings (learned)
        self.position_embedding = nn.Embedding(config.max_position, config.hidden_size)
        
        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Recursive transformer stack
        self.transformer = RecursiveTransformerStack(
            num_blocks=config.num_blocks,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            window_size=config.window_size,
            dropout=config.dropout,
            use_halting=config.use_adaptive_halting,
        )
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings (weight sharing between input and output)
        # Note: This is done before initialization, so both will be initialized together
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        steps_per_block: Optional[int] = None,
        use_adaptive_halting: bool = False,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through LANTERN model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask.
            steps_per_block: Recursion depth per block (uses config default if None).
            use_adaptive_halting: Whether to use learned halting.
            return_hidden_states: Whether to return hidden states.
            
        Returns:
            Tuple of (logits [batch, seq_len, vocab_size], hidden_states if requested).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Validate sequence length
        if seq_len > self.config.max_position:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum position "
                f"embeddings ({self.config.max_position}). Consider truncating the input."
            )
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embed_dropout(hidden_states)
        
        # Pass through recursive transformer stack
        steps = steps_per_block if steps_per_block is not None else self.config.steps_base
        hidden_states, total_steps = self.transformer(
            hidden_states,
            steps_per_block=steps,
            attention_mask=attention_mask,
            use_adaptive_halting=use_adaptive_halting,
        )
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        if return_hidden_states:
            return logits, hidden_states
        return logits, None
    
    def get_embedding_matrix(self) -> torch.Tensor:
        """
        Get the embedding matrix for uncertainty estimation.
        
        Returns:
            Embedding matrix [vocab_size, hidden_size].
        """
        return self.token_embedding.weight
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters.
        
        Args:
            non_embedding: If True, exclude embedding parameters.
            
        Returns:
            Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simple generation method for inference.
        
        Note: This is a basic generation loop. For uncertainty-aware generation
        with THINK tokens and adaptive recursion, use GenerationController.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            eos_token_id: End of sequence token ID.
            
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens].
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            seq_len = input_ids.shape[1]
            if seq_len > self.config.max_position:
                # Keep only the most recent max_position tokens
                input_ids = input_ids[:, -self.config.max_position:]
            
            # Forward pass
            logits, _ = self.forward(input_ids)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                # Ensure k does not exceed vocabulary size
                k = min(top_k, logits.size(-1))
                topk_values = torch.topk(logits, k)[0]
                indices_to_remove = logits < topk_values[..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
