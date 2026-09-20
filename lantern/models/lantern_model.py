"""
LANTERN Language Model.

Complete model combining embeddings, the recursive transformer stack, the
latent pause module, the epistemic probe and the LM head.

Signals available from a single forward pass (all at the last position):
- entropy of the next-token distribution
- epistemic probe output (distilled MC-dropout variance)
- step-KL: KL(p_T || p_{T-1}) between the logits at the last two recursion
  steps of the final block. If the distribution has stopped moving, more
  depth is unlikely to help.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lantern.models.kv_cache import KVCache
from lantern.models.latent_pause import LatentPauseModule
from lantern.models.recursive_transformer import RecursiveTransformerStack
from lantern.uncertainty.epistemic_probe import EpistemicProbe
from lantern.utils.config import LANTERNConfig


class LANTERNModel(nn.Module):
    """
    Complete LANTERN Language Model.

    Integrates:
    - Token embeddings with learned positional encoding
    - Recursive transformer stack with step embeddings
    - Latent pause reasoning module (cross-attention over the stack output)
    - Epistemic probe for uncertainty estimation
    - Tied LM head
    """

    def __init__(self, config: LANTERNConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.transformer = RecursiveTransformerStack(
            num_blocks=config.num_blocks,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            window_size=config.window_size,
            dropout=config.dropout,
            use_halting=config.use_adaptive_halting,
            max_steps=config.max_steps,
            use_rope=config.use_rope,
            attn_impl=config.attn_impl,
            max_position=config.max_position,
            global_token_indices=config.global_token_indices,
        )

        self.epistemic_probe = EpistemicProbe(config.hidden_size)

        self.pause_module = LatentPauseModule(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_pause_steps=config.max_pause_steps,
            window_size=config.window_size,
            dropout=config.dropout,
            use_rope=config.use_rope,
            attn_impl=config.attn_impl,
            max_position=config.max_position,
            global_token_indices=config.global_token_indices,
        )

        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tied

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    # ------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        steps_per_block: Optional[int] = None,
        use_adaptive_halting: bool = False,
        return_hidden_states: bool = False,
        pause_steps: int = 0,
        step_states: Optional[List[torch.Tensor]] = None,
        kv_caches: Optional[List[KVCache]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len].
            attention_mask: Optional additive attention mask.
            steps_per_block: Recursion depth per block (config.steps_base if None).
            use_adaptive_halting: Use the learned halting head (ACT).
            return_hidden_states: Also return the final (post-LayerNorm)
                hidden states, which is what the epistemic probe reads.
            pause_steps: Number of latent pause cycles to apply after the
                stack (0 = none).
            step_states: If a list is given, it receives the final block's
                hidden state after each recursion step (see ``step_kl``).
            kv_caches: Caches from ``create_kv_caches`` for incremental decoding.
            start_pos: Absolute position of input_ids[:, 0] when using caches.

        Returns:
            (logits, hidden_states or None, ponder_cost or None)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if start_pos + seq_len > self.config.max_position:
            raise ValueError(
                f"Sequence length {start_pos + seq_len} exceeds max_position "
                f"{self.config.max_position}. Truncate the input."
            )

        positions = torch.arange(start_pos, start_pos + seq_len, device=device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.embed_dropout(hidden_states)

        steps = steps_per_block if steps_per_block is not None else self.config.steps_base
        hidden_states, _, ponder_cost = self.transformer(
            hidden_states,
            steps_per_block=steps,
            attention_mask=attention_mask,
            use_adaptive_halting=use_adaptive_halting,
            kv_caches=kv_caches[:-1] if kv_caches is not None else None,
            start_pos=start_pos,
            step_states=step_states,
        )

        if kv_caches is not None:
            pause_cache = kv_caches[-1]
            # Every token's context is stored so later tokens can pause over it.
            self.pause_module.write_context(hidden_states, pause_cache, start_pos)
            if pause_steps > 0:
                hidden_states = self.pause_module(
                    hidden_states, num_steps=pause_steps,
                    attention_mask=attention_mask,
                    kv_cache=pause_cache, start_pos=start_pos,
                )
        elif pause_steps > 0:
            hidden_states = self.pause_module(
                hidden_states, num_steps=pause_steps,
                attention_mask=attention_mask, context=hidden_states,
            )

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_hidden_states:
            return logits, hidden_states, ponder_cost
        return logits, None, ponder_cost

    # ------------------------------------------------------------ signals
    def step_logits(self, step_states: List[torch.Tensor], last_only: bool = True) -> torch.Tensor:
        """
        Read out logits from per-step hidden states of the final block.

        Note: pause steps are not part of the trace; the trace reflects
        recursion depth only.

        Returns:
            [num_steps, batch, vocab] if last_only else [num_steps, batch, seq_len, vocab]
        """
        states = torch.stack(step_states, dim=0)
        if last_only:
            states = states[:, :, -1, :]
        return self.lm_head(self.ln_f(states))

    def step_kl(self, step_states: List[torch.Tensor], last_only: bool = True) -> torch.Tensor:
        """
        Convergence signal: KL(p_T || p_{T-1}) between the last two recursion
        steps of the final block. Zero when there was only one step.

        Returns:
            [batch] if last_only else [batch, seq_len]
        """
        if len(step_states) < 2:
            ref = step_states[0][:, -1, 0] if last_only else step_states[0][..., 0]
            return torch.zeros_like(ref)
        logits = self.step_logits(step_states[-2:], last_only=last_only)
        log_p_prev = F.log_softmax(logits[0].float(), dim=-1)
        log_p_last = F.log_softmax(logits[1].float(), dim=-1)
        return torch.sum(log_p_last.exp() * (log_p_last - log_p_prev), dim=-1)

    def probe_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Epistemic probe on post-LayerNorm hidden states -> [batch, seq_len] in [0, 1]."""
        return self.epistemic_probe(hidden_states)

    # ------------------------------------------------------------ caches
    def create_kv_caches(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        max_steps: Optional[int] = None,
    ) -> List[KVCache]:
        """
        One depth-indexed cache per transformer block plus a single-slot
        cache for the pause module's context. Pass the list to ``forward``.

        Args:
            max_steps: Cache slots per block, i.e. the deepest
                ``steps_per_block`` that will be run through these caches.
                Defaults to ``config.max_steps``.
        """
        head_dim = self.config.hidden_size // self.config.num_heads
        slots = max_steps if max_steps is not None else self.config.max_steps
        caches = [
            KVCache(
                max_steps=slots,
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.config.num_blocks)
        ]
        caches.append(
            KVCache(
                max_steps=1,
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
        )
        return caches

    # ------------------------------------------------------------ misc
    def get_embedding_matrix(self) -> torch.Tensor:
        """Embedding matrix [vocab_size, hidden_size] for semantic dispersion."""
        return self.token_embedding.weight

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params

    # ------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        steps_per_block: Optional[int] = None,
        pause_steps: int = 0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Fixed-depth sampling with an incremental KV cache.

        For uncertainty-triggered depth and pause steps use
        ``lantern.controller.adaptive_generation.AdaptiveGenerator``.

        Args:
            input_ids: [batch, seq_len] prompt.
            max_new_tokens: Tokens to generate (stops early at max_position).
            temperature, top_k, top_p: Sampling parameters. temperature=0 is greedy.
            eos_token_id: Stop when every sequence has emitted it.
            steps_per_block: Recursion depth (config.steps_base if None).
            pause_steps: Latent pause cycles per generated token.
            use_cache: Use the KV cache (recomputes the prefix if False).

        Returns:
            [batch, seq_len + generated]
        """
        self.eval()
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        budget = min(max_new_tokens, self.config.max_position - prompt_len)
        if budget <= 0:
            return input_ids

        caches = None
        if use_cache:
            depth = steps_per_block if steps_per_block is not None else self.config.steps_base
            caches = self.create_kv_caches(
                batch_size, prompt_len + budget, device,
                dtype=self.token_embedding.weight.dtype,
                max_steps=max(self.config.max_steps, depth),
            )

        generated = input_ids
        cur_input = input_ids
        start_pos = 0
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(budget):
            logits, _, _ = self.forward(
                cur_input,
                steps_per_block=steps_per_block,
                pause_steps=pause_steps,
                kv_caches=caches,
                start_pos=start_pos,
            )
            next_token = sample_from_logits(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1), torch.full_like(next_token, eos_token_id), next_token
                )
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

            generated = torch.cat([generated, next_token], dim=1)
            if caches is not None:
                start_pos = generated.shape[1] - 1
                cur_input = next_token
            else:
                cur_input = generated

            if eos_token_id is not None and finished.all():
                break

        return generated


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample one token per row from [batch, vocab] logits.

    temperature <= 0 selects the argmax. top_k <= 0 and top_p >= 1 disable
    the respective filters.

    Returns:
        [batch, 1] token ids.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits.float() / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(1, sorted_indices, remove)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
