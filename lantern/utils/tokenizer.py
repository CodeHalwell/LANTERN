"""
Character-level tokenizer for LANTERN.

Provides a simple, dependency-light character-level tokenizer that can be
built from text, persisted to JSON, and reloaded. Supports optional special
tokens (bos/eos/unknown/think/pad) which are assigned stable IDs appended
after the character vocabulary.
"""

import json
from typing import Dict, List, Optional, Sequence, Union


# Canonical ordering of supported special tokens. Special tokens are appended
# to the vocabulary after the character vocab in this fixed order so that IDs
# remain stable across save/load round-trips.
_SPECIAL_TOKEN_ORDER = ("bos", "eos", "unknown", "think", "pad")


class CharTokenizer:
    """
    A character-level tokenizer.

    The vocabulary consists of the sorted unique characters of the source text,
    followed by any configured special tokens (in a fixed canonical order).

    Attributes:
        char_to_idx: Mapping from character to integer id.
        idx_to_char: Mapping from integer id to character (or special token name).
        bos_token_id / eos_token_id / unknown_token_id / think_token_id /
        pad_token_id: Integer ids of the configured special tokens, or None
            if that special token is not configured.
    """

    def __init__(
        self,
        chars: Sequence[str],
        special_tokens: Optional[Sequence[str]] = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            chars: Ordered sequence of characters forming the base vocabulary.
                Order is preserved as given (use ``from_text`` for sorted unique).
            special_tokens: Optional sequence of special token names to append.
                Recognized names: bos, eos, unknown, think, pad. They are added
                in a fixed canonical order regardless of input order.
        """
        # Preserve order, drop duplicates.
        seen = set()
        ordered_chars: List[str] = []
        for ch in chars:
            if ch not in seen:
                seen.add(ch)
                ordered_chars.append(ch)
        self._chars: List[str] = ordered_chars

        # Normalize / validate special tokens, keeping canonical order.
        if special_tokens is None:
            special_tokens = []
        special_set = set(special_tokens)
        unknown_names = special_set - set(_SPECIAL_TOKEN_ORDER)
        if unknown_names:
            raise ValueError(
                f"Unrecognized special token name(s): {sorted(unknown_names)}. "
                f"Supported: {list(_SPECIAL_TOKEN_ORDER)}"
            )
        self._special_tokens: List[str] = [
            name for name in _SPECIAL_TOKEN_ORDER if name in special_set
        ]

        self._build_maps()

    def _build_maps(self) -> None:
        """(Re)build the char<->id maps and special-token id attributes."""
        self.char_to_idx: Dict[str, int] = {
            ch: i for i, ch in enumerate(self._chars)
        }
        self.idx_to_char: Dict[int, str] = {
            i: ch for i, ch in enumerate(self._chars)
        }

        # Assign special token ids after the char vocab.
        self._special_token_ids: Dict[str, int] = {}
        next_id = len(self._chars)
        for name in self._special_tokens:
            self._special_token_ids[name] = next_id
            # Represent special tokens in idx_to_char with an angle-bracket marker.
            self.idx_to_char[next_id] = f"<{name}>"
            next_id += 1

        # Expose convenient attributes (None if not configured).
        self.bos_token_id: Optional[int] = self._special_token_ids.get("bos")
        self.eos_token_id: Optional[int] = self._special_token_ids.get("eos")
        self.unknown_token_id: Optional[int] = self._special_token_ids.get("unknown")
        self.think_token_id: Optional[int] = self._special_token_ids.get("think")
        self.pad_token_id: Optional[int] = self._special_token_ids.get("pad")

    @classmethod
    def from_text(
        cls,
        text: str,
        special_tokens: Optional[Sequence[str]] = None,
    ) -> "CharTokenizer":
        """
        Build a tokenizer from text, using sorted unique characters as vocab.

        Args:
            text: Source text to derive the vocabulary from.
            special_tokens: Optional special token names to append.

        Returns:
            A new CharTokenizer.
        """
        chars = sorted(set(text))
        return cls(chars, special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size, including special tokens."""
        return len(self._chars) + len(self._special_tokens)

    @property
    def special_tokens(self) -> List[str]:
        """List of configured special token names (canonical order)."""
        return list(self._special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token ids.

        Unknown characters map to the unknown token id if configured, otherwise
        a clear error is raised.

        Args:
            text: Text to encode.

        Returns:
            List of integer token ids.
        """
        ids: List[int] = []
        for ch in text:
            idx = self.char_to_idx.get(ch)
            if idx is None:
                if self.unknown_token_id is not None:
                    ids.append(self.unknown_token_id)
                else:
                    raise ValueError(
                        f"Character {ch!r} is not in the vocabulary and no "
                        f"'unknown' special token is configured. Configure the "
                        f"tokenizer with special_tokens=['unknown'] to allow "
                        f"out-of-vocabulary characters."
                    )
            else:
                ids.append(idx)
        return ids

    def decode(self, ids: Union[Sequence[int], "object"]) -> str:
        """
        Decode token ids back into a string.

        Args:
            ids: A sequence of ints, or a torch.Tensor (1-D, or with a leading
                batch dim of size 1).

        Returns:
            Decoded string. Special token ids render as ``<name>`` markers.
        """
        ids = self._to_int_list(ids)

        out: List[str] = []
        for i in ids:
            ch = self.idx_to_char.get(i)
            if ch is None:
                raise ValueError(
                    f"Token id {i} is out of range for a vocabulary of size "
                    f"{self.vocab_size}."
                )
            out.append(ch)
        return "".join(out)

    @staticmethod
    def _to_int_list(ids: Union[Sequence[int], "object"]) -> List[int]:
        """Normalize ids (list / tensor / nested single-batch) to a flat int list."""
        # Lazy torch handling so torch stays an optional dependency.
        try:
            import torch

            if isinstance(ids, torch.Tensor):
                if ids.dim() == 0:
                    return [int(ids.item())]
                # Squeeze a leading batch dim of size 1, e.g. [1, T].
                if ids.dim() == 2 and ids.shape[0] == 1:
                    ids = ids[0]
                elif ids.dim() > 1:
                    ids = ids.reshape(-1)
                return [int(x) for x in ids.tolist()]
        except ImportError:
            pass

        return [int(x) for x in ids]

    def to_dict(self) -> Dict:
        """Serialize the tokenizer to a plain dict."""
        return {
            "type": "CharTokenizer",
            "version": 1,
            "chars": self._chars,
            "special_tokens": self._special_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CharTokenizer":
        """Reconstruct a tokenizer from a dict produced by ``to_dict``."""
        if "chars" not in data:
            raise ValueError("Invalid tokenizer dict: missing 'chars'.")
        return cls(
            chars=data["chars"],
            special_tokens=data.get("special_tokens", []),
        )

    def save(self, path: str) -> None:
        """Save the tokenizer to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load a tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"CharTokenizer(vocab_size={self.vocab_size}, "
            f"num_chars={len(self._chars)}, "
            f"special_tokens={self._special_tokens})"
        )
