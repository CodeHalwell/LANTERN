"""
Byte-level BPE tokenizer for LANTERN.

Thin wrapper over the Hugging Face ``tokenizers`` library. Special tokens
have fixed ids: <pad>=0, <bos>=1, <eos>=2. Documents are separated by <eos>
in the training stream, which is also what ``generate`` stops on.
"""

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "BPETokenizer needs the `tokenizers` package: pip install -e '.[data]'"
    ) from e


PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, EOS]


class BPETokenizer:
    """Byte-level BPE tokenizer with pad/bos/eos special tokens."""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.pad_token_id = tokenizer.token_to_id(PAD)
        self.bos_token_id = tokenizer.token_to_id(BOS)
        self.eos_token_id = tokenizer.token_to_id(EOS)

    # ------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8192,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train from an iterable of documents."""
        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def train_from_files(
        cls, files: Sequence[Union[str, Path]], vocab_size: int = 8192
    ) -> "BPETokenizer":
        def _iter() -> Iterator[str]:
            for f in files:
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            yield line

        return cls.train(_iter(), vocab_size=vocab_size)

    # ------------------------------------------------------------ io
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # ------------------------------------------------------------ encode/decode
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def encode_batch(
        self, texts: Sequence[str], add_bos: bool = False, add_eos: bool = False
    ) -> List[List[int]]:
        encoded = self._tok.encode_batch(list(texts))
        out = []
        for enc in encoded:
            ids = enc.ids
            if add_bos:
                ids = [self.bos_token_id] + ids
            if add_eos:
                ids = ids + [self.eos_token_id]
            out.append(ids)
        return out

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)
