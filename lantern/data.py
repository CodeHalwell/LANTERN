"""
Datasets for LANTERN language-model training.

Two sources:

- ``MemmapDataset``: a flat ``uint16`` token file produced by
  ``scripts/prepare_data.py`` (nanoGPT-style ``train.bin`` / ``val.bin``).
  Yields non-overlapping ``seq_length + 1`` windows.
- ``TokenizedTextDataset``: tokenizes a plain text file on the fly with a
  ``BPETokenizer``. Fine for smoke tests and small corpora.

Both return ``{"input_ids", "labels"}`` with labels shifted by one.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from lantern.utils.bpe_tokenizer import BPETokenizer

TOKEN_DTYPE = np.uint16


class MemmapDataset(Dataset):
    """Non-overlapping windows over a flat uint16 token file."""

    def __init__(self, path: Union[str, Path], seq_length: int = 512):
        self.path = Path(path)
        self.seq_length = seq_length
        self.data = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        if len(self.data) < seq_length + 1:
            raise ValueError(
                f"{self.path} holds {len(self.data)} tokens; need at least "
                f"{seq_length + 1} for one window."
            )
        self.num_windows = (len(self.data) - 1) // seq_length

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_length
        chunk = torch.from_numpy(
            self.data[start:start + self.seq_length + 1].astype(np.int64)
        )
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class TokenizedTextDataset(Dataset):
    """Tokenize a text file (one document per line) and window it."""

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer: BPETokenizer,
        seq_length: int = 512,
    ):
        self.seq_length = seq_length
        ids = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ids.extend(tokenizer.encode(line, add_eos=True))
        if len(ids) < seq_length + 1:
            raise ValueError(
                f"{path} tokenizes to {len(ids)} tokens; need at least "
                f"{seq_length + 1}. Use a shorter --seq_length or more text."
            )
        self.data = torch.tensor(ids, dtype=torch.long)
        self.num_windows = (len(ids) - 1) // seq_length

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_length
        chunk = self.data[start:start + self.seq_length + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def load_data_dir_meta(data_dir: Union[str, Path]) -> dict:
    """Read ``meta.json`` written by ``scripts/prepare_data.py``."""
    with open(Path(data_dir) / "meta.json") as fh:
        return json.load(fh)


def load_tokenizer_for(data_dir: Union[str, Path]) -> Optional[BPETokenizer]:
    path = Path(data_dir) / "tokenizer.json"
    return BPETokenizer.load(path) if path.exists() else None
