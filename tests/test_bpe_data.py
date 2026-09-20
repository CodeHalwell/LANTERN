"""Tests for the BPE tokenizer and the binary/text datasets."""

import numpy as np
import pytest

pytest.importorskip("tokenizers")

from lantern.data import MemmapDataset, TokenizedTextDataset  # noqa: E402
from lantern.utils.bpe_tokenizer import BPETokenizer  # noqa: E402

DOCS = [
    "Once upon a time there was a little robot who liked to paint.",
    "The robot painted the sky blue and the grass green.",
    "Every day the robot learned a new colour.",
] * 20


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(DOCS, vocab_size=400)


class TestBPETokenizer:
    def test_special_ids(self, tok):
        assert (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) == (0, 1, 2)
        assert tok.vocab_size <= 400

    def test_roundtrip(self, tok):
        text = "The robot painted the sky blue."
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_bos_eos(self, tok):
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id
        assert tok.decode(ids) == "hello"

    def test_unseen_bytes_survive(self, tok):
        text = "zebra ünïcode ✓"
        assert tok.decode(tok.encode(text)) == text

    def test_save_load(self, tok, tmp_path):
        p = tmp_path / "tok.json"
        tok.save(p)
        again = BPETokenizer.load(p)
        assert again.encode("robot") == tok.encode("robot")
        assert again.vocab_size == tok.vocab_size

    def test_encode_batch(self, tok):
        out = tok.encode_batch(["a b", "c"], add_eos=True)
        assert len(out) == 2 and all(o[-1] == tok.eos_token_id for o in out)


class TestDatasets:
    def test_memmap_windows(self, tmp_path):
        arr = np.arange(101, dtype=np.uint16)
        path = tmp_path / "train.bin"
        arr.tofile(path)
        ds = MemmapDataset(path, seq_length=10)
        assert len(ds) == 10
        item = ds[3]
        assert item["input_ids"].tolist() == list(range(30, 40))
        assert item["labels"].tolist() == list(range(31, 41))

    def test_memmap_too_short(self, tmp_path):
        np.arange(5, dtype=np.uint16).tofile(tmp_path / "x.bin")
        with pytest.raises(ValueError):
            MemmapDataset(tmp_path / "x.bin", seq_length=10)

    def test_text_dataset(self, tok, tmp_path):
        p = tmp_path / "docs.txt"
        p.write_text("\n".join(DOCS[:10]), encoding="utf-8")
        ds = TokenizedTextDataset(p, tok, seq_length=8)
        assert len(ds) > 0
        item = ds[0]
        assert item["input_ids"].shape == (8,)
        assert (item["labels"][:-1] == item["input_ids"][1:]).all()
        assert (ds.data == tok.eos_token_id).sum() == 10
