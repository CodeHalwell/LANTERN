"""Tests for the CharTokenizer."""

import pytest

from lantern.utils.tokenizer import CharTokenizer


SAMPLE_TEXT = "hello world\nthe quick brown fox jumps over the lazy dog 1234567890."


def test_from_text_vocab_size_matches_unique_chars():
    tok = CharTokenizer.from_text(SAMPLE_TEXT)
    expected = len(set(SAMPLE_TEXT))
    assert tok.vocab_size == expected
    assert len(tok) == expected


def test_vocab_is_sorted_unique():
    tok = CharTokenizer.from_text("cba")
    # Sorted unique chars -> a, b, c with ids 0, 1, 2.
    assert tok.encode("abc") == [0, 1, 2]


def test_encode_decode_roundtrip_lossless():
    tok = CharTokenizer.from_text(SAMPLE_TEXT)
    ids = tok.encode(SAMPLE_TEXT)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == SAMPLE_TEXT


def test_decode_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    tok = CharTokenizer.from_text(SAMPLE_TEXT)
    ids = tok.encode("hello")
    tensor_1d = torch.tensor(ids, dtype=torch.long)
    assert tok.decode(tensor_1d) == "hello"
    # Leading batch dim of size 1 should be squeezed.
    tensor_2d = torch.tensor([ids], dtype=torch.long)
    assert tok.decode(tensor_2d) == "hello"


def test_special_token_ids_appended_after_char_vocab():
    tok = CharTokenizer.from_text(
        "abc", special_tokens=["bos", "eos", "unknown", "think", "pad"]
    )
    n_chars = 3
    assert tok.vocab_size == n_chars + 5
    # Canonical order: bos, eos, unknown, think, pad.
    assert tok.bos_token_id == n_chars + 0
    assert tok.eos_token_id == n_chars + 1
    assert tok.unknown_token_id == n_chars + 2
    assert tok.think_token_id == n_chars + 3
    assert tok.pad_token_id == n_chars + 4


def test_special_token_order_is_canonical_regardless_of_input_order():
    tok = CharTokenizer.from_text("abc", special_tokens=["pad", "eos", "bos"])
    # bos before eos before pad regardless of input ordering.
    assert tok.bos_token_id < tok.eos_token_id < tok.pad_token_id


def test_unconfigured_special_tokens_are_none():
    tok = CharTokenizer.from_text("abc")
    assert tok.bos_token_id is None
    assert tok.eos_token_id is None
    assert tok.unknown_token_id is None
    assert tok.think_token_id is None
    assert tok.pad_token_id is None


def test_unknown_char_maps_to_unknown_id_when_configured():
    tok = CharTokenizer.from_text("abc", special_tokens=["unknown"])
    ids = tok.encode("axc")  # 'x' is out of vocab
    assert ids[1] == tok.unknown_token_id


def test_unknown_char_raises_when_not_configured():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("axc")


def test_invalid_special_token_name_raises():
    with pytest.raises(ValueError):
        CharTokenizer.from_text("abc", special_tokens=["nope"])


def test_save_load_roundtrip(tmp_path):
    tok = CharTokenizer.from_text(
        SAMPLE_TEXT, special_tokens=["eos", "unknown", "think"]
    )
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))

    loaded = CharTokenizer.load(str(path))
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.eos_token_id == tok.eos_token_id
    assert loaded.unknown_token_id == tok.unknown_token_id
    assert loaded.think_token_id == tok.think_token_id
    # Encoding/decoding behaviour is preserved.
    assert loaded.encode(SAMPLE_TEXT) == tok.encode(SAMPLE_TEXT)
    assert loaded.decode(loaded.encode(SAMPLE_TEXT)) == SAMPLE_TEXT


def test_to_dict_from_dict_roundtrip():
    tok = CharTokenizer.from_text("abc", special_tokens=["eos"])
    data = tok.to_dict()
    rebuilt = CharTokenizer.from_dict(data)
    assert rebuilt.vocab_size == tok.vocab_size
    assert rebuilt.eos_token_id == tok.eos_token_id
    assert rebuilt.encode("abc") == tok.encode("abc")
