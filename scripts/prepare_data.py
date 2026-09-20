#!/usr/bin/env python
"""
Download a corpus, train a byte-level BPE tokenizer and write flat uint16
token files for training.

Output directory layout::

    <out_dir>/tokenizer.json   trained BPE tokenizer
    <out_dir>/train.bin        uint16 tokens, documents separated by <eos>
    <out_dir>/val.bin          held-out documents, same format
    <out_dir>/meta.json        vocab size, token counts, special ids

Examples::

    # ~470M tokens of children's stories; enough for models up to ~50M params
    python scripts/prepare_data.py --dataset tinystories --out_dir data/tinystories --vocab_size 8192

    # For the 300M model: ~10B tokens of educational web text
    python scripts/prepare_data.py --dataset fineweb-edu --out_dir data/fineweb-edu \\
        --vocab_size 32000 --max_train_docs 3000000

    # Any text file, one document per line
    python scripts/prepare_data.py --dataset text --text_path my.txt --out_dir data/mine
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lantern.data import TOKEN_DTYPE  # noqa: E402
from lantern.utils.bpe_tokenizer import BPETokenizer  # noqa: E402

DATASETS = {
    "tinystories": {"path": "roneneldan/TinyStories", "name": None, "text_field": "text"},
    "fineweb-edu": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "text_field": "text"},
}


def iter_hf_docs(dataset: str, split: str, limit: Optional[int]) -> Iterator[str]:
    from datasets import load_dataset

    spec = DATASETS[dataset]
    ds = load_dataset(spec["path"], name=spec["name"], split=split, streaming=True)
    field = spec["text_field"]
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        text = ex[field]
        if text and text.strip():
            yield text.strip()


def iter_text_file(path: str, limit: Optional[int]) -> Iterator[str]:
    with open(path, encoding="utf-8") as fh:
        n = 0
        for line in fh:
            line = line.strip()
            if line:
                yield line
                n += 1
                if limit is not None and n >= limit:
                    break


def choose_val_docs(requested: int, n_docs: int) -> int:
    """
    Held-out document count for a finite source of ``n_docs`` documents.
    Never consumes the whole corpus: falls back to a tenth (at least one
    document, leaving at least one for training).
    """
    if n_docs <= 1:
        raise ValueError("need at least two documents to make a train/val split")
    if requested < n_docs:
        return requested
    return max(1, min(n_docs - 1, n_docs // 10))


def take(it: Iterable[str], n: int) -> Iterator[str]:
    for i, x in enumerate(it):
        if i >= n:
            break
        yield x


def write_tokens(
    docs: Iterable[str], tokenizer: BPETokenizer, out_path: Path, batch_size: int = 1000
) -> int:
    total = 0
    batch = []
    with open(out_path, "wb") as fh:
        def flush():
            nonlocal total
            if not batch:
                return
            for ids in tokenizer.encode_batch(batch, add_eos=True):
                arr = np.asarray(ids, dtype=TOKEN_DTYPE)
                arr.tofile(fh)
                total += len(arr)
            batch.clear()

        for i, doc in enumerate(docs):
            batch.append(doc)
            if len(batch) >= batch_size:
                flush()
                if (i + 1) % 100_000 == 0:
                    print(f"  {i + 1:,} docs, {total:,} tokens", flush=True)
        flush()
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=[*DATASETS, "text"], required=True)
    ap.add_argument("--text_path", type=str, default=None, help="For --dataset text")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--tokenizer_docs", type=int, default=200_000,
                    help="Documents used to train the tokenizer")
    ap.add_argument("--max_train_docs", type=int, default=None)
    ap.add_argument("--val_docs", type=int, default=10_000,
                    help="Held-out documents (taken from the validation split "
                         "where one exists, else from the head of train)")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="Reuse an existing tokenizer.json instead of training one")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "text":
        if not args.text_path:
            ap.error("--text_path is required with --dataset text")
        all_docs = lambda: iter_text_file(args.text_path, None)  # noqa: E731
        has_val_split = False
    else:
        all_docs = lambda: iter_hf_docs(args.dataset, "train", None)  # noqa: E731
        has_val_split = args.dataset == "tinystories"

    # ---- tokenizer
    tok_path = out_dir / "tokenizer.json"
    if args.tokenizer:
        tokenizer = BPETokenizer.load(args.tokenizer)
        tokenizer.save(tok_path)
        print(f"Loaded tokenizer from {args.tokenizer} (vocab {tokenizer.vocab_size})")
    else:
        print(f"Training BPE tokenizer (vocab {args.vocab_size}) on {args.tokenizer_docs:,} docs...")
        tokenizer = BPETokenizer.train(take(all_docs(), args.tokenizer_docs), vocab_size=args.vocab_size)
        tokenizer.save(tok_path)
        print(f"Saved tokenizer to {tok_path} (vocab {tokenizer.vocab_size})")

    max_vocab = np.iinfo(TOKEN_DTYPE).max + 1
    if tokenizer.vocab_size > max_vocab:
        ap.error(
            f"tokenizer has {tokenizer.vocab_size} tokens but {TOKEN_DTYPE.__name__} storage "
            f"holds at most {max_vocab}; use a smaller vocabulary"
        )

    # ---- validation
    print("Writing val.bin ...")
    if has_val_split:
        val_docs = iter_hf_docs(args.dataset, "validation", args.val_docs)
        train_skip = 0
    else:
        n_val_docs = args.val_docs
        if args.dataset == "text":
            n_docs = sum(1 for _ in all_docs())
            n_val_docs = choose_val_docs(args.val_docs, n_docs)
            if n_val_docs != args.val_docs:
                print(f"  note: {n_docs} documents in total; holding out {n_val_docs} "
                      f"instead of --val_docs {args.val_docs} so training keeps the rest")
        val_docs = take(all_docs(), n_val_docs)
        train_skip = n_val_docs
    n_val = write_tokens(val_docs, tokenizer, out_dir / "val.bin")
    print(f"  val: {n_val:,} tokens")
    if n_val == 0:
        (out_dir / "val.bin").unlink()
        print("  no held-out documents; val.bin not written (train.py will run without validation)")

    # ---- train
    print("Writing train.bin ...")

    def train_docs():
        for i, d in enumerate(all_docs()):
            if i < train_skip:
                continue
            if args.max_train_docs is not None and i - train_skip >= args.max_train_docs:
                break
            yield d

    n_train = write_tokens(train_docs(), tokenizer, out_dir / "train.bin")
    print(f"  train: {n_train:,} tokens")
    if n_train == 0:
        ap.error("no training documents were written; lower --val_docs or supply more data")

    meta = {
        "dataset": args.dataset,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "dtype": "uint16",
    }
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
