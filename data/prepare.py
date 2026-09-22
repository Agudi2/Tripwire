"""CLI that turns raw (or synthetic) data into split processed document files.

    python -m data.prepare --dataset sroie --seed 42
    python -m data.prepare --dataset sroie --seed 42 --fixtures
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace

from core import paths
from core.io import write_jsonl
from core.schema import Document
from data.fixtures import DATASETS, generate_documents
from data.loaders import load_dataset
from data.splits import SPLIT_NAMES, assign_splits, write_split_ids

# Corpus size used by --fixtures: large enough for all three splits to be usable.
FIXTURE_COUNT = 120


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare one dataset; return 0 on success, 1 when the raw data is missing."""
    args = _parse_args(argv)
    try:
        documents = (generate_documents(args.dataset, args.count, args.seed)
                     if args.fixtures else load_dataset(args.dataset))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    for split, count in _write_split(args.dataset, documents, args.seed).items():
        print(f"{args.dataset} {split}: {count} documents -> "
              f"{paths.documents_path(args.dataset, split)}")
    return 0


def _write_split(dataset: str, documents: Sequence[Document], seed: int) -> dict[str, int]:
    """Assign splits, then write the documents and the doc_id list for each one."""
    by_id = {document.doc_id: document for document in documents}
    splits = assign_splits(by_id.keys(), seed)

    written: dict[str, int] = {}
    for split in SPLIT_NAMES:
        doc_ids = splits[split]
        # `replace` keeps Document frozen: the split is stamped on a copy, not mutated in.
        in_split = tuple(replace(by_id[doc_id], split=split) for doc_id in doc_ids)
        written[split] = write_jsonl(paths.documents_path(dataset, split), in_split)
        write_split_ids(dataset, split, doc_ids)
    return written


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m data.prepare",
        description="Normalise a dataset into processed Documents with fixed splits.",
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASETS),
                        help="which dataset to prepare")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the deterministic split assignment")
    parser.add_argument("--fixtures", action="store_true",
                        help="use synthetic documents instead of raw data (runs fully offline)")
    parser.add_argument("--count", type=int, default=FIXTURE_COUNT,
                        help=f"number of synthetic documents when --fixtures is set "
                             f"(default: {FIXTURE_COUNT})")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
