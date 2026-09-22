"""CLI for the extract stage: `python -m extract.run --method llm [--dataset D] [--split S]`.

Reads the normalised documents for one dataset split, runs the chosen extractor over every
document, and writes the results through `core.io` to `core.paths.extraction_path(...)`.

`--live` is opt-in and defaults to off: without it the LLM method replays purely from the
content-addressed cache and a cold cache is a loud error, so no run reaches the network by
accident.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from core import io, paths
from core.schema import Document, Extraction

from extract import llm, rules

# The methods the README names, in the order the results table reports them.
METHODS = ("rules", "llm")


def build_parser() -> argparse.ArgumentParser:
    """Define the command line; kept separate so tests can assert on the defaults."""
    parser = argparse.ArgumentParser(
        prog="extract.run",
        description="Extract document fields with the rules baseline or the LLM extractor.",
    )
    parser.add_argument("--method", required=True, choices=METHODS,
                        help="which extractor to run")
    parser.add_argument("--dataset", default="sroie",
                        help="dataset name, matching the processed documents file")
    parser.add_argument("--split", default="train",
                        help="split to extract; test is reserved for final numbers")
    parser.add_argument("--live", action="store_true",
                        help="allow live API calls on a cache miss (off by default)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one extraction pass and write its artifacts; returns a process exit code."""
    args = build_parser().parse_args(argv)
    documents = _load_documents(args.dataset, args.split)
    extractions = tuple(_extract_one(doc, args.method, args.live) for doc in documents)
    destination = paths.extraction_path(args.dataset, args.method, args.split)
    written = io.write_jsonl(destination, extractions)
    print(f"wrote {written} extractions to {destination}")
    return 0


def _load_documents(dataset: str, split: str) -> tuple[Document, ...]:
    """Load one split's processed documents, failing clearly if it has not been prepared."""
    source = paths.documents_path(dataset, split)
    if not source.exists():
        raise FileNotFoundError(
            f"no processed documents at {source}; run data.prepare for "
            f"dataset {dataset!r} split {split!r} first"
        )
    return io.load_documents(source)


def _extract_one(doc: Document, method: str, live: bool) -> Extraction:
    """Dispatch a single document to the chosen extractor."""
    if method == "rules":
        return rules.extract(doc)
    return llm.extract(doc, live=live)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
