"""Deterministic train/calibration/test assignment, keyed on doc_id and a seed.

Assignment is *hash-based*, not shuffle-based, and that choice is deliberate.

Shuffling a list and slicing it makes a document's split depend on how many other
documents were in the corpus at the time, so adding one receipt next month silently
reassigns documents already used to fit a calibrator, leaking test data into training.
Hashing `(seed, doc_id)` makes each document's split a pure function of its own id:
the corpus can grow, shrink, or be loaded in any order and every existing document
stays exactly where it was. The price is that split sizes land near the target ratios
rather than on them exactly, which is a fine trade for a calibration pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import blake2b
from pathlib import Path

from core import paths

# Split value carried by a document that has not been partitioned yet.
UNASSIGNED = "unassigned"

# Ordered so cumulative boundaries below read the same way.
SPLIT_NAMES: tuple[str, str, str] = ("train", "calibration", "test")

# Target share of the corpus per split; must sum to 1.0.
SPLIT_RATIOS: Mapping[str, float] = {"train": 0.6, "calibration": 0.2, "test": 0.2}

_DIGEST_BYTES = 8
_DIGEST_SCALE = float(1 << (8 * _DIGEST_BYTES))


def assign_split(doc_id: str, seed: int) -> str:
    """Return the split a single document belongs to, from its id and the seed alone."""
    position = _hash_fraction(doc_id, seed)
    cumulative = 0.0
    for name in SPLIT_NAMES:
        cumulative += SPLIT_RATIOS[name]
        if position < cumulative:
            return name
    return SPLIT_NAMES[-1]  # guards float rounding at the top of the range


def assign_splits(doc_ids: Iterable[str], seed: int) -> dict[str, tuple[str, ...]]:
    """Partition doc ids into disjoint train/calibration/test tuples, preserving input order."""
    buckets: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for doc_id in doc_ids:
        buckets[assign_split(doc_id, seed)].append(doc_id)
    return {name: tuple(ids) for name, ids in buckets.items()}


def split_ids_path(dataset: str, split: str) -> Path:
    """Where the doc_id list for one dataset split is stored, resolved through core.paths."""
    return paths.SPLITS / f"{dataset}.{split}.txt"


def write_split_ids(dataset: str, split: str, doc_ids: Iterable[str]) -> Path:
    """Write one doc_id per line for a split and return the file written."""
    path = split_ids_path(dataset, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{doc_id}\n" for doc_id in doc_ids), encoding="utf-8")
    return path


def _hash_fraction(doc_id: str, seed: int) -> float:
    """Map (seed, doc_id) to a stable, uniformly distributed float in [0.0, 1.0)."""
    digest = blake2b(f"{seed}:{doc_id}".encode(), digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big") / _DIGEST_SCALE
