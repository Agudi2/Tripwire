"""Split assignment must be disjoint, total, reproducible, and stable under growth."""

from data.splits import assign_split, assign_splits, split_ids_path

DOC_IDS = tuple(f"sroie-{i:04d}" for i in range(200))


def test_splits_are_disjoint_and_cover_every_document() -> None:
    """Every doc lands in exactly one split and nothing is dropped."""
    splits = assign_splits(DOC_IDS, seed=42)
    assert set(splits) == {"train", "calibration", "test"}
    members = [doc_id for ids in splits.values() for doc_id in ids]
    assert sorted(members) == sorted(DOC_IDS)
    assert len(members) == len(set(members))


def test_every_split_is_non_empty() -> None:
    """A 200-doc corpus produces usable train/calibration/test sets, not empty ones."""
    splits = assign_splits(DOC_IDS, seed=42)
    for ids in splits.values():
        assert len(ids) > 0


def test_same_seed_reproduces_identical_splits() -> None:
    """Two independent runs with the same seed agree exactly."""
    assert assign_splits(DOC_IDS, seed=42) == assign_splits(DOC_IDS, seed=42)


def test_different_seed_produces_different_splits() -> None:
    """The seed actually re-partitions the corpus."""
    assert assign_splits(DOC_IDS, seed=42) != assign_splits(DOC_IDS, seed=7)


def test_adding_documents_does_not_reshuffle_existing_ones() -> None:
    """Growing the corpus leaves every previously assigned doc where it was."""
    before = assign_splits(DOC_IDS, seed=42)
    grown = DOC_IDS + tuple(f"sroie-{i:04d}" for i in range(200, 260))
    after = assign_splits(grown, seed=42)
    for doc_id in DOC_IDS:
        assert _split_of(before, doc_id) == _split_of(after, doc_id)


def test_assignment_depends_only_on_doc_id_and_seed() -> None:
    """A single doc's split is the same whether it is assigned alone or in a corpus."""
    splits = assign_splits(DOC_IDS, seed=42)
    for doc_id in DOC_IDS[:20]:
        assert assign_split(doc_id, seed=42) == _split_of(splits, doc_id)


def test_split_ids_path_is_under_the_shared_splits_dir(monkeypatch, tmp_path) -> None:
    """Split id lists resolve through core.paths, not a hand-built path."""
    from core import paths

    monkeypatch.setattr(paths, "SPLITS", tmp_path)
    assert split_ids_path("sroie", "train").parent == tmp_path


def _split_of(splits: dict, doc_id: str) -> str:
    """Return the name of the split holding doc_id."""
    return next(name for name, ids in splits.items() if doc_id in ids)
