"""The prepare CLI writes processed documents and split id lists, and nothing else."""

from pathlib import Path

import pytest

from core import paths
from core.io import load_documents
from data.prepare import main
from data.splits import SPLIT_NAMES, split_ids_path


@pytest.fixture()
def sandbox(monkeypatch, tmp_path: Path) -> Path:
    """Redirect every core.paths output directory into tmp_path so the repo stays clean."""
    monkeypatch.setattr(paths, "RAW", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "SPLITS", tmp_path / "splits")
    return tmp_path


def test_cli_writes_documents_and_split_lists(sandbox: Path) -> None:
    """--fixtures runs end to end offline and writes one file per split, plus id lists."""
    assert main(["--dataset", "sroie", "--seed", "42", "--fixtures"]) == 0

    total = 0
    for split in SPLIT_NAMES:
        docs_path = paths.documents_path("sroie", split)
        ids_path = split_ids_path("sroie", split)
        assert docs_path.exists(), docs_path
        assert ids_path.exists(), ids_path

        docs = load_documents(docs_path)
        ids = ids_path.read_text(encoding="utf-8").split()
        assert docs, f"{split} split is empty"
        assert [d.doc_id for d in docs] == ids
        assert all(d.split == split for d in docs)
        total += len(docs)

    assert total > 0


def test_cli_output_is_reproducible(sandbox: Path) -> None:
    """Running the same command twice produces byte-identical output files."""
    main(["--dataset", "sroie", "--seed", "42", "--fixtures"])
    first = {s: paths.documents_path("sroie", s).read_text(encoding="utf-8") for s in SPLIT_NAMES}
    main(["--dataset", "sroie", "--seed", "42", "--fixtures"])
    second = {s: paths.documents_path("sroie", s).read_text(encoding="utf-8") for s in SPLIT_NAMES}
    assert first == second


def test_cli_seed_changes_the_partition(sandbox: Path) -> None:
    """A different --seed moves documents between splits."""
    main(["--dataset", "sroie", "--seed", "42", "--fixtures"])
    first = paths.documents_path("sroie", "train").read_text(encoding="utf-8")
    main(["--dataset", "sroie", "--seed", "7", "--fixtures"])
    second = paths.documents_path("sroie", "train").read_text(encoding="utf-8")
    assert first != second


@pytest.mark.parametrize("dataset", ["sroie", "cord", "funsd"])
def test_cli_runs_for_every_dataset(dataset: str, sandbox: Path) -> None:
    """All three datasets prepare offline from fixtures."""
    assert main(["--dataset", dataset, "--seed", "42", "--fixtures"]) == 0
    assert load_documents(paths.documents_path(dataset, "test"))


def test_cli_without_fixtures_requires_raw_data(sandbox: Path, capsys) -> None:
    """Without --fixtures the CLI reports the missing raw directory and exits non-zero."""
    assert main(["--dataset", "sroie", "--seed", "42"]) == 1
    assert str(sandbox / "raw" / "sroie") in capsys.readouterr().err


def test_cli_rejects_unknown_dataset(sandbox: Path) -> None:
    """argparse constrains --dataset to the three supported names."""
    with pytest.raises(SystemExit):
        main(["--dataset", "mnist", "--seed", "42", "--fixtures"])


def test_cli_does_not_write_outside_the_sandbox(sandbox: Path) -> None:
    """Every file the CLI creates lives under the redirected paths."""
    main(["--dataset", "cord", "--seed", "42", "--fixtures"])
    written = {p for p in sandbox.rglob("*") if p.is_file()}
    assert written
    assert all(sandbox in p.parents or p.parent.parent == sandbox for p in written)
