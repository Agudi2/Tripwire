"""The extract CLI reads processed documents and writes a well-formed extractions file."""

from pathlib import Path

import pytest

from core import io, paths
from extract import cache, llm, run

RECEIPT = (
    "SUPER MART SDN BHD",
    "NO 12, JALAN BESAR, 43000 KAJANG",
    "DATE: 18/03/2018",
    "TOTAL 18.50",
)


@pytest.fixture
def project(tmp_path: Path, monkeypatch):
    """Redirect the processed-documents and artifacts directories into a temp project."""
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    return tmp_path


def test_rules_run_writes_one_extraction_per_document(project, make_document) -> None:
    """A rules run over two documents produces a loadable, complete extractions file."""
    docs = [make_document(RECEIPT, doc_id="sroie-001"),
            make_document(RECEIPT, doc_id="sroie-002")]
    io.write_jsonl(paths.documents_path("sroie", "train"), docs)

    assert run.main(["--method", "rules", "--dataset", "sroie", "--split", "train"]) == 0

    written = io.load_extractions(paths.extraction_path("sroie", "rules", "train"))
    assert [e.doc_id for e in written] == ["sroie-001", "sroie-002"]
    assert {e.method for e in written} == {"rules"}
    for extraction in written:
        assert len(extraction.fields) == 4
        for field in extraction.fields:
            assert isinstance(field.value, str)
            assert field.signals["ocr_conf"] > 0.0 or field.value == ""


def test_llm_run_replays_from_the_cache(project, make_document, no_network) -> None:
    """With the cache seeded, an offline llm run writes extractions without any network call."""
    doc = make_document(RECEIPT, doc_id="sroie-001")
    io.write_jsonl(paths.documents_path("sroie", "train"), [doc])
    key = cache.cache_key(llm.MODEL, llm.build_prompt(doc), llm.SCHEMA_VERSION)
    cache.store(key, {"model": llm.MODEL, "schema_version": llm.SCHEMA_VERSION,
                      "samples": [{"company": "SUPER MART SDN BHD", "date": "18/03/2018",
                                   "address": "NO 12, JALAN BESAR, 43000 KAJANG",
                                   "total": "18.50"}]})

    assert run.main(["--method", "llm", "--dataset", "sroie", "--split", "train"]) == 0

    written = io.load_extractions(paths.extraction_path("sroie", "llm", "train"))
    assert len(written) == 1
    assert {f.name: f.value for f in written[0].fields}["total"] == "18.50"


def test_live_defaults_to_off() -> None:
    """--live is opt-in; a bare invocation never enables network access."""
    assert run.build_parser().parse_args(["--method", "llm"]).live is False


def test_missing_documents_file_is_a_clear_error(project) -> None:
    """Pointing at a split that has not been prepared yet fails with a readable message."""
    with pytest.raises(FileNotFoundError, match="sroie"):
        run.main(["--method", "rules", "--dataset", "sroie", "--split", "train"])
