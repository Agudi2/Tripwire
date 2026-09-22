"""The LLM extractor is cache-first: it replays from disk, and never reaches the network here."""

from pathlib import Path

import pytest

from extract import cache, llm

RECEIPT = (
    "SUPER MART SDN BHD",
    "NO 12, JALAN BESAR, 43000 KAJANG",
    "DATE: 18/03/2018",
    "TOTAL 18.50",
)

# Three sampled outputs that agree on everything except the total.
SAMPLES = [
    {"company": "SUPER MART SDN BHD", "date": "18/03/2018",
     "address": "NO 12, JALAN BESAR, 43000 KAJANG", "total": "18.50"},
    {"company": "SUPER MART SDN BHD", "date": "18/03/2018",
     "address": "NO 12, JALAN BESAR, 43000 KAJANG", "total": "18.50"},
    {"company": "SUPER MART SDN BHD", "date": "18/03/2018",
     "address": "NO 12, JALAN BESAR, 43000 KAJANG", "total": "185.0"},
]


def _seed(doc, samples, root: Path) -> str:
    """Write the cache entry the extractor will look for, and return its key."""
    key = cache.cache_key(llm.MODEL, llm.build_prompt(doc), llm.SCHEMA_VERSION)
    cache.store(key, {"model": llm.MODEL, "schema_version": llm.SCHEMA_VERSION,
                      "samples": samples}, root=root)
    return key


def test_replays_from_a_seeded_cache_without_touching_the_network(
    make_document, tmp_path: Path, no_network
) -> None:
    """A cache hit produces the full Extraction with no API client constructed at all."""
    doc = make_document(RECEIPT)
    _seed(doc, SAMPLES, tmp_path)
    extraction = llm.extract(doc, live=False, cache_root=tmp_path)
    assert extraction.method == "llm"
    assert extraction.doc_id == doc.doc_id
    values = {f.name: f.value for f in extraction.fields}
    assert values["company"] == "SUPER MART SDN BHD"
    assert values["total"] == "18.50"


def test_agreement_signal_counts_the_samples_that_matched(
    make_document, tmp_path: Path, no_network
) -> None:
    """Fields all three samples agree on score 1.0; the contested total scores 2/3."""
    doc = make_document(RECEIPT)
    _seed(doc, SAMPLES, tmp_path)
    signals = {f.name: f.signals for f in llm.extract(doc, live=False, cache_root=tmp_path).fields}
    assert signals["company"]["agreement"] == 1.0
    # Signals are rounded to 4dp so the artifacts and cache files stay readable.
    assert signals["total"]["agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert signals["total"]["format_valid"] == 1.0


def test_logprob_signal_is_emitted_only_when_the_cache_carries_one(
    make_document, tmp_path: Path, no_network
) -> None:
    """Log-probs are optional: absent from the payload means absent from the signals."""
    doc = make_document(RECEIPT)
    _seed(doc, SAMPLES, tmp_path)
    plain = {f.name: f.signals for f in llm.extract(doc, live=False, cache_root=tmp_path).fields}
    assert "mean_logprob" not in plain["total"]

    with_logprobs = [dict(sample, logprobs={"total": -0.2}) for sample in SAMPLES]
    _seed(doc, with_logprobs, tmp_path)
    scored = {f.name: f.signals for f in llm.extract(doc, live=False, cache_root=tmp_path).fields}
    assert scored["total"]["mean_logprob"] == pytest.approx(-0.2)


def test_cache_miss_without_live_raises_and_names_the_key(
    make_document, tmp_path: Path, no_network
) -> None:
    """Offline with an empty cache is an error that tells the user exactly what to populate."""
    doc = make_document(RECEIPT)
    key = cache.cache_key(llm.MODEL, llm.build_prompt(doc), llm.SCHEMA_VERSION)
    with pytest.raises(llm.CacheMissError) as excinfo:
        llm.extract(doc, live=False, cache_root=tmp_path)
    assert key in str(excinfo.value)
    assert excinfo.value.key == key


def test_live_without_an_api_key_fails_before_any_request(
    make_document, tmp_path: Path, monkeypatch, no_network
) -> None:
    """Live mode refuses to start when ANTHROPIC_API_KEY is not in the environment."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    doc = make_document(RECEIPT)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.extract(doc, live=True, cache_root=tmp_path)


def test_prompt_is_deterministic_so_cache_keys_are_reproducible(make_document) -> None:
    """The same document always renders the same prompt, otherwise the cache never hits."""
    doc = make_document(RECEIPT)
    assert llm.build_prompt(doc) == llm.build_prompt(make_document(RECEIPT))
    assert "SUPER MART SDN BHD" in llm.build_prompt(doc)
