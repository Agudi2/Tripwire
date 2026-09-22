"""Cached LLM outputs are content-addressed, so a rerun replays byte-identical results."""

from pathlib import Path

from extract import cache

MODEL = "claude-opus-5"
PROMPT = "Extract the fields from this receipt:\nSUPER MART SDN BHD"
VERSION = "1"


def test_key_is_stable_across_calls() -> None:
    """The same (model, prompt, schema version) always hashes to the same key."""
    first = cache.cache_key(MODEL, PROMPT, VERSION)
    second = cache.cache_key(MODEL, PROMPT, VERSION)
    assert first == second
    assert len(first) == 64


def test_key_changes_with_every_input() -> None:
    """Changing the prompt, the model, or the schema version all invalidate the key."""
    base = cache.cache_key(MODEL, PROMPT, VERSION)
    assert cache.cache_key(MODEL, PROMPT + " and the total.", VERSION) != base
    assert cache.cache_key("claude-sonnet-5", PROMPT, VERSION) != base
    assert cache.cache_key(MODEL, PROMPT, "2") != base


def test_miss_reports_the_key_and_no_payload(tmp_path: Path) -> None:
    """An empty cache returns a miss that still names the key the caller needs to populate."""
    key = cache.cache_key(MODEL, PROMPT, VERSION)
    result = cache.lookup(key, root=tmp_path)
    assert result.hit is False
    assert result.payload is None
    assert result.key == key


def test_store_then_lookup_is_a_hit(tmp_path: Path) -> None:
    """A stored payload reads back unchanged on the next lookup."""
    key = cache.cache_key(MODEL, PROMPT, VERSION)
    payload = {"model": MODEL, "schema_version": VERSION, "samples": [{"total": "18.50"}]}
    path = cache.store(key, payload, root=tmp_path)
    assert path.exists()
    result = cache.lookup(key, root=tmp_path)
    assert result.hit is True
    assert result.payload == payload


def test_cache_file_is_named_by_the_key(tmp_path: Path) -> None:
    """Cache entries are addressed purely by content hash, so they are easy to seed by hand."""
    key = cache.cache_key(MODEL, PROMPT, VERSION)
    assert cache.cache_file(key, root=tmp_path) == tmp_path / f"{key}.json"
