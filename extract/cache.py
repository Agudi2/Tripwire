"""Content-addressed cache for LLM outputs, so a rerun reproduces the same extractions.

An entry is addressed purely by sha256 over (model, schema version, prompt): anything that
could change what the model returns changes the key, and nothing else does. Entries live as
plain JSON under `core.paths.CACHE`, which makes them easy to inspect, seed by hand, or
commit alongside a paper. Lookups always report the key, hit or miss, so a caller that
misses can say exactly which file needs to exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core import paths

# Separator between key components; a byte that cannot occur in a prompt or a model id.
_SEPARATOR = "\x00"


@dataclass(frozen=True)
class CacheLookup:
    """The result of one cache read: the key that was tried, whether it hit, and the payload."""

    key: str
    hit: bool
    payload: Mapping[str, Any] | None = None


def cache_key(model: str, prompt: str, schema_version: str) -> str:
    """Hash the three inputs that determine an LLM response into a stable cache key."""
    material = _SEPARATOR.join((model, schema_version, prompt))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def cache_file(key: str, root: Path | None = None) -> Path:
    """Where the entry for one key lives on disk."""
    return _root(root) / f"{key}.json"


def lookup(key: str, root: Path | None = None) -> CacheLookup:
    """Read one entry, returning a miss rather than raising when it is not there yet."""
    path = cache_file(key, root)
    if not path.exists():
        return CacheLookup(key=key, hit=False, payload=None)
    return CacheLookup(key=key, hit=True, payload=json.loads(path.read_text(encoding="utf-8")))


def store(key: str, payload: Mapping[str, Any], root: Path | None = None) -> Path:
    """Write one entry, creating the cache directory if needed; returns the path written."""
    path = cache_file(key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
                    encoding="utf-8")
    return path


def _root(root: Path | None) -> Path:
    """Resolve the cache directory, defaulting to the project-wide one at call time."""
    return paths.CACHE if root is None else Path(root)
