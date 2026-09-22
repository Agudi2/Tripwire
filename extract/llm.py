"""LLM extractor driven by a JSON schema, cache-first so results stay reproducible.

The only path that reaches the network is `_sample_live`, and it runs only when the caller
passes `live=True` *and* the cache missed *and* ANTHROPIC_API_KEY is in the environment.
Offline with a cold cache is a hard error naming the missing key, never a silent live call.

Signals emitted per field: `agreement` (fraction of sampled outputs whose value matches the
one we kept, compared with the project's shared normalised match), `format_valid` (the
shared validators), `n_samples`, and `mean_logprob` when - and only when - the cached
payload carries per-field log-probs. The Claude Messages API does not expose token
log-probs, so that last signal stays absent for live Anthropic runs by design.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.schema import Document, Extraction, FieldPrediction, is_correct

from extract import cache, rules, validators
from extract.validators import FIELDS

# Model id and schema version both feed the cache key: changing either invalidates entries.
MODEL = "claude-opus-5"
SCHEMA_VERSION = "1"

# Sampling the same prompt several times is what produces the `agreement` signal. Claude
# Opus 5 rejects temperature/top_p, so the spread comes from the model's own variability.
DEFAULT_SAMPLES = 3

# Four short strings plus adaptive thinking; ample headroom without inviting a runaway turn.
_MAX_TOKENS = 4096

# The structured-output contract. Schema-constrained decoding is what makes parsing safe.
EXTRACTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {name: {"type": "string"} for name in FIELDS},
    "required": list(FIELDS),
    "additionalProperties": False,
}

_PROMPT_TEMPLATE = (
    "You are extracting fields from a scanned receipt that has already been through OCR.\n"
    "Copy each value verbatim from the text where possible, keeping the original spelling\n"
    "and punctuation. Use an empty string for a field the receipt does not show.\n"
    "Fields: company, date, address, total.\n\n"
    "OCR text:\n{text}\n"
)


class CacheMissError(RuntimeError):
    """Raised when an offline run needs a cache entry that has not been populated yet."""

    def __init__(self, key: str, path: Path, doc_id: str) -> None:
        """Record the missing key and spell out how to populate it."""
        super().__init__(
            f"no cached LLM response for document {doc_id!r}: cache key {key} "
            f"(expected file {path}). Re-run with --live to populate it, or seed that file."
        )
        self.key = key
        self.path = path


def extract(doc: Document, *, live: bool = False, samples: int = DEFAULT_SAMPLES,
            model: str = MODEL, cache_root: Path | None = None) -> Extraction:
    """Extract one document's fields, replaying from cache and calling the API only if asked."""
    prompt = build_prompt(doc)
    key = cache.cache_key(model, prompt, SCHEMA_VERSION)
    found = cache.lookup(key, root=cache_root)
    if found.hit:
        payload = found.payload
    else:
        if not live:
            raise CacheMissError(key, cache.cache_file(key, cache_root), doc.doc_id)
        payload = {
            "model": model,
            "schema_version": SCHEMA_VERSION,
            "doc_id": doc.doc_id,
            "prompt": prompt,
            "samples": list(_sample_live(prompt, model, samples)),
        }
        cache.store(key, payload, root=cache_root)
    return Extraction(
        doc_id=doc.doc_id,
        method="llm",
        fields=tuple(_aggregate(name, payload["samples"]) for name in FIELDS),
    )


def build_prompt(doc: Document) -> str:
    """Render one document's OCR text into the extraction prompt.

    Deterministic by construction: the cache key hashes this string, so any nondeterminism
    here (dict ordering, a timestamp) would silently destroy every future cache hit.
    """
    return _PROMPT_TEMPLATE.format(text="\n".join(rules.line_texts(doc.tokens)))


def _aggregate(name: str, samples: Sequence[Mapping[str, Any]]) -> FieldPrediction:
    """Reduce the sampled outputs for one field to a kept value plus its signals."""
    values = [str(sample.get(name, "")).strip() for sample in samples]
    if not values:
        return FieldPrediction(name=name, value="",
                               signals={"agreement": 0.0, "format_valid": 0.0, "n_samples": 0.0})
    value, matched = _majority(values)
    signals = {
        "agreement": round(matched / len(values), 4),
        "format_valid": validators.validate(name, value),
        "n_samples": float(len(values)),
    }
    logprob = _mean_logprob(samples, name)
    if logprob is not None:
        signals["mean_logprob"] = round(logprob, 4)
    return FieldPrediction(name=name, value=value, signals=signals)


def _majority(values: Sequence[str]) -> tuple[str, int]:
    """Pick the value most samples agree on, using the project's shared normalised match."""
    best, best_count = "", 0
    for candidate in values:
        count = sum(1 for value in values if is_correct(value, candidate))
        if count > best_count:
            best, best_count = candidate, count
    return best, best_count


def _mean_logprob(samples: Sequence[Mapping[str, Any]], name: str) -> float | None:
    """Average a per-field log-prob across samples, or None when the payload carries none."""
    scores = [sample["logprobs"][name] for sample in samples
              if isinstance(sample.get("logprobs"), Mapping) and name in sample["logprobs"]]
    return sum(scores) / len(scores) if scores else None


def _sample_live(prompt: str, model: str, samples: int) -> tuple[Mapping[str, Any], ...]:
    """Call the Claude API `samples` times. The only function in this project that networks."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError(
            "live extraction needs ANTHROPIC_API_KEY in the environment; export it, "
            "or drop --live to replay from the cache"
        )
    import anthropic  # imported lazily so offline runs need neither the SDK nor a key

    client = anthropic.Anthropic()  # reads the key from the environment itself
    return tuple(_one_call(client, prompt, model) for _ in range(samples))


def _one_call(client: Any, prompt: str, model: str) -> Mapping[str, Any]:
    """Issue one schema-constrained request and parse the JSON object it returns."""
    response = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "effort": "low",  # field copying off OCR text does not need deep reasoning
            "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA},
        },
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"the model declined this extraction request: {response.stop_details}")
    text = "".join(block.text for block in response.content if block.type == "text")
    return json.loads(text)
