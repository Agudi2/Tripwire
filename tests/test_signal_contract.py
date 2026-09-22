"""The extract and confidence stages must agree on one signal vocabulary.

These two stages were built independently, and nothing in the type system connects the
names an extractor puts into `FieldPrediction.signals` to the names the calibrator reads
out of them. A mismatch does not raise: the signal is silently dropped and the calibrator
quietly fits on one feature less. That already happened once -- `match_strength` was
dropped entirely, and `mean_logprob` never matched a column named `logprob`.

So these tests run the real extractors and compare what they emit against the schema.
"""

from pathlib import Path

from confidence.features import FEATURE_COLUMNS, IGNORED_SIGNALS
from core.schema import Document
from extract import cache, llm, rules

KNOWN = set(FEATURE_COLUMNS) | set(IGNORED_SIGNALS)

RECEIPT = (
    "SUPER MART SDN BHD",
    "NO 12, JALAN BESAR, 43000 KAJANG",
    "DATE: 18/03/2018",
    "TOTAL 18.50",
)

SAMPLES = [
    {"company": "SUPER MART SDN BHD", "date": "18/03/2018",
     "address": "NO 12, JALAN BESAR, 43000 KAJANG", "total": "18.50"},
    {"company": "SUPER MART SDN BHD", "date": "18/03/2018",
     "address": "NO 12, JALAN BESAR, 43000 KAJANG", "total": "18.50"},
]


def _emitted_signal_names(extraction) -> set[str]:
    """Collect every signal name appearing anywhere in one extraction's fields."""
    names: set[str] = set()
    for field in extraction.fields:
        names.update(field.signals)
    return names


def test_rules_emits_only_signals_the_calibrator_knows(make_document) -> None:
    """Every signal the rules extractor produces is either a feature column or ignored on purpose."""
    extraction = rules.extract(make_document(RECEIPT))
    unknown = _emitted_signal_names(extraction) - KNOWN
    assert not unknown, f"rules emits signals the confidence stage will silently drop: {unknown}"


def test_llm_emits_only_signals_the_calibrator_knows(
    make_document, tmp_path: Path, no_network
) -> None:
    """Same contract for the LLM extractor, replayed from a seeded cache."""
    doc = make_document(RECEIPT)
    key = cache.cache_key(llm.MODEL, llm.build_prompt(doc), llm.SCHEMA_VERSION)
    cache.store(key, {"model": llm.MODEL, "schema_version": llm.SCHEMA_VERSION,
                      "samples": SAMPLES}, root=tmp_path)
    extraction = llm.extract(doc, live=False, cache_root=tmp_path)
    unknown = _emitted_signal_names(extraction) - KNOWN
    assert not unknown, f"llm emits signals the confidence stage will silently drop: {unknown}"


def test_match_strength_is_a_feature_not_a_dropped_signal() -> None:
    """Pins the specific regression: match_strength is the rules extractor's main discriminator."""
    assert "match_strength" in FEATURE_COLUMNS


def test_every_ignored_signal_is_genuinely_unused() -> None:
    """An ignored name must not also be a feature column, or intent becomes ambiguous."""
    assert not (set(IGNORED_SIGNALS) & set(FEATURE_COLUMNS))
