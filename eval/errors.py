"""Failure analysis: classify every wrong field by its likely cause.

The README promises a breakdown by cause -- OCR misread, wrong field, formatting,
hallucination -- because the remedy differs for each. A formatting error means the scorer is
too strict; a misread means the OCR stage needs work; a wrong-field error means the
extractor's anchoring is off; a hallucination means the value was never on the page at all.

Classification is heuristic and order-dependent: each rule is checked most-specific first,
and a field that matches nothing lands in `unclassified` rather than being forced into a
bucket it does not belong in.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Mapping, Sequence

from core import io, paths
from core.schema import Document, Extraction, is_correct

# Causes in the order they are tested. Earlier rules are more specific.
CAUSES = ("missing", "formatting", "over_capture", "wrong_field", "ocr_misread",
          "hallucination", "wrong_value", "unclassified")

# Edit distance below this fraction of the gold length counts as a near-miss, i.e. a misread
# rather than an unrelated string.
MISREAD_RATIO = 0.34


def strip_punctuation(value: str) -> str:
    """Lowercase and drop everything except letters and digits, for formatting comparisons."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance, iterative with a single row to keep it allocation-light."""
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, start=1):
        current = [i]
        for j, rchar in enumerate(right, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (lchar != rchar),  # substitution
            ))
        previous = current
    return previous[-1]


def document_text(document: Document) -> str:
    """All OCR token text for one document, normalised for substring containment checks."""
    return strip_punctuation(" ".join(token.text for token in document.tokens))


def classify(name: str, predicted: str, gold: Mapping[str, str], page_text: str) -> str:
    """Name the most likely cause of one wrong field. Assumes the field is already known wrong."""
    if not predicted.strip():
        return "missing"

    target = gold.get(name, "")
    # Same characters once punctuation and case are removed: the value is right, the
    # formatting is not, so this is a scoring strictness problem rather than an extraction one.
    if strip_punctuation(predicted) == strip_punctuation(target):
        return "formatting"

    # The gold value is present but the extractor kept reading past it -- the known
    # over-capture flaw in the address rule, which runs to the first date/total line and
    # swallows anything in between. Distinct from a misread: the text is right, the span is not.
    stripped_target = strip_punctuation(target)
    if stripped_target and stripped_target in strip_punctuation(predicted):
        return "over_capture"

    # The value belongs to a different field on the same document: anchoring picked the
    # wrong line, which is a different fix from misreading the right one.
    for other_name, other_value in gold.items():
        if other_name != name and other_value and is_correct(predicted, other_value):
            return "wrong_field"

    # Close to the gold string: consistent with characters being misrecognised.
    if target:
        distance = edit_distance(strip_punctuation(predicted), strip_punctuation(target))
        if distance <= max(1, int(len(strip_punctuation(target)) * MISREAD_RATIO)):
            return "ocr_misread"

    # Nothing like it appears anywhere on the page, so it was not read off the document.
    if strip_punctuation(predicted) not in page_text:
        return "hallucination"

    # It is genuinely printed on the page, just not the right text: the rule anchored on the
    # wrong line. Separated from `hallucination` because the fix is anchoring, not grounding.
    return "wrong_value"


def analyse(documents: Sequence[Document], extractions: Sequence[Extraction]) -> dict:
    """Tally error causes across a split, overall and per field name."""
    by_id = {doc.doc_id: doc for doc in documents}
    overall: Counter[str] = Counter()
    per_field: dict[str, Counter[str]] = {}
    examples: dict[str, tuple[str, str, str]] = {}
    n_fields = 0

    for extraction in extractions:
        document = by_id.get(extraction.doc_id)
        if document is None:
            continue
        page_text = document_text(document)
        for field in extraction.fields:
            n_fields += 1
            target = document.gold.get(field.name, "")
            if is_correct(field.value, target):
                continue
            cause = classify(field.name, field.value, document.gold, page_text)
            overall[cause] += 1
            per_field.setdefault(field.name, Counter())[cause] += 1
            # Keep one concrete example per cause so the report is inspectable, not just counts.
            examples.setdefault(cause, (extraction.doc_id, field.value, target))

    return {"n_fields": n_fields, "n_errors": sum(overall.values()),
            "overall": overall, "per_field": per_field, "examples": examples}


def build_report(split: str, method: str, analysis: Mapping) -> str:
    """Render the failure analysis as markdown."""
    n_errors = analysis["n_errors"]
    lines = [
        f"# Tripwire failure analysis - {split} split, {method} extractor",
        "",
        f"{n_errors} wrong fields out of {analysis['n_fields']} "
        f"({n_errors / max(analysis['n_fields'], 1):.1%}).",
        "",
        "Causes are assigned by the first matching rule, most specific first. `unclassified` "
        "means no rule fired and the case needs eyes on it.",
        "",
        "| Cause | Count | Share of errors |",
        "|---|---|---|",
    ]
    for cause in CAUSES:
        count = analysis["overall"].get(cause, 0)
        if count:
            lines.append(f"| {cause} | {count} | {count / max(n_errors, 1):.1%} |")

    lines += ["", "## By field", "", "| Field | " + " | ".join(CAUSES) + " |",
              "|---|" + "---|" * len(CAUSES)]
    for field_name, counter in sorted(analysis["per_field"].items()):
        lines.append(f"| {field_name} | "
                     + " | ".join(str(counter.get(c, 0)) for c in CAUSES) + " |")

    lines += ["", "## One example per cause", "",
              "| Cause | Document | Predicted | Gold |", "|---|---|---|---|"]
    for cause in CAUSES:
        if cause in analysis["examples"]:
            doc_id, predicted, gold = analysis["examples"][cause]
            lines.append(f"| {cause} | {doc_id} | `{predicted}` | `{gold}` |")

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the failure analysis for one split; returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="eval.errors", description="Break extraction errors down by likely cause.")
    parser.add_argument("--split", default="test", help="split to analyse")
    parser.add_argument("--dataset", default="sroie")
    parser.add_argument("--method", default="rules", help="extractor whose output to analyse")
    args = parser.parse_args(argv)

    documents_file = paths.documents_path(args.dataset, args.split)
    extractions_file = paths.extraction_path(args.dataset, args.method, args.split)
    for required in (documents_file, extractions_file):
        if not required.exists():
            print(f"missing {required}; run the earlier stage first")
            return 1

    analysis = analyse(io.load_documents(documents_file), io.load_extractions(extractions_file))
    destination = paths.REPORTS / "errors.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report(args.split, args.method, analysis), encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
