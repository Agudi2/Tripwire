"""JSONL read/write helpers shared by every stage, so no stage invents its own format."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from core.schema import Document, Extraction, FieldPrediction, OCRToken, ScoredField


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    """Write dataclasses (or plain dicts) as one JSON object per line; return the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record) if is_dataclass(record) else record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield one parsed dict per line, skipping blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_documents(path: Path) -> tuple[Document, ...]:
    """Read a processed documents file back into Document objects."""
    return tuple(_to_document(row) for row in read_jsonl(path))


def load_extractions(path: Path) -> tuple[Extraction, ...]:
    """Read an extractions file back into Extraction objects."""
    return tuple(_to_extraction(row) for row in read_jsonl(path))


def load_scored_fields(path: Path) -> tuple[ScoredField, ...]:
    """Read a scored-fields file back into ScoredField objects."""
    return tuple(ScoredField(**row) for row in read_jsonl(path))


def _to_document(row: dict) -> Document:
    """Rebuild one Document from its JSON form, restoring nested OCR tokens."""
    return Document(
        doc_id=row["doc_id"],
        dataset=row["dataset"],
        split=row["split"],
        image_path=row["image_path"],
        tokens=tuple(OCRToken(text=t["text"], bbox=tuple(t["bbox"]), confidence=t["confidence"])
                     for t in row.get("tokens", [])),
        gold=row.get("gold", {}),
    )


def _to_extraction(row: dict) -> Extraction:
    """Rebuild one Extraction from its JSON form, restoring nested field predictions."""
    return Extraction(
        doc_id=row["doc_id"],
        method=row["method"],
        fields=tuple(FieldPrediction(name=f["name"], value=f["value"], signals=f.get("signals", {}))
                     for f in row.get("fields", [])),
    )
