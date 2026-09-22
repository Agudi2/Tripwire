"""Loaders for the real SROIE, CORD and FUNSD releases, normalised into `Document`.

Nothing here downloads anything. Each loader reads the dataset's own on-disk format
from `core.paths.RAW` and raises a FileNotFoundError naming the exact directory to
populate when the data is absent.

Expected layouts (flat, no train/test subdirectories -- merge the official splits into
one directory and let `data.splits` do the partitioning, so the split is a property of
the doc_id rather than of how the publisher happened to divide the release):

    data/raw/sroie/img/<id>.jpg
    data/raw/sroie/box/<id>.txt          x1,y1,...,x4,y4,text   (line-level)
    data/raw/sroie/entities/<id>.txt     {"company","date","address","total"}

    data/raw/cord/image/<id>.png
    data/raw/cord/json/<id>.json         {"valid_line":[{"category","words":[{"text","quad"}]}]}

    data/raw/funsd/images/<id>.png
    data/raw/funsd/annotations/<id>.json {"form":[{"id","text","label","words","linking"}]}
"""

from __future__ import annotations

import json
from pathlib import Path

from core import paths
from core.schema import Document, OCRToken
from data.splits import UNASSIGNED

# None of these releases ship per-token OCR confidences: the annotations are human
# transcriptions. Downstream stages must not read this as a real engine score.
NO_CONFIDENCE = 1.0


def load_dataset(dataset: str) -> tuple[Document, ...]:
    """Load one dataset by name, dispatching to its format-specific loader."""
    loaders = {"sroie": load_sroie, "cord": load_cord, "funsd": load_funsd}
    if dataset not in loaders:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {', '.join(loaders)}")
    return loaders[dataset]()


def load_sroie() -> tuple[Document, ...]:
    """Read SROIE receipts: line-level box files plus per-receipt entity JSON."""
    base = paths.RAW / "sroie"
    boxes = _require_files(base / "box", "sroie", "per-receipt box files (<id>.txt)", ".txt")
    entities_dir = _require_dir(base / "entities", "sroie", "entity files (<id>.txt)")

    documents = []
    for box_file in boxes:
        stem = box_file.stem
        gold = _read_json(entities_dir / f"{stem}.txt")
        documents.append(Document(
            doc_id=f"sroie-{stem}",
            dataset="sroie",
            split=UNASSIGNED,
            image_path=str(_image_path(base / "img", stem, (".jpg", ".jpeg", ".png"))),
            tokens=tuple(_sroie_tokens(box_file)),
            gold={key: str(value) for key, value in gold.items() if value},
        ))
    return tuple(documents)


def _sroie_tokens(box_file: Path) -> list[OCRToken]:
    """Split each SROIE line into word tokens sharing that line's bounding box."""
    tokens: list[OCRToken] = []
    for line in box_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        # The text field may itself contain commas, so only the first 8 fields are split.
        parts = line.split(",", 8)
        if len(parts) < 9:
            continue
        bbox = _bbox_from_points([int(float(value)) for value in parts[:8]])
        tokens.extend(OCRToken(text=word, bbox=bbox, confidence=NO_CONFIDENCE)
                      for word in parts[8].split())
    return tokens


def load_cord() -> tuple[Document, ...]:
    """Read CORD receipts: `valid_line` words become tokens, categories become gold fields."""
    base = paths.RAW / "cord"
    annotations = _require_files(base / "json", "cord", "receipt annotations (<id>.json)", ".json")

    documents = []
    for annotation in annotations:
        stem = annotation.stem
        tokens: list[OCRToken] = []
        gold: dict[str, list[str]] = {}
        for line in _read_json(annotation).get("valid_line", []):
            words = [word for word in line.get("words", []) if word.get("text")]
            for word in words:
                tokens.append(OCRToken(text=word["text"],
                                       bbox=_bbox_from_quad(word["quad"]),
                                       confidence=NO_CONFIDENCE))
            category = line.get("category")
            if category:
                gold.setdefault(category, []).extend(word["text"] for word in words)
        documents.append(Document(
            doc_id=f"cord-{stem}",
            dataset="cord",
            split=UNASSIGNED,
            image_path=str(_image_path(base / "image", stem, (".png", ".jpg"))),
            tokens=tuple(tokens),
            gold={key: " ".join(values) for key, values in gold.items() if values},
        ))
    return tuple(documents)


def load_funsd() -> tuple[Document, ...]:
    """Read FUNSD forms, turning annotated question->answer links into gold key/value pairs."""
    base = paths.RAW / "funsd"
    annotations = _require_files(base / "annotations", "funsd", "form annotations (<id>.json)", ".json")

    documents = []
    for annotation in annotations:
        stem = annotation.stem
        entities = _read_json(annotation).get("form", [])
        tokens = tuple(OCRToken(text=word["text"],
                                bbox=_bbox_from_box(word["box"]),
                                confidence=NO_CONFIDENCE)
                       for entity in entities for word in entity.get("words", [])
                       if word.get("text"))
        documents.append(Document(
            doc_id=f"funsd-{stem}",
            dataset="funsd",
            split=UNASSIGNED,
            image_path=str(_image_path(base / "images", stem, (".png", ".jpg"))),
            tokens=tokens,
            gold=_funsd_gold(entities),
        ))
    return tuple(documents)


def _funsd_gold(entities: list[dict]) -> dict[str, str]:
    """Pair each question entity with its linked answer, keyed by the normalised question."""
    by_id = {entity["id"]: entity for entity in entities if "id" in entity}
    gold: dict[str, str] = {}
    for entity in entities:
        if entity.get("label") != "question":
            continue
        key = _normalize_key(entity.get("text", ""))
        if not key:
            continue
        for source, target in entity.get("linking", []):
            answer = by_id.get(target if source == entity["id"] else source, {})
            if answer.get("label") == "answer" and answer.get("text"):
                gold[key] = answer["text"]
                break
    return gold


def _normalize_key(text: str) -> str:
    """Turn a printed form label ("FAX NO:") into a stable field name ("fax_no")."""
    cleaned = "".join(char if char.isalnum() else " " for char in text).split()
    return "_".join(cleaned).lower()


def _bbox_from_points(points: list[int]) -> tuple[int, int, int, int]:
    """Collapse a flat 8-value polygon into an axis-aligned (x0, y0, x1, y1) box."""
    xs, ys = points[0::2], points[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_from_quad(quad: dict) -> tuple[int, int, int, int]:
    """Collapse CORD's {x1,y1,...,x4,y4} quad into an axis-aligned box."""
    xs = [int(quad[f"x{i}"]) for i in range(1, 5)]
    ys = [int(quad[f"y{i}"]) for i in range(1, 5)]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_from_box(box: list[int]) -> tuple[int, int, int, int]:
    """Normalise FUNSD's already axis-aligned [x0, y0, x1, y1] box."""
    x0, y0, x1, y1 = (int(value) for value in box)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _read_json(path: Path) -> dict:
    """Parse one JSON annotation file, with a clear error if it is missing or malformed."""
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON in {path}: {error}") from error


def _image_path(directory: Path, stem: str, extensions: tuple[str, ...]) -> Path:
    """Locate the page image for one document, falling back to the conventional filename."""
    for extension in extensions:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return directory / f"{stem}{extensions[0]}"


def _require_dir(directory: Path, dataset: str, expectation: str) -> Path:
    """Return the directory, or raise an error naming what the user must put where."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No raw {dataset.upper()} data found. Expected {expectation} in:\n"
            f"  {directory}\n"
            f"Download {dataset.upper()} into {paths.RAW / dataset} using the layout documented "
            f"in data/loaders.py, or re-run with --fixtures to use synthetic data instead."
        )
    return directory


def _require_files(directory: Path, dataset: str, expectation: str, suffix: str) -> list[Path]:
    """Return the sorted annotation files in a directory, erroring if there are none."""
    _require_dir(directory, dataset, expectation)
    files = sorted(path for path in directory.iterdir() if path.suffix == suffix)
    if not files:
        raise FileNotFoundError(
            f"No raw {dataset.upper()} data found: {directory} exists but contains no "
            f"{suffix} files. Expected {expectation} there, or re-run with --fixtures."
        )
    return files
