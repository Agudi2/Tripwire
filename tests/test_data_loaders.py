"""Loaders read each dataset's real on-disk format and fail loudly when it is absent."""

import json
from pathlib import Path

import pytest

from core import paths
from data.loaders import UNASSIGNED, load_cord, load_dataset, load_funsd, load_sroie

LOADERS = {"sroie": load_sroie, "cord": load_cord, "funsd": load_funsd}


@pytest.fixture()
def raw(monkeypatch, tmp_path: Path) -> Path:
    """Point core.paths.RAW at an empty temporary tree."""
    root = tmp_path / "raw"
    root.mkdir()
    monkeypatch.setattr(paths, "RAW", root)
    return root


@pytest.mark.parametrize("dataset", sorted(LOADERS))
def test_missing_raw_data_names_the_directory_to_populate(dataset: str, raw: Path) -> None:
    """The error is actionable: it names the exact absolute directory the user must fill."""
    with pytest.raises(FileNotFoundError) as excinfo:
        LOADERS[dataset]()
    message = str(excinfo.value)
    assert str(raw / dataset) in message
    assert dataset in message.lower()


@pytest.mark.parametrize("dataset", sorted(LOADERS))
def test_present_but_empty_dataset_dir_also_errors(dataset: str, raw: Path) -> None:
    """An existing but empty dataset directory is reported, not silently loaded as zero docs."""
    (raw / dataset).mkdir()
    with pytest.raises(FileNotFoundError):
        LOADERS[dataset]()


def test_load_dataset_dispatches_by_name(raw: Path) -> None:
    """load_dataset routes to the right loader and rejects unknown names."""
    with pytest.raises(FileNotFoundError):
        load_dataset("sroie")
    with pytest.raises(ValueError, match="mnist"):
        load_dataset("mnist")


def test_sroie_reads_boxes_and_entities(raw: Path) -> None:
    """SROIE: img/ + box/ (8-point quads) + entities/ (JSON gold) normalise into a Document."""
    base = raw / "sroie"
    (base / "img").mkdir(parents=True)
    (base / "box").mkdir()
    (base / "entities").mkdir()
    (base / "img" / "X00016469612.jpg").write_bytes(b"")
    (base / "box" / "X00016469612.txt").write_text(
        "72,25,326,25,326,64,72,64,TAN WOON YANN\n"
        "50,82,440,82,440,121,50,121,TOTAL 9.00\n",
        encoding="utf-8",
    )
    (base / "entities" / "X00016469612.txt").write_text(
        json.dumps({"company": "TAN WOON YANN", "date": "25/12/2018",
                    "address": "NO 1 JALAN", "total": "9.00"}),
        encoding="utf-8",
    )

    (doc,) = load_sroie()
    assert doc.doc_id == "sroie-X00016469612"
    assert doc.dataset == "sroie"
    assert doc.image_path.endswith("X00016469612.jpg")
    assert [t.text for t in doc.tokens] == ["TAN", "WOON", "YANN", "TOTAL", "9.00"]
    assert doc.tokens[0].bbox == (72, 25, 326, 64)
    assert doc.gold["total"] == "9.00"
    assert doc.split == UNASSIGNED  # splits.py assigns splits, loaders do not


def test_cord_reads_valid_lines_and_categories(raw: Path) -> None:
    """CORD: json/ valid_line words become tokens; categories become gold fields."""
    base = raw / "cord"
    (base / "json").mkdir(parents=True)
    (base / "image").mkdir()
    (base / "image" / "receipt-00001.png").write_bytes(b"")
    (base / "json" / "receipt-00001.json").write_text(
        json.dumps({
            "valid_line": [
                {"category": "menu.nm",
                 "words": [{"text": "ICE", "quad": {"x1": 10, "y1": 20, "x2": 40, "y2": 20,
                                                    "x3": 40, "y3": 35, "x4": 10, "y4": 35}},
                           {"text": "TEA", "quad": {"x1": 45, "y1": 20, "x2": 80, "y2": 20,
                                                    "x3": 80, "y3": 35, "x4": 45, "y4": 35}}]},
                {"category": "menu.price",
                 "words": [{"text": "3.000", "quad": {"x1": 90, "y1": 20, "x2": 130, "y2": 20,
                                                      "x3": 130, "y3": 35, "x4": 90, "y4": 35}}]},
                {"category": "total.total_price",
                 "words": [{"text": "3.000", "quad": {"x1": 90, "y1": 60, "x2": 130, "y2": 60,
                                                      "x3": 130, "y3": 75, "x4": 90, "y4": 75}}]},
            ]
        }),
        encoding="utf-8",
    )

    (doc,) = load_cord()
    assert doc.doc_id == "cord-receipt-00001"
    assert [t.text for t in doc.tokens] == ["ICE", "TEA", "3.000", "3.000"]
    assert doc.tokens[0].bbox == (10, 20, 40, 35)
    assert doc.gold["menu.nm"] == "ICE TEA"
    assert doc.gold["total.total_price"] == "3.000"
    assert doc.split == UNASSIGNED


def test_funsd_reads_linked_question_answer_pairs(raw: Path) -> None:
    """FUNSD: annotation `linking` turns question/answer entities into gold key-value pairs."""
    base = raw / "funsd"
    (base / "annotations").mkdir(parents=True)
    (base / "images").mkdir()
    (base / "images" / "0000971160.png").write_bytes(b"")
    (base / "annotations" / "0000971160.json").write_text(
        json.dumps({"form": [
            {"id": 0, "text": "DATE:", "box": [10, 10, 60, 25], "label": "question",
             "linking": [[0, 1]], "words": [{"text": "DATE:", "box": [10, 10, 60, 25]}]},
            {"id": 1, "text": "11/15/93", "box": [70, 10, 140, 25], "label": "answer",
             "linking": [[0, 1]], "words": [{"text": "11/15/93", "box": [70, 10, 140, 25]}]},
            {"id": 2, "text": "Registration", "box": [10, 40, 90, 55], "label": "header",
             "linking": [], "words": [{"text": "Registration", "box": [10, 40, 90, 55]}]},
        ]}),
        encoding="utf-8",
    )

    (doc,) = load_funsd()
    assert doc.doc_id == "funsd-0000971160"
    assert [t.text for t in doc.tokens] == ["DATE:", "11/15/93", "Registration"]
    assert doc.tokens[0].bbox == (10, 10, 60, 25)
    assert doc.gold == {"date": "11/15/93"}
    assert doc.split == UNASSIGNED
