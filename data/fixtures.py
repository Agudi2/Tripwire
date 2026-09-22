"""Deterministic synthetic documents, so the whole pipeline runs offline with no downloads.

Fixtures exist to exercise the pipeline, not to imitate a real corpus. Two properties
matter. First, determinism: a document's content is derived from `(seed, doc_id)` alone,
so the same seed always yields the same corpus and adding documents never disturbs the
ones already generated. Second, realistic OCR noise: a share of tokens are deliberately
misread, and confidence is drawn from two overlapping ranges rather than two disjoint
ones. Perfectly separable confidence would make the downstream calibration stage look
flawless for reasons that would not survive contact with a real OCR engine.

No image files are produced; `image_path` is a stable placeholder.
"""

from __future__ import annotations

import random
from hashlib import blake2b

from core.schema import Document, OCRToken
from data.splits import UNASSIGNED

DATASETS = ("sroie", "cord", "funsd")

# Page layout in pixels. Crude but enough to give every token a plausible ordered box.
_CHAR_WIDTH, _LINE_HEIGHT, _WORD_GAP, _LINE_GAP, _MARGIN, _TOP = 11, 22, 8, 12, 40, 50

# Character swaps a real OCR engine actually makes on receipt scans.
_CONFUSIONS = {"O": "0", "0": "O", "I": "1", "1": "l", "S": "5", "5": "S",
               "B": "8", "8": "B", "G": "6", "2": "Z", "D": "O", "/": "7"}

_MISREAD_RATE = 0.15  # share of tokens corrupted
_HONEST_RATE = 0.90  # share of misreads that the engine correctly reports as low-confidence
_FALSE_ALARM_RATE = 0.08  # share of correct tokens the engine reports as low-confidence anyway

_COMPANIES = ("TAN WOON YANN SDN BHD", "SEHAT MART", "KEDAI PAPAN YEW CHUAN",
              "GARDENIA BAKERIES", "UNIHAKKA INTERNATIONAL", "MR D.I.Y. TRADING",
              "OJC MARKETING SDN BHD", "SANYU STATIONERY SHOP")
_STREETS = ("JALAN BUKIT TINGGI", "LORONG MELAKA", "JALAN SETIA", "PERSIARAN RAJA",
            "JALAN KENARI", "LEBUH ARMENIAN")
_CITIES = ("KAJANG SELANGOR", "GEORGETOWN PENANG", "JOHOR BAHRU", "PETALING JAYA")
_ITEMS = ("ICE LEMON TEA", "NASI LEMAK", "ROTI CANAI", "KOPI O", "MINERAL WATER",
          "FRIED RICE", "TEH TARIK", "CURRY PUFF")
_FUNSD_LABELS = (("date", "DATE:"), ("to", "TO:"), ("from", "FROM:"), ("subject", "SUBJECT:"),
                 ("fax_no", "FAX NO:"), ("phone_no", "PHONE:"), ("page_count", "PAGES:"),
                 ("company", "COMPANY:"))
_PEOPLE = ("J. R. HAMPTON", "L. K. WONG", "M. ELLIS", "D. SANDERS", "P. NGUYEN")
_SUBJECTS = ("BUDGET REVIEW", "SHIPMENT NOTICE", "PRODUCT TEST", "ANNUAL REPORT")


def generate_documents(dataset: str, count: int, seed: int) -> tuple[Document, ...]:
    """Generate `count` synthetic Documents for one dataset, reproducibly from `seed`."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {', '.join(DATASETS)}")
    return tuple(_generate_one(dataset, f"{dataset}-{index:05d}", seed) for index in range(count))


def _generate_one(dataset: str, doc_id: str, seed: int) -> Document:
    """Build one Document whose content depends only on its own id and the seed."""
    rng = _rng_for(doc_id, seed)
    gold, lines = _CONTENT_BUILDERS[dataset](rng)
    return Document(
        doc_id=doc_id,
        dataset=dataset,
        split=UNASSIGNED,
        image_path=f"fixtures/{dataset}/{doc_id}.png",
        tokens=_lay_out_tokens(lines, rng),
        gold=gold,
    )


def _rng_for(doc_id: str, seed: int) -> random.Random:
    """Seed a private RNG from (seed, doc_id) so documents are independent of corpus size."""
    digest = blake2b(f"{seed}:{doc_id}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _sroie_content(rng: random.Random) -> tuple[dict[str, str], tuple[tuple[str, ...], ...]]:
    """Receipt header fields: company, date, address, total."""
    company = rng.choice(_COMPANIES)
    address = f"NO {rng.randint(1, 99)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    date = _date(rng)
    total = _money(rng, 3.0, 250.0)
    gold = {"company": company, "date": date, "address": address, "total": total}
    lines = (
        tuple(company.split()),
        tuple(address.split()),
        ("DATE:", date),
        ("CASHIER", "COUNTER", str(rng.randint(1, 6))),
        ("TOTAL", total),
        ("CASH", _money(rng, 250.0, 400.0)),
        ("THANK", "YOU", "PLEASE", "COME", "AGAIN"),
    )
    return gold, lines


def _cord_content(rng: random.Random) -> tuple[dict[str, str], tuple[tuple[str, ...], ...]]:
    """Receipt header fields plus itemised lines, as CORD annotates them."""
    gold, header = _sroie_content(rng)
    lines = [header[0], header[1], header[2]]
    for index, name in enumerate(rng.sample(_ITEMS, rng.randint(2, 4)), start=1):
        price = _money(rng, 1.5, 40.0)
        gold[f"item_{index}_name"] = name
        gold[f"item_{index}_price"] = price
        lines.append((*name.split(), price))
    lines.append(("TOTAL", gold["total"]))
    return gold, tuple(lines)


def _funsd_content(rng: random.Random) -> tuple[dict[str, str], tuple[tuple[str, ...], ...]]:
    """Scanned-form key/value pairs, mirroring FUNSD's question/answer links."""
    labels = rng.sample(_FUNSD_LABELS, rng.randint(3, 5))
    gold: dict[str, str] = {}
    lines: list[tuple[str, ...]] = [("FACSIMILE", "TRANSMITTAL", "SHEET")]
    for key, printed in labels:
        value = _funsd_value(key, rng)
        gold[key] = value
        lines.append((printed, *value.split()))
    return gold, tuple(lines)


def _funsd_value(key: str, rng: random.Random) -> str:
    """Pick a plausible answer for one form field."""
    if key == "date":
        return _date(rng)
    if key in ("to", "from"):
        return rng.choice(_PEOPLE)
    if key == "subject":
        return rng.choice(_SUBJECTS)
    if key in ("fax_no", "phone_no"):
        return f"{rng.randint(200, 989)}-{rng.randint(200, 989)}-{rng.randint(1000, 9999)}"
    if key == "page_count":
        return str(rng.randint(1, 12))
    return rng.choice(_COMPANIES)


_CONTENT_BUILDERS = {"sroie": _sroie_content, "cord": _cord_content, "funsd": _funsd_content}


def _date(rng: random.Random) -> str:
    """A receipt-style DD/MM/YYYY date."""
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2017, 2019)}"


def _money(rng: random.Random, low: float, high: float) -> str:
    """A two-decimal amount as it would be printed."""
    return f"{rng.uniform(low, high):.2f}"


def _lay_out_tokens(lines: tuple[tuple[str, ...], ...], rng: random.Random) -> tuple[OCRToken, ...]:
    """Place each word on the page as an OCRToken, corrupting some of them as OCR would."""
    tokens: list[OCRToken] = []
    y = _TOP
    for line in lines:
        x = _MARGIN
        for word in line:
            width = _CHAR_WIDTH * len(word) + _WORD_GAP
            text, misread = _maybe_misread(word, rng)
            tokens.append(OCRToken(text=text,
                                   bbox=(x, y, x + width, y + _LINE_HEIGHT),
                                   confidence=_confidence(misread, rng)))
            x += width + _WORD_GAP
        y += _LINE_HEIGHT + _LINE_GAP
    return tuple(tokens)


def _maybe_misread(word: str, rng: random.Random) -> tuple[str, bool]:
    """Return the word as OCR "read" it, and whether it was corrupted."""
    if rng.random() >= _MISREAD_RATE:
        return word, False
    candidates = [i for i, char in enumerate(word) if char in _CONFUSIONS]
    if not candidates:
        return word[:-1] if len(word) > 2 else word + "l", True  # dropped/added stroke
    index = rng.choice(candidates)
    return word[:index] + _CONFUSIONS[word[index]] + word[index + 1:], True


def _confidence(misread: bool, rng: random.Random) -> float:
    """Draw an engine confidence whose low and high ranges deliberately overlap.

    Misreads are usually but not always reported as low-confidence, and a few correct
    tokens are flagged anyway, so the downstream stage faces a genuine calibration
    problem instead of a threshold that separates the two perfectly.
    """
    flagged = rng.random() < (_HONEST_RATE if misread else _FALSE_ALARM_RATE)
    span = (0.21, 0.59) if flagged else (0.61, 0.99)
    return round(rng.uniform(*span), 4)
