"""OCR + rules baseline: regex and layout heuristics over a Document's tokens.

Every field carries three signals the confidence stage can calibrate on:
`ocr_conf` (mean OCR confidence of the tokens the rule actually consumed), `format_valid`
(the shared validator score), and `match_strength` (how well the heuristic that fired
matches its ideal pattern - a positional prior for the company, keyword quality for the
total). A field the rules cannot find is still emitted, empty, with zero strength, so the
downstream stage sees a review candidate rather than a hole.
"""

from __future__ import annotations

import re

from core.schema import Document, Extraction, FieldPrediction, OCRToken

from extract import validators

# A date printed as 18/03/2018, 2018-03-18, or 18 Mar 2018.
_DATE_RE = re.compile(r"\d{1,4}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}")

# A money amount, with or without cents and thousands separators.
_AMOUNT_RE = re.compile(r"\d[\d,]*(?:\.\d{2})?")

# Keywords that anchor the grand total, and how strongly each one implies it.
_TOTAL_ANCHORS = (("total", 1.0), ("amount", 0.7))

# Lines naming one of these are a different number, never the grand total.
_EXCLUDED_TOTAL_LINES = ("subtotal", "sub total", "sub-total", "tax", "rounding")

# Legal suffixes that confirm a line really is the merchant's registered name.
_COMPANY_SUFFIXES = ("sdn bhd", "bhd", "ltd", "llc", "inc", "enterprise", "trading", "gmbh")

# Reaching one of these ends the address block that follows the company name.
_ADDRESS_STOPWORDS = ("total", "amount", "cash", "change", "invoice", "receipt no")

# Confidence assigned to a total found with no keyword anchoring it at all.
_UNANCHORED_TOTAL_STRENGTH = 0.3


def extract(doc: Document) -> Extraction:
    """Run every field rule over one document and return its baseline Extraction."""
    lines = _lines(doc.tokens)
    company, company_index = _company(lines)
    return Extraction(
        doc_id=doc.doc_id,
        method="rules",
        fields=(company, _date(lines), _address(lines, company_index), _total(lines)),
    )


def line_texts(tokens: tuple[OCRToken, ...]) -> tuple[str, ...]:
    """Reconstruct the page as text lines in reading order; shared with the LLM prompt."""
    return tuple(_text_and_spans(line)[0] for line in _lines(tokens))


def _company(lines: tuple[tuple[OCRToken, ...], ...]) -> tuple[FieldPrediction, int]:
    """Take the topmost line that reads like a name; also report which line it was."""
    for index, line in enumerate(lines):
        text = _text_and_spans(line)[0]
        if validators.validate_company(text) < 1.0:
            continue
        strength = max(0.0, 1.0 - 0.25 * index)
        if any(suffix in text.lower() for suffix in _COMPANY_SUFFIXES):
            strength = min(1.0, strength + 0.25)
        return _prediction("company", text, line, strength), index
    return _missing("company"), -1


def _date(lines: tuple[tuple[OCRToken, ...], ...]) -> FieldPrediction:
    """Take the first date-shaped run of tokens; a nearby `date:` label strengthens it."""
    for line in lines:
        text, spans = _text_and_spans(line)
        match = _DATE_RE.search(text)
        if match is None:
            continue
        used = _tokens_in_span(spans, match.start(), match.end())
        strength = 1.0 if "date" in text.lower() else 0.7
        return _prediction("date", match.group(0), used, strength)
    return _missing("date")


def _address(lines: tuple[tuple[OCRToken, ...], ...], company_index: int) -> FieldPrediction:
    """Take the block of lines under the company name, stopping at the first date or total."""
    if company_index < 0:
        return _missing("address")
    parts: list[str] = []
    used: list[OCRToken] = []
    scores: list[float] = []
    for line in lines[company_index + 1:]:
        text = _text_and_spans(line)[0]
        lowered = text.lower()
        if _DATE_RE.search(text) or any(word in lowered for word in _ADDRESS_STOPWORDS):
            break
        score = validators.validate_address(text)
        if score == 0.0:
            continue
        parts.append(text)
        used.extend(line)
        scores.append(score)
    if not parts:
        return _missing("address")
    return _prediction("address", " ".join(parts), used, sum(scores) / len(scores))


def _total(lines: tuple[tuple[OCRToken, ...], ...]) -> FieldPrediction:
    """Take the amount on the best-anchored total line, falling back to the largest amount."""
    best: tuple[float, str, tuple[OCRToken, ...]] | None = None
    for line in lines:
        text, spans = _text_and_spans(line)
        lowered = text.lower()
        if any(word in lowered for word in _EXCLUDED_TOTAL_LINES):
            continue
        anchor = next((pair for pair in _TOTAL_ANCHORS if pair[0] in lowered), None)
        if anchor is None:
            continue
        matches = list(_AMOUNT_RE.finditer(text))
        if not matches:
            continue
        match = matches[-1]
        used = _tokens_in_span(spans, match.start(), match.end()) + _anchors(line, anchor[0])
        if best is None or anchor[1] > best[0]:
            best = (anchor[1], match.group(0), used)
    if best is not None:
        return _prediction("total", best[1], best[2], best[0])
    return _fallback_total(lines)


def _fallback_total(lines: tuple[tuple[OCRToken, ...], ...]) -> FieldPrediction:
    """With no keyword anywhere, guess the largest token printed with cents, at low strength.

    Requiring cents is what keeps postcodes and item quantities out of the candidate pool.
    """
    candidates = [
        token for line in lines for token in line
        if validators.validate_total(token.text) == 1.0
    ]
    if not candidates:
        return _missing("total")
    best = max(candidates, key=lambda token: float(token.text.replace(",", "")))
    return _prediction("total", best.text, (best,), _UNANCHORED_TOTAL_STRENGTH)


def _prediction(name: str, value: str, tokens, strength: float) -> FieldPrediction:
    """Wrap a found value with the three signals every baseline field reports."""
    return FieldPrediction(
        name=name,
        value=value,
        signals={
            "ocr_conf": _mean_confidence(tokens),
            "format_valid": validators.validate(name, value),
            "match_strength": round(strength, 4),
        },
    )


def _missing(name: str) -> FieldPrediction:
    """Report a field the rules could not locate: empty value, every signal at zero."""
    return FieldPrediction(name=name, value="",
                           signals={"ocr_conf": 0.0, "format_valid": 0.0, "match_strength": 0.0})


def _mean_confidence(tokens) -> float:
    """Average the OCR engine's own confidence over the tokens a rule consumed."""
    tokens = tuple(tokens)
    if not tokens:
        return 0.0
    return round(sum(token.confidence for token in tokens) / len(tokens), 4)


def _lines(tokens: tuple[OCRToken, ...]) -> tuple[tuple[OCRToken, ...], ...]:
    """Group tokens into lines by vertical overlap, each line sorted left to right."""
    ordered = sorted(tokens, key=lambda token: (_centre(token), token.bbox[0]))
    lines: list[tuple[OCRToken, ...]] = []
    current: list[OCRToken] = []
    centre = 0.0
    for token in ordered:
        height = max(1, token.bbox[3] - token.bbox[1])
        if current and abs(_centre(token) - centre) > height / 2:
            lines.append(tuple(sorted(current, key=lambda t: t.bbox[0])))
            current = []
        if not current:
            centre = _centre(token)
        current.append(token)
    if current:
        lines.append(tuple(sorted(current, key=lambda t: t.bbox[0])))
    return tuple(lines)


def _centre(token: OCRToken) -> float:
    """Vertical midpoint of a token's box, used to decide which line it belongs to."""
    return (token.bbox[1] + token.bbox[3]) / 2


def _text_and_spans(line: tuple[OCRToken, ...]) -> tuple[str, tuple[tuple[int, int, OCRToken], ...]]:
    """Join a line into text and record each token's character span within it."""
    spans: list[tuple[int, int, OCRToken]] = []
    cursor = 0
    for token in line:
        spans.append((cursor, cursor + len(token.text), token))
        cursor += len(token.text) + 1
    return " ".join(token.text for token in line), tuple(spans)


def _tokens_in_span(spans, start: int, end: int) -> tuple[OCRToken, ...]:
    """Select the tokens whose characters overlap a regex match inside the joined line."""
    return tuple(token for begin, finish, token in spans if begin < end and finish > start)


def _anchors(line: tuple[OCRToken, ...], keyword: str) -> tuple[OCRToken, ...]:
    """Select the keyword tokens that anchored a match, since their OCR quality matters too."""
    return tuple(token for token in line if keyword in token.text.lower())
