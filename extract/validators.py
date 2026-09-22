"""Per-field format validators.

These do double duty: they gate obviously malformed output, and their score is fed straight
into `FieldPrediction.signals` as `format_valid` for the confidence stage to calibrate on.
Every validator returns a float in [0, 1] - 1.0 for a well-formed value, an intermediate
score for something that has the right shape but does not fully check out, 0.0 for junk.
"""

from __future__ import annotations

import re
from datetime import datetime

# The fixed SROIE field set every extractor in this project emits.
FIELDS: tuple[str, ...] = ("company", "date", "address", "total")

# Formats a receipt date is plausibly printed in; ambiguity between d/m and m/d is fine
# because we only care whether *some* real calendar date is expressed.
_DATE_FORMATS = (
    "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d",
    "%d.%m.%Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%d/%m/%y", "%d-%m-%y",
)

# Three groups separated by the usual date punctuation - shape only, no calendar check.
_DATE_SHAPE = re.compile(r"^\d{1,4}[/.\- ][A-Za-z0-9]{1,9}[/.\- ]\d{2,4}$")

_CURRENCY_PREFIX = re.compile(r"^(?:RM|USD|MYR|\$)\s*", re.IGNORECASE)
_WITH_CENTS = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$|^\d+\.\d{2}$")
_WITHOUT_CENTS = re.compile(r"^\d{1,3}(?:,\d{3})*$|^\d+$")

# An address needs enough characters to be a real one, plus a street number or a separator.
_MIN_ADDRESS_LENGTH = 10


def validate(name: str, value: str) -> float:
    """Score one field value's format, dispatching to the validator for that field name."""
    validator = _VALIDATORS.get(name, _validate_non_empty)
    return validator(value)


def validate_date(value: str) -> float:
    """1.0 if the value is a real calendar date, 0.5 if merely date-shaped, else 0.0."""
    text = value.strip()
    if not text:
        return 0.0
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(text, fmt)
        except ValueError:
            continue
        return 1.0
    return 0.5 if _DATE_SHAPE.match(text) else 0.0


def validate_total(value: str) -> float:
    """1.0 for a currency amount with cents, 0.5 for a bare number, else 0.0."""
    text = _CURRENCY_PREFIX.sub("", value.strip())
    if _WITH_CENTS.match(text):
        return 1.0
    return 0.5 if _WITHOUT_CENTS.match(text) else 0.0


def validate_company(value: str) -> float:
    """1.0 for a name with actual letters in it, 0.0 for blank or punctuation-only values."""
    text = value.strip()
    return 1.0 if len(text) >= 2 and any(char.isalpha() for char in text) else 0.0


def validate_address(value: str) -> float:
    """1.0 for a full street address, 0.5 for a fragment such as a bare city, 0.0 for blank."""
    text = value.strip()
    if not text:
        return 0.0
    has_marker = any(char.isdigit() for char in text) or "," in text
    return 1.0 if has_marker and len(text) >= _MIN_ADDRESS_LENGTH else 0.5


def _validate_non_empty(value: str) -> float:
    """Fallback for fields with no dedicated rule: anything non-blank counts as valid."""
    return 1.0 if value.strip() else 0.0


_VALIDATORS = {
    "company": validate_company,
    "date": validate_date,
    "address": validate_address,
    "total": validate_total,
}
